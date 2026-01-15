#!/usr/bin/env python3
"""
Evaluation script for box_builder (Minecraft box building task).

Modes supported via run_label:
- parallel: 2 agents, 1 turn (no feedback)
- discussion: 2 agents, 2 turns, cross-visibility refine (no external)
- single_agent: 1 agent, 1 turn

Metrics:
- IoU (from score_box_builder)
- Time (per attempt, summed per-turn max for multi-agent)
- Tokens (per attempt, total new tokens)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List

import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_DIR)
REPO_PARENT = os.path.dirname(REPO_ROOT)
for p in (REPO_PARENT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from LLM_Collab_MC.box_builder.evals.eval_common import (  # noqa: E402
    allowed_blocks_for_agents,
    build_command_limits,
    combine_commands,
    compute_resource_limits_for_task,
    generate_completion,
    simulate_and_score,
    task_to_item_dict,
)
from LLM_Collab_MC.box_builder.train.train_magrpo import _build_formatters, _map_dtype, _slice_items  # noqa: E402
from LLM_Collab_MC.box_builder.utils.config import apply_overrides, load_yaml, resolve_path  # noqa: E402
from LLM_Collab_MC.box_builder.utils.prompting import apply_prompt_defaults  # noqa: E402
from LLM_Collab_MC.box_builder.utils.box_builder import TaskSpec, load_tasks_from_json  # noqa: E402


def evaluate_box_builder(cfg: Dict[str, Any], args_ns: argparse.Namespace, *, run_label: str = "eval") -> Dict[str, Any]:
    apply_prompt_defaults(cfg)

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    json_path = resolve_path(args_ns.config, args_ns.json_path or dataset_cfg.get("json_path"))
    eval_split = args_ns.eval_split or dataset_cfg.get("eval_split") or "[:]"

    # Load tasks
    tasks = load_tasks_from_json(json_path)
    task_items = _slice_items(tasks, eval_split)
    if not task_items:
        raise ValueError(f"No tasks after applying eval_split={eval_split}")

    # Model settings
    model_cfg = cfg.get("model") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model_name = args_ns.model_name or model_cfg.get("name") or ""
    if not model_name:
        raise ValueError("model.name is required")
    tokenizer_kwargs = model_cfg.get("tokenizer_kwargs") or {}
    if not isinstance(tokenizer_kwargs, dict):
        tokenizer_kwargs = {}
    model_kwargs = model_cfg.get("model_kwargs") or {}
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}
    dtype = _map_dtype(model_cfg.get("dtype") or model_cfg.get("torch_dtype") or model_kwargs.get("torch_dtype"))
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    # Eval hyperparameters
    eval_cfg = cfg.get("evaluation") or {}
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}
    magrpo_cfg = cfg.get("magrpo") or {}
    if not isinstance(magrpo_cfg, dict):
        magrpo_cfg = {}

    num_agents = int(getattr(args_ns, "num_agents", None) or eval_cfg.get("num_agents") or magrpo_cfg.get("num_agents") or 1)
    num_turns = int(getattr(args_ns, "num_turns", None) or eval_cfg.get("num_turns") or magrpo_cfg.get("num_turns") or 1)
    num_samples = int(getattr(args_ns, "num_samples", None) or eval_cfg.get("num_samples") or 1)
    max_new_tokens = int(getattr(args_ns, "max_new_tokens", None) or eval_cfg.get("max_new_tokens") or magrpo_cfg.get("max_new_tokens") or 256)
    temperature = float(getattr(args_ns, "temperature", None) or eval_cfg.get("temperature") or model_cfg.get("temperature") or 0.6)
    top_p = float(getattr(args_ns, "top_p", None) or eval_cfg.get("top_p") or model_cfg.get("top_p") or 0.6)
    base_seed = int(getattr(args_ns, "seed", None) or time.time())

    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    max_commands_total = int(task_cfg.get("max_commands") or 600)
    limited_resource = bool(task_cfg.get("limited_resource", False))
    block_agent1_over = task_cfg.get("block_agent1") or []
    block_agent2_over = task_cfg.get("block_agent2") or []
    command_limits = build_command_limits(num_agents, max_commands_total)

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
    output_base_dir = args_ns.output_dir or output_cfg.get("base_dir") or os.path.join(PROJECT_DIR, "evals", "results")
    verbose = bool(args_ns.verbose if args_ns.verbose is not None else output_cfg.get("verbose", False))

    # Tokenizer + models
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {num_agents} model instance(s): {model_name}")
    agents: List[AutoModelForCausalLM] = []
    for idx in range(num_agents):
        print(f"  - Agent {idx+1}")
        agent = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        agent.eval()
        agents.append(agent)

    formatters = _build_formatters(cfg, num_agents=num_agents, tokenizer=tokenizer)

    all_results: List[Dict[str, Any]] = []
    best_iou_per_task: Dict[str, float] = {}

    for task_idx, task in enumerate(task_items):
        if verbose:
            print(f"\n{'='*60}\nTask {task_idx+1}/{len(task_items)}: {task.task_id}")

        task_item_dict = task_to_item_dict(task)
        allowed_blocks_per_agent = allowed_blocks_for_agents(task, block_agent1_over, block_agent2_over, num_agents)
        resource_limits = compute_resource_limits_for_task(task, num_agents, limited_resource)

        for attempt in range(num_samples):
            if verbose:
                print(f"  Attempt {attempt+1}/{num_samples}")
            attempt_seed = base_seed + attempt

            prompt_history: List[List[str]] = [[] for _ in range(num_agents)]
            response_history: List[List[str]] = [[] for _ in range(num_agents)]
            completions_per_turn: List[List[str]] = []
            tokens_per_turn: List[List[int]] = []
            times_per_turn: List[List[float]] = []

            current_prompts = [fmt(task_item_dict) if callable(fmt) else "" for fmt in formatters]
            base_prompts = list(current_prompts)

            for turn_idx in range(num_turns):
                turn_completions: List[str] = []
                turn_tokens: List[int] = []
                turn_times: List[float] = []

                for agent_idx, agent in enumerate(agents):
                    prompt_text = current_prompts[agent_idx] if agent_idx < len(current_prompts) else ""
                    completion, gen_time, tok_count = generate_completion(
                        agent,
                        tokenizer,
                        prompt_text,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=attempt_seed * 100 + turn_idx * 10 + agent_idx,
                    )
                    turn_completions.append(completion)
                    turn_tokens.append(tok_count)
                    turn_times.append(gen_time)
                    prompt_history[agent_idx].append(prompt_text)
                    response_history[agent_idx].append(completion)

                completions_per_turn.append(turn_completions)
                tokens_per_turn.append(turn_tokens)
                times_per_turn.append(turn_times)

                if turn_idx < num_turns - 1:
                    def _refine_prompt(base: str, prior: List[str], agent_idx: int) -> str:
                        prior_text = "\n".join(
                            f"--- Agent {i+1} previous commands ---\n{(p or '').strip()}"
                            for i, p in enumerate(prior)
                            if (p or "").strip()
                        )
                        guidance = (
                            "\n\nRefine your /fill commands to better match the target."
                            " You may overwrite earlier commands."
                            f" Max commands allowed: {command_limits[agent_idx]}."
                            "\nOutput ONLY Minecraft commands, one per line (no markdown)."
                        )
                        return f"{base.rstrip()}\n\n{prior_text}{guidance}".strip()

                    current_prompts = [
                        _refine_prompt(base_prompts[idx], turn_completions, idx) for idx in range(num_agents)
                    ]

            # Normalize and score
            merged_commands, accepted_counts = combine_commands(
                completions_per_turn=completions_per_turn,
                task=task,
                allowed_blocks_agent1=allowed_blocks_per_agent[0],
                allowed_blocks_agent2=allowed_blocks_per_agent[1] if len(allowed_blocks_per_agent) > 1 else allowed_blocks_per_agent[0],
                max_commands_total=max_commands_total,
                num_agents=num_agents,
                resource_limits=resource_limits,
            )

            metrics = simulate_and_score(
                merged_commands=merged_commands,
                task=task,
            )
            iou = float(metrics.get("iou", 0.0))
            reward = float(metrics.get("score_mean", 0.0))

            agent_tokens_total = [sum(t[agent_idx] for t in tokens_per_turn) for agent_idx in range(num_agents)]
            agent_times_total = [sum(t[agent_idx] for t in times_per_turn) for agent_idx in range(num_agents)]
            turn_times = [max(t) for t in times_per_turn]
            total_time = sum(turn_times)

            result = {
                "task_id": task.task_id,
                "attempt_id": attempt,
                "num_agents": num_agents,
                "num_turns": num_turns,
                "iou": round(iou, 4),
                "score_match": round(float(metrics.get("score_match", 0.0)), 4),
                "reward": round(reward, 4),
                "commands_agent1": accepted_counts[0] if accepted_counts else 0,
                "commands_agent2": accepted_counts[1] if len(accepted_counts) > 1 else 0,
                "tokens_agent1_total": agent_tokens_total[0] if agent_tokens_total else 0,
                "tokens_agent2_total": agent_tokens_total[1] if len(agent_tokens_total) > 1 else 0,
                "time_agent1_s": round(agent_times_total[0] if agent_times_total else 0.0, 4),
                "time_agent2_s": round(agent_times_total[1] if len(agent_times_total) > 1 else 0.0, 4),
                "total_time_s": round(total_time, 4),
                "total_tokens": sum(agent_tokens_total),
                "max_commands_total": max_commands_total,
            }
            all_results.append(result)
            best_iou = best_iou_per_task.get(task.task_id, 0.0)
            if iou > best_iou:
                best_iou_per_task[task.task_id] = iou

            if verbose:
                print(
                    f"    IoU={iou:.4f} | match={metrics.get('score_match', 0.0):.3f} | reward={reward:.3f} "
                    f"| time={total_time:.2f}s | tokens={sum(agent_tokens_total)}"
                )

    if not all_results:
        raise ValueError("No results generated.")

    def _avg(key: str) -> float:
        return mean(float(r.get(key, 0.0)) for r in all_results)

    aggregated = {
        "avg_iou": round(_avg("iou"), 4),
        "avg_score_match": round(_avg("score_match"), 4),
        "avg_reward": round(_avg("reward"), 4),
        "avg_total_time": round(_avg("total_time_s"), 4),
        "avg_total_tokens": round(_avg("total_tokens"), 2),
        "avg_commands_agent1": round(mean(r.get("commands_agent1", 0) for r in all_results), 2),
        "avg_commands_agent2": round(mean(r.get("commands_agent2", 0) for r in all_results), 2),
        "best_iou_per_task": round(mean(best_iou_per_task.values()), 4),
        "num_tasks": len(task_items),
        "num_attempts_per_task": num_samples,
    }

    os.makedirs(output_base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    label_safe = str(run_label or "eval").strip().lower().replace(" ", "_")
    results_csv_path = os.path.join(output_base_dir, f"box_builder_{label_safe}_{timestamp}.csv")
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved to: {results_csv_path}")

    summary = {
        "config": label_safe,
        "model": model_name,
        "num_agents": num_agents,
        "num_turns": num_turns,
        "dataset": {
            "json_path": json_path,
            "eval_split": eval_split,
        },
        "hyperparameters": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "num_samples": num_samples,
        },
        "timestamp": timestamp,
        "aggregated": aggregated,
    }
    summary_path = os.path.join(output_base_dir, f"box_builder_{label_safe}_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("BOX_BUILDER EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Tasks: {len(task_items)} | Attempts per task: {num_samples}")
    print(f"Avg IoU: {aggregated['avg_iou']:.4f} (best-per-task {aggregated['best_iou_per_task']:.4f})")
    print(f"Avg Reward: {aggregated['avg_reward']:.4f} | Avg Time: {aggregated['avg_total_time']:.2f}s | Avg Tokens: {aggregated['avg_total_tokens']:.1f}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate box_builder (Minecraft box building) with IoU metric.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_DIR, "evals", "configs", "parallel_config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="HF model name (overrides config).")
    parser.add_argument("--json-path", type=str, default=None, help="Dataset JSON path (overrides config).")
    parser.add_argument("--eval-split", type=str, default=None, help="Slice expression, e.g., '[:32]'.")
    parser.add_argument("--num-turns", type=int, default=None, help="Number of turns (overrides config).")
    parser.add_argument("--num-agents", type=int, default=None, help="Number of agents (overrides config).")
    parser.add_argument("--num-samples", type=int, default=None, help="Attempts per task (overrides config).")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens per generation.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to store results.")
    parser.add_argument("--verbose", action="store_true", default=None, help="Verbose logging.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for sampling (different per attempt).")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="Config overrides in a.b.c=value format (comma-separated or space-separated).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.override:
        override_items: List[str] = []
        for item in args.override:
            if item is None:
                continue
            for part in str(item).split(","):
                part = part.strip()
                if part:
                    override_items.append(part)
        if override_items:
            cfg = apply_overrides(cfg, ",".join(override_items))

    evaluate_box_builder(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
