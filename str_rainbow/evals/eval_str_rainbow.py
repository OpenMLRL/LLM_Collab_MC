#!/usr/bin/env python3
"""
Evaluation script for the str_rainbow (Minecraft string builder) task.

Design goals:
- Mirror training defaults (model, prompt formatting, blocks, command limits).
- Support 1 or 2 agents and multi-turn rollouts that reuse the external feedback
  logic from training (`perfect_feedback` / `position_feedback`).
- Primary metric is IoU (covered / (target + extra)), not Pass@k.

Outputs:
- CSV with per-attempt metrics.
-.JSON summary with aggregated statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

# ---------------------------------------------------------------------------
# Imports from the project (match training script path logic)
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_DIR)
REPO_PARENT = os.path.dirname(REPO_ROOT)
for p in (REPO_PARENT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from LLM_Collab_MC.str_rainbow.evals.eval_common import (
    as_block_list,
    build_command_limits,
    build_context_resolver,
    combine_commands,
    compute_iou,
    generate_completion,
    prepare_prompt_parts,
    simulate_and_score,
    task_to_item_dict,
)
from LLM_Collab_MC.str_rainbow.external import get_external_transition, set_context_resolver
from LLM_Collab_MC.str_rainbow.train.train_magrpo import _build_formatters, _map_dtype, _slice_items
from LLM_Collab_MC.str_rainbow.utils.config import apply_overrides, load_yaml, resolve_path
from LLM_Collab_MC.str_rainbow.utils.prompting import apply_prompt_defaults
from LLM_Collab_MC.str_rainbow.utils.str_rainbow import TaskSpec, load_tasks_from_csv


def evaluate_str_rainbow(cfg: Dict[str, Any], args_ns: argparse.Namespace, *, run_label: str = "eval") -> Dict[str, Any]:
    """Run evaluation and return summary dict."""
    apply_prompt_defaults(cfg)

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    csv_path = resolve_path(args_ns.config, args_ns.csv_path or dataset_cfg.get("csv_path"))
    spacing = int(dataset_cfg.get("spacing") or 1)
    local_z = int(dataset_cfg.get("local_z") or 0)

    # Load tasks and slice split
    tasks = load_tasks_from_csv(csv_path, spacing=spacing, local_z=local_z)
    eval_split = args_ns.eval_split or dataset_cfg.get("eval_split") or "[:]"
    task_items = _slice_items(tasks, eval_split)
    if not task_items:
        raise ValueError(f"No tasks after applying eval_split={eval_split}")

    # Model and generation settings
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

    # Eval hyperparameters (fall back to training magrpo for defaults)
    eval_cfg = cfg.get("evaluation") or {}
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}
    magrpo_cfg = cfg.get("magrpo") or {}
    if not isinstance(magrpo_cfg, dict):
        magrpo_cfg = {}

    num_agents = int(args_ns.num_agents or eval_cfg.get("num_agents") or magrpo_cfg.get("num_agents") or 1)
    num_turns = int(args_ns.num_turns or eval_cfg.get("num_turns") or magrpo_cfg.get("num_turns") or 1)
    num_samples = int(args_ns.num_samples or eval_cfg.get("num_samples") or eval_cfg.get("num_attempts") or 1)
    max_new_tokens = int(args_ns.max_new_tokens or eval_cfg.get("max_new_tokens") or magrpo_cfg.get("max_new_tokens") or 256)
    temperature = float(args_ns.temperature or eval_cfg.get("temperature") or model_cfg.get("temperature") or 0.6)
    top_p = float(args_ns.top_p or eval_cfg.get("top_p") or model_cfg.get("top_p") or 0.6)

    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    max_commands_total = int(task_cfg.get("max_commands") or 600)
    allowed_blocks_agent1 = as_block_list(task_cfg.get("block_agent1"))
    if not allowed_blocks_agent1:
        allowed_blocks_agent1 = ["black_concrete", "white_concrete"]
    allowed_blocks_agent2 = as_block_list(task_cfg.get("block_agent2"))
    allowed_blocks_per_agent = [allowed_blocks_agent1]
    if num_agents >= 2:
        allowed_blocks_per_agent.append(allowed_blocks_agent2 if allowed_blocks_agent2 else allowed_blocks_agent1)
    command_limits = build_command_limits(num_agents, max_commands_total)

    external_cfg = cfg.get("external") or {}
    if not isinstance(external_cfg, dict):
        external_cfg = {}
    external_mode = str(external_cfg.get("mode") or "position_feedback")
    external_original_prompt = bool(external_cfg.get("original_prompt", True))
    external_previous_response = bool(external_cfg.get("previous_response", False))
    external_enabled = bool(external_cfg.get("enabled", False))

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
    output_base_dir = args_ns.output_dir or output_cfg.get("base_dir") or os.path.join(PROJECT_DIR, "evals", "results")
    verbose = bool(args_ns.verbose if args_ns.verbose is not None else output_cfg.get("verbose", False))

    # Tokenizer and per-agent models (mirror training: separate instances)
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

    # Prompt prep
    prompt_parts = prepare_prompt_parts(cfg)
    context_map = build_context_resolver(task_items, prompt_parts, max_commands_total=max_commands_total, set_resolver_fn=set_context_resolver)
    formatters = _build_formatters(cfg, num_agents=num_agents, tokenizer=tokenizer)

    # Evaluation storage
    all_results: List[Dict[str, Any]] = []
    best_iou_per_task: Dict[str, float] = {}

    for task_idx, task in enumerate(task_items):
        if verbose:
            print(f"\n{'='*60}\nTask {task_idx+1}/{len(task_items)}: {task.task_id} ({task.text})")

        task_prompt_key = f"str_rainbow:{task.task_id}"
        task_item_dict = task_to_item_dict(task)

        for attempt in range(num_samples):
            if verbose:
                print(f"  Attempt {attempt+1}/{num_samples}")

            prompt_history: List[List[str]] = [[] for _ in range(num_agents)]
            response_history: List[List[str]] = [[] for _ in range(num_agents)]
            completions_per_turn: List[List[str]] = []
            tokens_per_turn: List[List[int]] = []
            times_per_turn: List[List[float]] = []

            # Initial prompts
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
                    if external_enabled:
                        current_prompts = list(
                            get_external_transition(
                                prompt=task_prompt_key,
                                agent_completions=turn_completions,
                                num_agents=num_agents,
                                mode=external_mode,
                                original_prompt=external_original_prompt,
                                previous_response=external_previous_response,
                                prompt_history_per_agent=prompt_history,
                                response_history_per_agent=response_history,
                            )
                        )
                    else:
                        def _refine_prompt(base: str, prior: List[str], agent_idx: int) -> str:
                            prior_text = "\n".join(
                                f"--- Agent {i+1} previous commands ---\n{(p or '').strip()}"
                                for i, p in enumerate(prior)
                                if (p or "").strip()
                            )
                            guidance = (
                                "\n\nRefine your /setblock commands to better match the target."
                                " You may overwrite earlier commands."
                                f" Max commands allowed: {command_limits[agent_idx]}."
                                "\nOutput ONLY Minecraft commands, one per line (no markdown)."
                            )
                            return f"{base.rstrip()}\n\n{prior_text}{guidance}".strip()

                        current_prompts = [
                            _refine_prompt(base_prompts[idx], turn_completions, idx) for idx in range(num_agents)
                        ]

            # Normalize commands and score
            merged_commands, accepted_counts = combine_commands(
                completions_per_turn=completions_per_turn,
                task=task,
                allowed_blocks_agent1=allowed_blocks_per_agent[0],
                allowed_blocks_agent2=allowed_blocks_per_agent[1] if len(allowed_blocks_per_agent) > 1 else allowed_blocks_per_agent[0],
                max_commands_total=max_commands_total,
                num_agents=num_agents,
            )

            metrics = simulate_and_score(
                merged_commands=merged_commands,
                task=task,
                allowed_blocks_per_agent=allowed_blocks_per_agent,
            )
            iou = compute_iou(metrics)

            agent_tokens_total = [sum(t[agent_idx] for t in tokens_per_turn) for agent_idx in range(num_agents)]
            agent_times_total = [sum(t[agent_idx] for t in times_per_turn) for agent_idx in range(num_agents)]
            turn_times = [max(t) for t in times_per_turn]  # agents run "in parallel" per turn
            total_time = sum(turn_times)

            result = {
                "task_id": task.task_id,
                "text": task.text,
                "difficulty": task.difficulty,
                "attempt_id": attempt,
                "num_agents": num_agents,
                "num_turns": num_turns,
                "iou": round(iou, 4),
                "coverage_ratio": round(float(metrics.get("coverage_ratio", 0.0)), 4),
                "extra_ratio": round(float(metrics.get("extra_ratio", 0.0)), 4),
                "adjacent_same_color_ratio": round(float(metrics.get("adjacent_same_color_ratio", 0.0)), 4),
                "reward": round(float(metrics.get("score_total", metrics.get("score_mean", 0.0))), 4),
                "target_total": int(metrics.get("target_total", 0)),
                "covered": int(metrics.get("covered", 0)),
                "extra_blocks": int(metrics.get("extra_blocks", 0)),
                "missing": int(metrics.get("missing", 0)),
                "commands_agent1": accepted_counts[0] if accepted_counts else 0,
                "commands_agent2": accepted_counts[1] if len(accepted_counts) > 1 else 0,
                "tokens_agent1_total": agent_tokens_total[0] if agent_tokens_total else 0,
                "tokens_agent2_total": agent_tokens_total[1] if len(agent_tokens_total) > 1 else 0,
                "time_agent1_s": round(agent_times_total[0] if agent_times_total else 0.0, 4),
                "time_agent2_s": round(agent_times_total[1] if len(agent_times_total) > 1 else 0.0, 4),
                "total_time_s": round(total_time, 4),
                "total_tokens": sum(agent_tokens_total),
                "max_commands_total": max_commands_total,
                "external_mode": external_mode,
            }
            all_results.append(result)
            best_iou = best_iou_per_task.get(task.task_id, 0.0)
            if iou > best_iou:
                best_iou_per_task[task.task_id] = iou

            if verbose:
                print(
                    f"    IoU={iou:.4f} | coverage={metrics.get('coverage_ratio', 0.0):.3f} "
                    f"| extra={metrics.get('extra_ratio', 0.0):.3f} | reward={metrics.get('score_total', 0.0):.3f}"
                )

    if not all_results:
        raise ValueError("No results generated.")

    # Aggregation
    def _avg(key: str) -> float:
        return mean(float(r.get(key, 0.0)) for r in all_results)

    aggregated = {
        "avg_iou": round(_avg("iou"), 4),
        "avg_coverage": round(_avg("coverage_ratio"), 4),
        "avg_extra_ratio": round(_avg("extra_ratio"), 4),
        "avg_adjacent_same_color_ratio": round(_avg("adjacent_same_color_ratio"), 4),
        "avg_reward": round(_avg("reward"), 4),
        "avg_total_time": round(_avg("total_time_s"), 4),
        "avg_total_tokens": round(_avg("total_tokens"), 2),
        "avg_commands_agent1": round(mean(r.get("commands_agent1", 0) for r in all_results), 2),
        "avg_commands_agent2": round(mean(r.get("commands_agent2", 0) for r in all_results), 2),
        "best_iou_per_task": round(mean(best_iou_per_task.values()), 4),
        "num_tasks": len(task_items),
        "num_attempts_per_task": num_samples,
    }

    # Persist outputs
    os.makedirs(output_base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    label_safe = str(run_label or "eval").strip().lower().replace(" ", "_")
    results_csv_path = os.path.join(output_base_dir, f"str_rainbow_{label_safe}_{timestamp}.csv")
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
            "csv_path": csv_path,
            "eval_split": eval_split,
            "spacing": spacing,
            "local_z": local_z,
        },
        "hyperparameters": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "num_samples": num_samples,
            "external_mode": external_mode,
        },
        "timestamp": timestamp,
        "aggregated": aggregated,
    }
    json_path = os.path.join(output_base_dir, f"str_rainbow_{label_safe}_summary_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to: {json_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("STR RAINBOW EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Tasks: {len(task_items)} | Attempts per task: {num_samples}")
    print(f"Agents: {num_agents} | Turns: {num_turns} | External: {external_mode}")
    print(f"Avg IoU: {aggregated['avg_iou']:.4f} (best-per-task {aggregated['best_iou_per_task']:.4f})")
    print(f"Avg Coverage: {aggregated['avg_coverage']:.4f} | Avg Extra Ratio: {aggregated['avg_extra_ratio']:.4f}")
    print(f"Avg Reward: {aggregated['avg_reward']:.4f}")
    print(f"Avg Total Time: {aggregated['avg_total_time']:.2f}s | Avg Total Tokens: {aggregated['avg_total_tokens']:.1f}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate str_rainbow (Minecraft string builder) with IoU metric.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_DIR, "evals", "configs", "str_rainbow_eval.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="HF model name (overrides config).")
    parser.add_argument("--csv-path", type=str, default=None, help="Dataset CSV path (overrides config).")
    parser.add_argument("--eval-split", type=str, default=None, help="Slice expression, e.g., '[:32]'.")
    parser.add_argument("--num-turns", type=int, default=None, help="Number of turns (overrides config).")
    parser.add_argument("--num-agents", type=int, default=None, help="Number of agents (overrides config).")
    parser.add_argument("--num-samples", type=int, default=None, help="Attempts per task (overrides config).")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens per generation.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to store results.")
    parser.add_argument("--verbose", action="store_true", default=None, help="Verbose logging.")
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

    evaluate_str_rainbow(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
