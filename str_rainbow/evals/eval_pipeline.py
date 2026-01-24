#!/usr/bin/env python3
"""
Pipeline evaluation for str_rainbow (Agent 1 then Agent 2 sees Agent 1 commands).
Sequential time (sum), IoU metric, configurable num_turns (agents alternate, each sees prior commands).
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

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_DIR)
REPO_PARENT = os.path.dirname(REPO_ROOT)
for p in (REPO_PARENT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from LLM_Collab_MC.str_rainbow.evals.eval_common import (
    as_block_list,
    combine_commands,
    compute_iou,
    generate_completion,
    prepare_prompt_parts,
    simulate_and_score,
    task_to_item_dict,
)
from LLM_Collab_MC.str_rainbow.train.train_magrpo import _build_formatters, _map_dtype, _slice_items
from LLM_Collab_MC.str_rainbow.utils.config import apply_overrides, load_yaml, resolve_path
from LLM_Collab_MC.str_rainbow.utils.prompting import apply_prompt_defaults
from LLM_Collab_MC.str_rainbow.utils.str_rainbow import TaskSpec, load_tasks_from_csv


def _format_followup_prompt(base_prompt: str, prior_commands: List[str], max_commands: int) -> str:
    """Append prior commands so the next agent/turn can refine."""
    prior_lines = "\n".join(cmd.strip() for cmd in prior_commands if (cmd or "").strip())
    if prior_lines:
        return (
            f"{base_prompt.rstrip()}\n\n"
            "Commands already placed (do not repeat unless fixing):\n"
            f"{prior_lines}\n"
            "\nContinue with your /setblock commands to complete the target. "
            f"Max commands allowed: {max_commands}\n"
            "Output ONLY Minecraft commands, one per line (no markdown)."
        )
    return base_prompt.rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline evaluation for str_rainbow (IoU metric).")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_DIR, "evals", "configs", "pipeline_config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="HF model name (overrides config).")
    parser.add_argument("--csv-path", type=str, default=None, help="Dataset CSV path (overrides config).")
    parser.add_argument("--eval-split", type=str, default=None, help="Slice expression, e.g., '[:32]'.")
    parser.add_argument("--num-samples", type=int, default=None, help="Attempts per task (overrides config).")
    parser.add_argument("--num-turns", type=int, default=None, help="Number of turns (overrides config).")
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


def evaluate_pipeline(cfg: Dict[str, Any], args_ns: argparse.Namespace) -> Dict[str, Any]:
    apply_prompt_defaults(cfg)

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    csv_path = resolve_path(args_ns.config, args_ns.csv_path or dataset_cfg.get("csv_path"))
    spacing = int(dataset_cfg.get("spacing") or 1)
    local_z = int(dataset_cfg.get("local_z") or 0)

    tasks = load_tasks_from_csv(csv_path, spacing=spacing, local_z=local_z)
    eval_split = args_ns.eval_split or dataset_cfg.get("eval_split") or "[:]"
    task_items = _slice_items(tasks, eval_split)
    if not task_items:
        raise ValueError(f"No tasks after applying eval_split={eval_split}")

    # Model config
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
    num_samples = int(args_ns.num_samples or eval_cfg.get("num_samples") or 1)
    num_turns = int(args_ns.num_turns or eval_cfg.get("num_turns") or 2)
    if num_turns < 1:
        raise ValueError("num_turns must be >= 1 for pipeline evaluation")
    max_new_tokens = int(args_ns.max_new_tokens or eval_cfg.get("max_new_tokens") or 512)
    temperature = float(args_ns.temperature or eval_cfg.get("temperature") or model_cfg.get("temperature") or 0.6)
    top_p = float(args_ns.top_p or eval_cfg.get("top_p") or model_cfg.get("top_p") or 0.6)

    # Task config
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    max_commands_total = int(task_cfg.get("max_commands") or 600)
    allowed_blocks_agent1 = as_block_list(task_cfg.get("block_agent1"))
    if not allowed_blocks_agent1:
        allowed_blocks_agent1 = ["black_concrete", "white_concrete"]
    allowed_blocks_agent2 = as_block_list(task_cfg.get("block_agent2"))
    allowed_blocks_per_agent = [allowed_blocks_agent1, allowed_blocks_agent2 if allowed_blocks_agent2 else allowed_blocks_agent1]

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
    output_base_dir = args_ns.output_dir or output_cfg.get("base_dir") or os.path.join(PROJECT_DIR, "evals", "results")
    verbose = bool(args_ns.verbose if args_ns.verbose is not None else output_cfg.get("verbose", False))

    # Tokenizer + models (2 agents)
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model instances: {model_name}")
    agent1 = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    agent1.eval()
    agent2 = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    agent2.eval()

    prompt_parts = prepare_prompt_parts(cfg)
    formatters = _build_formatters(cfg, num_agents=2, tokenizer=tokenizer)

    all_results: List[Dict[str, Any]] = []
    best_iou_per_task: Dict[str, float] = {}

    for task_idx, task in enumerate(task_items):
        if verbose:
            print(f"\n{'='*60}\nTask {task_idx+1}/{len(task_items)}: {task.task_id} ({task.text})")
        task_item = task_to_item_dict(task)

        base_prompt_agent1 = formatters[0](task_item)
        base_prompt_agent2 = formatters[1](task_item)
        base_prompts = [base_prompt_agent1, base_prompt_agent2]

        for attempt in range(num_samples):
            if verbose:
                print(f"  Attempt {attempt+1}/{num_samples}")

            history_commands: List[str] = []
            completions_per_turn: List[List[str]] = []
            tokens_per_turn: List[List[int]] = []
            times_per_turn: List[List[float]] = []
            current_prompts = list(base_prompts)

            for turn_idx in range(num_turns):
                agent_idx = turn_idx % 2
                prompt_text = current_prompts[agent_idx]
                completion, gen_time, tok_count = generate_completion(
                    agent1 if agent_idx == 0 else agent2,
                    tokenizer,
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

                turn_completions = ["", ""]
                turn_tokens = [0, 0]
                turn_times = [0.0, 0.0]
                turn_completions[agent_idx] = completion
                turn_tokens[agent_idx] = tok_count
                turn_times[agent_idx] = gen_time

                completions_per_turn.append(turn_completions)
                tokens_per_turn.append(turn_tokens)
                times_per_turn.append(turn_times)
                history_commands.append(completion)

                if turn_idx < num_turns - 1:
                    current_prompts = [
                        _format_followup_prompt(base_prompts[0], history_commands, max_commands_total),
                        _format_followup_prompt(base_prompts[1], history_commands, max_commands_total),
                    ]

            merged_commands, accepted_counts = combine_commands(
                completions_per_turn=completions_per_turn,
                task=task,
                allowed_blocks_agent1=allowed_blocks_per_agent[0],
                allowed_blocks_agent2=allowed_blocks_per_agent[1],
                max_commands_total=max_commands_total,
                num_agents=2,
            )

            metrics = simulate_and_score(
                merged_commands=merged_commands,
                task=task,
                allowed_blocks_per_agent=allowed_blocks_per_agent,
            )
            iou = compute_iou(metrics)

            agent_tokens_total = [sum(t[0] for t in tokens_per_turn), sum(t[1] for t in tokens_per_turn)]
            agent_times_total = [sum(t[0] for t in times_per_turn), sum(t[1] for t in times_per_turn)]
            turn_times = [max(t) for t in times_per_turn]  # sequential per turn (only one agent active)
            total_time = sum(turn_times)
            result = {
                "task_id": task.task_id,
                "text": task.text,
                "difficulty": task.difficulty,
                "attempt_id": attempt,
                "num_agents": 2,
                "num_turns": num_turns,
                "pipeline": True,
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

    os.makedirs(output_base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(output_base_dir, f"str_rainbow_pipeline_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved to: {csv_path}")

    summary = {
        "config": "str_rainbow_pipeline",
        "model": model_name,
        "num_agents": 2,
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
        },
        "timestamp": timestamp,
        "aggregated": aggregated,
    }
    json_path = os.path.join(output_base_dir, f"str_rainbow_pipeline_summary_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to: {json_path}")

    print("\n" + "=" * 60)
    print("STR RAINBOW PIPELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Tasks: {len(task_items)} | Attempts per task: {num_samples}")
    print(f"Avg IoU: {aggregated['avg_iou']:.4f} (best-per-task {aggregated['best_iou_per_task']:.4f})")
    print(f"Avg Coverage: {aggregated['avg_coverage']:.4f} | Avg Extra Ratio: {aggregated['avg_extra_ratio']:.4f}")
    print(f"Avg Reward: {aggregated['avg_reward']:.4f}")
    print(f"Avg Total Time: {aggregated['avg_total_time']:.2f}s | Avg Total Tokens: {aggregated['avg_total_tokens']:.1f}")

    return summary


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

    evaluate_pipeline(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
