#!/usr/bin/env python3
"""
Pipeline evaluation for box_builder (Agent 1 then Agent 2 sees Agent 1 commands).
Sequential time (sum), IoU metric.
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

from LLM_Collab_MC.box_builder.evals.eval_common import (  # noqa: E402
    allowed_blocks_for_agents,
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


def _pipeline_prompt_agent2(base_prompt: str, agent1_output: str, max_commands: int) -> str:
    agent1_output = (agent1_output or "").strip()
    extra = ""
    if agent1_output:
        extra = (
            "\n\nAgent 1 commands already placed (do not repeat unless fixing):\n"
            f"{agent1_output}\n"
            "\nContinue with your /fill commands to complete the target. "
            f"Max commands allowed: {max_commands}\n"
            "Output ONLY Minecraft commands, one per line (no markdown)."
        )
    return f"{base_prompt.rstrip()}{extra}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline evaluation for box_builder (IoU metric).")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_DIR, "evals", "configs", "pipeline_config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="HF model name (overrides config).")
    parser.add_argument("--json-path", type=str, default=None, help="Dataset JSON path (overrides config).")
    parser.add_argument("--eval-split", type=str, default=None, help="Slice expression, e.g., '[:32]'.")
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


def evaluate_pipeline(cfg: Dict[str, Any], args_ns: argparse.Namespace) -> Dict[str, Any]:
    apply_prompt_defaults(cfg)

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    json_path = resolve_path(args_ns.config, args_ns.json_path or dataset_cfg.get("json_path"))
    eval_split = args_ns.eval_split or dataset_cfg.get("eval_split") or "[:]"

    tasks = load_tasks_from_json(json_path)
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
    max_new_tokens = int(args_ns.max_new_tokens or eval_cfg.get("max_new_tokens") or 512)
    temperature = float(args_ns.temperature or eval_cfg.get("temperature") or model_cfg.get("temperature") or 0.6)
    top_p = float(args_ns.top_p or eval_cfg.get("top_p") or model_cfg.get("top_p") or 0.6)

    # Task config
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    max_commands_total = int(task_cfg.get("max_commands") or 600)
    limited_resource = bool(task_cfg.get("limited_resource", False))
    block_agent1_over = task_cfg.get("block_agent1") or []
    block_agent2_over = task_cfg.get("block_agent2") or []

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

    print(f"Loading model instances: {model_name}")
    agent1 = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    agent1.eval()
    agent2 = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    agent2.eval()

    formatters = _build_formatters(cfg, num_agents=2, tokenizer=tokenizer)

    all_results: List[Dict[str, Any]] = []
    best_iou_per_task: Dict[str, float] = {}

    for task_idx, task in enumerate(task_items):
        if verbose:
            print(f"\n{'='*60}\nTask {task_idx+1}/{len(task_items)}: {task.task_id}")
        task_item = task_to_item_dict(task)

        allowed_blocks_per_agent = allowed_blocks_for_agents(task, block_agent1_over, block_agent2_over, num_agents=2)
        resource_limits = compute_resource_limits_for_task(task, num_agents=2, limited=limited_resource)

        base_prompt_agent1 = formatters[0](task_item)
        base_prompt_agent2 = formatters[1](task_item)

        for attempt in range(num_samples):
            if verbose:
                print(f"  Attempt {attempt+1}/{num_samples}")

            # Agent 1
            c1, t1, tok1 = generate_completion(
                agent1,
                tokenizer,
                base_prompt_agent1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # Agent 2 sees Agent 1 output
            agent2_prompt = _pipeline_prompt_agent2(base_prompt_agent2, c1, max_commands_total)
            c2, t2, tok2 = generate_completion(
                agent2,
                tokenizer,
                agent2_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            completions_per_turn = [
                [c1, ""],
                ["", c2],
            ]

            merged_commands, accepted_counts = combine_commands(
                completions_per_turn=completions_per_turn,
                task=task,
                allowed_blocks_agent1=allowed_blocks_per_agent[0],
                allowed_blocks_agent2=allowed_blocks_per_agent[1],
                max_commands_total=max_commands_total,
                num_agents=2,
                resource_limits=resource_limits,
            )

            metrics = simulate_and_score(
                merged_commands=merged_commands,
                task=task,
            )
            iou = float(metrics.get("iou", 0.0))
            reward = float(metrics.get("score_mean", 0.0))

            total_time = t1 + t2
            result = {
                "task_id": task.task_id,
                "attempt_id": attempt,
                "num_agents": 2,
                "num_turns": 2,
                "pipeline": True,
                "iou": round(iou, 4),
                "score_match": round(float(metrics.get("score_match", 0.0)), 4),
                "reward": round(reward, 4),
                "commands_agent1": accepted_counts[0] if accepted_counts else 0,
                "commands_agent2": accepted_counts[1] if len(accepted_counts) > 1 else 0,
                "tokens_agent1_total": tok1,
                "tokens_agent2_total": tok2,
                "time_agent1_s": round(t1, 4),
                "time_agent2_s": round(t2, 4),
                "total_time_s": round(total_time, 4),
                "total_tokens": tok1 + tok2,
                "max_commands_total": max_commands_total,
            }
            all_results.append(result)
            best_iou = best_iou_per_task.get(task.task_id, 0.0)
            if iou > best_iou:
                best_iou_per_task[task.task_id] = iou

            if verbose:
                print(
                    f"    IoU={iou:.4f} | match={metrics.get('score_match', 0.0):.3f} "
                    f"| reward={reward:.3f} | time={total_time:.2f}s | tokens={tok1+tok2}"
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

    results_csv_path = os.path.join(output_base_dir, f"box_builder_pipeline_{timestamp}.csv")
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved to: {results_csv_path}")

    summary = {
        "config": "box_builder_pipeline",
        "model": model_name,
        "num_agents": 2,
        "num_turns": 2,
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
    summary_path = os.path.join(output_base_dir, f"box_builder_pipeline_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("BOX_BUILDER PIPELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Tasks: {len(task_items)} | Attempts per task: {num_samples}")
    print(f"Avg IoU: {aggregated['avg_iou']:.4f} (best-per-task {aggregated['best_iou_per_task']:.4f})")
    print(f"Avg Reward: {aggregated['avg_reward']:.4f} | Avg Time: {aggregated['avg_total_time']:.2f}s | Avg Tokens: {aggregated['avg_total_tokens']:.1f}")

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
