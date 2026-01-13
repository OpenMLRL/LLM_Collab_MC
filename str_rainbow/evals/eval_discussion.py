#!/usr/bin/env python3
"""
Discussion evaluation for str_rainbow (multi-turn with feedback, IoU metric).
Leverages the shared evaluator with num_turns>1 and external feedback mode from config.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_DIR)
REPO_PARENT = os.path.dirname(REPO_ROOT)
for p in (REPO_PARENT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from LLM_Collab_MC.str_rainbow.evals.eval_str_rainbow import evaluate_str_rainbow
from LLM_Collab_MC.str_rainbow.utils.config import apply_overrides, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discussion (multi-turn) evaluation for str_rainbow (IoU metric).")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_DIR, "evals", "configs", "discussion_config.yaml"),
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
