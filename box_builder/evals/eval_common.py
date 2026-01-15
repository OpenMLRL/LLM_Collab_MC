from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

# Make sure we can import project modules
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_DIR)
REPO_PARENT = os.path.dirname(REPO_ROOT)
for p in (REPO_PARENT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from LLM_Collab_MC.box_builder.utils.box_builder import (  # noqa: E402
    TaskSpec,
    compute_resource_limits,
    extract_command_lines,
    score_box_builder,
    simulate_commands_to_scan_blocks,
    validate_and_normalize_mc_commands,
)


def build_command_limits(num_agents: int, max_commands_total: int) -> List[int]:
    if num_agents <= 1:
        return [int(max_commands_total)]
    max_per = max(1, int(max_commands_total) // num_agents)
    extra = int(max_commands_total) % num_agents
    limits = [max_per] * num_agents
    limits[0] += extra
    return limits


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
) -> Tuple[str, float, int]:
    """Generate one completion and return text, wall time, and new-token count."""
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    prompt_len = inputs.input_ids.shape[1]

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,
        )
    end = time.perf_counter()

    completion_tokens = outputs[0, prompt_len:]
    token_count = len(completion_tokens)
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    return completion_text, end - start, token_count


def task_to_item_dict(task: TaskSpec) -> Dict[str, Any]:
    """Convert TaskSpec to dict expected by formatters."""
    return {
        "task_id": task.task_id,
        "local_bbox_from": task.local_bbox_from,
        "local_bbox_to": task.local_bbox_to,
        "palette": task.palette,
        "layers_by_y": task.layers_by_y,
        "prompt": f"box_builder:{task.task_id}",
    }


def combine_commands(
    *,
    completions_per_turn: List[List[str]],
    task: TaskSpec,
    allowed_blocks_agent1: List[str],
    allowed_blocks_agent2: List[str],
    max_commands_total: int,
    num_agents: int,
    resource_limits: Dict[str, int] | None = None,
) -> Tuple[List[str], List[int]]:
    """Normalize commands and return merged command list + accepted counts per agent."""
    limits = build_command_limits(num_agents, max_commands_total)
    accepted_counts = [0 for _ in range(num_agents)]

    def _allowed(idx: int) -> List[str]:
        if idx == 0:
            return allowed_blocks_agent1
        if idx == 1 and allowed_blocks_agent2:
            return allowed_blocks_agent2
        return allowed_blocks_agent1

    accepted_all: List[List[str]] = [[] for _ in range(num_agents)]
    for agent_idx in range(num_agents):
        lines: List[str] = []
        for turn_vals in completions_per_turn:
            if agent_idx < len(turn_vals):
                lines.extend(extract_command_lines(turn_vals[agent_idx]))
        accepted, _rejected = validate_and_normalize_mc_commands(
            lines=lines,
            allowed_blocks=_allowed(agent_idx),
            world_bbox_from=task.local_bbox_from,
            world_bbox_to=task.local_bbox_to,
            max_commands=limits[agent_idx],
            resource_limits=resource_limits,
        )
        accepted_all[agent_idx] = accepted
        accepted_counts[agent_idx] = len(accepted)

    merged: List[str] = []
    for agent_list in accepted_all:
        merged.extend(agent_list)
    return merged, accepted_counts


def simulate_and_score(
    *,
    merged_commands: List[str],
    task: TaskSpec,
) -> Dict[str, Any]:
    blocks = simulate_commands_to_scan_blocks(
        commands=merged_commands,
        world_bbox_from=task.local_bbox_from,
        world_bbox_to=task.local_bbox_to,
    )
    metrics = score_box_builder(task=task, world_scan_blocks=blocks)
    return metrics


def allowed_blocks_for_agents(task: TaskSpec, overrides1: List[str], overrides2: List[str], num_agents: int) -> List[List[str]]:
    def _norm_list(raw: List[str]) -> List[str]:
        out: List[str] = []
        for b in raw:
            s = str(b).strip()
            if s:
                out.append(s)
        return out

    palette_blocks = list({str(v).strip() for v in task.palette.values() if str(v).strip()})
    a1 = _norm_list(overrides1) or palette_blocks
    a2 = _norm_list(overrides2) or palette_blocks
    blocks = [a1]
    if num_agents >= 2:
        blocks.append(a2)
    return blocks


def compute_resource_limits_for_task(task: TaskSpec, num_agents: int, limited: bool) -> Dict[str, int] | None:
    if not limited:
        return None
    return compute_resource_limits(task, num_agents=num_agents)
