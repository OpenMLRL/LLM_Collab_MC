from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from LLM_Collab_MC.str_rainbow.utils.prompting import apply_graph_setting
from LLM_Collab_MC.str_rainbow.utils.str_rainbow import (
    TaskSpec,
    build_target_color_map,
    extract_command_lines,
    score_str_rainbow,
    simulate_commands_to_scan_blocks,
    validate_and_normalize_mc_commands,
)


# ------------------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------------------
def compute_iou(metrics: Mapping[str, Any]) -> float:
    """Compute IoU = covered / (target_total + extra_blocks)."""
    covered = float(metrics.get("covered", 0.0))
    extra = float(metrics.get("extra_blocks", 0.0))
    target_total = float(metrics.get("target_total", 0.0))
    denom = target_total + extra
    if denom <= 0:
        return 0.0
    return covered / denom


def as_block_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(v).strip()
    return [s] if s else []


def prepare_prompt_parts(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract prompt pieces to mirror training formatting."""
    prompt_cfg = cfg.get("prompt") or {}
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
    provide_graph = bool(prompt_cfg.get("provide_graph", True))
    use_chat_template = bool(prompt_cfg.get("use_chat_template", False))
    system_prompt = str(prompt_cfg.get("system") or "").rstrip()
    user_template = str(prompt_cfg.get("user_template") or "").rstrip()
    user_template_agent1 = str(prompt_cfg.get("user_template_agent1") or user_template).rstrip()
    user_template_agent2 = str(prompt_cfg.get("user_template_agent2") or user_template).rstrip()

    # Apply graph toggle to keep parity with training
    user_template = apply_graph_setting(user_template, provide_graph=provide_graph)
    user_template_agent1 = apply_graph_setting(user_template_agent1, provide_graph=provide_graph)
    user_template_agent2 = apply_graph_setting(user_template_agent2, provide_graph=provide_graph)

    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    agent1_blocks = as_block_list(task_cfg.get("block_agent1"))
    if not agent1_blocks:
        b0 = str(task_cfg.get("block_even") or "white_concrete").strip()
        b1 = str(task_cfg.get("block_odd") or "black_concrete").strip()
        agent1_blocks = [b0, b1]
    agent2_blocks = as_block_list(task_cfg.get("block_agent2"))
    if not agent2_blocks:
        agent2_blocks = [str(task_cfg.get("block_agent2") or "red_concrete").strip() or "red_concrete"]

    block_agent1_lines = "\n".join(f"- {b}" for b in agent1_blocks)
    block_agent2_lines = "\n".join(f"- {b}" for b in agent2_blocks)

    return {
        "provide_graph": provide_graph,
        "use_chat_template": use_chat_template,
        "system_prompt": system_prompt,
        "user_template": user_template,
        "user_template_agent1": user_template_agent1,
        "user_template_agent2": user_template_agent2,
        "block_agent1_lines": block_agent1_lines,
        "block_agent2_lines": block_agent2_lines,
        "allowed_blocks_agent1": agent1_blocks,
        "allowed_blocks_agent2": agent2_blocks,
    }


def format_context_payload(task: TaskSpec, prompt_parts: Mapping[str, Any], max_commands_total: int) -> Dict[str, Any]:
    """Build context payload for external feedback (mirrors training resolver)."""
    target_ascii = "" if not prompt_parts.get("provide_graph", True) else "\n".join(str(r) for r in task.target_rows_topdown)
    fmt_kwargs = {
        "task_id": task.task_id,
        "text": task.text,
        "difficulty": task.difficulty,
        "world_bbox_from": json.dumps(task.local_bbox_from, separators=(",", ":")),
        "world_bbox_to": json.dumps(task.local_bbox_to, separators=(",", ":")),
        "target_ascii": target_ascii,
        "block_agent1_lines": prompt_parts.get("block_agent1_lines", ""),
        "block_agent2_lines": prompt_parts.get("block_agent2_lines", ""),
    }

    return {
        "system_prompt": prompt_parts.get("system_prompt", ""),
        "user_prompt_single": str(prompt_parts.get("user_template", "")).format(**fmt_kwargs).rstrip(),
        "user_prompt_agent1": str(prompt_parts.get("user_template_agent1", prompt_parts.get("user_template", ""))).format(**fmt_kwargs).rstrip(),
        "user_prompt_agent2": str(prompt_parts.get("user_template_agent2", prompt_parts.get("user_template", ""))).format(**fmt_kwargs).rstrip(),
        "task_id": task.task_id,
        "text": task.text,
        "difficulty": task.difficulty,
        "local_bbox_from": task.local_bbox_from,
        "local_bbox_to": task.local_bbox_to,
        "target_rows_topdown": task.target_rows_topdown,
        "allowed_blocks_agent1": prompt_parts.get("allowed_blocks_agent1", []),
        "allowed_blocks_agent2": prompt_parts.get("allowed_blocks_agent2", []),
        "max_commands_total": int(max_commands_total),
    }


def build_context_resolver(tasks: Sequence[TaskSpec], prompt_parts: Mapping[str, Any], *, max_commands_total: int, set_resolver_fn) -> Dict[str, Dict[str, Any]]:
    """Create context map and register resolver for external feedback."""
    context_map: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        key = f"str_rainbow:{task.task_id}"
        context_map[key] = format_context_payload(task, prompt_parts, max_commands_total)

    def _resolver(prompt: str) -> Dict[str, Any] | None:
        if not prompt:
            return None
        return context_map.get(str(prompt).strip())

    set_resolver_fn(_resolver)
    return context_map


def task_to_item_dict(task: TaskSpec) -> Dict[str, Any]:
    """Convert TaskSpec to the dict format expected by training formatters."""
    return {
        "task_id": task.task_id,
        "csv_row_index": task.csv_row_index,
        "string": task.text,
        "difficulty": task.difficulty,
        "local_bbox_from": task.local_bbox_from,
        "local_bbox_to": task.local_bbox_to,
        "target_rows_topdown": task.target_rows_topdown,
        "prompt": f"str_rainbow:{task.task_id}",
    }


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, float, int]:
    """Generate one completion and return text, wall time, and new-token count."""
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    prompt_len = inputs.input_ids.shape[1]

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


def build_command_limits(num_agents: int, max_commands_total: int) -> List[int]:
    """Match reward function's command budget split."""
    if num_agents <= 1:
        return [int(max_commands_total)]
    max_per = max(1, int(max_commands_total) // num_agents)
    extra = int(max_commands_total) % num_agents
    limits = [max_per] * num_agents
    limits[0] += extra
    return limits


def combine_commands(
    *,
    completions_per_turn: List[List[str]],
    task: TaskSpec,
    allowed_blocks_agent1: List[str],
    allowed_blocks_agent2: List[str],
    max_commands_total: int,
    num_agents: int,
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
        # Keep chronological order across turns for overwrite semantics.
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
        )
        accepted_all[agent_idx] = accepted
        accepted_counts[agent_idx] = len(accepted)

    # Merge in agent order (agent 2 overwrites agent 1 if positions collide).
    merged: List[str] = []
    for agent_list in accepted_all:
        merged.extend(agent_list)
    return merged, accepted_counts


def simulate_and_score(
    *,
    merged_commands: List[str],
    task: TaskSpec,
    allowed_blocks_per_agent: List[List[str]],
) -> Dict[str, Any]:
    blocks = simulate_commands_to_scan_blocks(
        commands=merged_commands,
        world_bbox_from=task.local_bbox_from,
        world_bbox_to=task.local_bbox_to,
    )
    expected_map, _owners = build_target_color_map(
        task=task,
        allowed_blocks_per_agent=allowed_blocks_per_agent,
        num_agents=len(allowed_blocks_per_agent),
    )
    metrics = score_str_rainbow(
        task=task,
        world_scan_blocks=blocks,
        expected_map=expected_map,
        allowed_blocks_per_agent=allowed_blocks_per_agent,
    )
    return metrics
