from __future__ import annotations

from typing import Any, Dict


DEFAULT_SYSTEM_PROMPT = (
    "You are a Minecraft building agent.\n"
    "Output must be Minecraft commands only (no markdown, no code fences, no extra text).\n"
    "This is strict: any non-command text is invalid."
)

DEFAULT_USER_TEMPLATE = """Build the 3D structure from the provided y-axis slices.

Available blocks (use ONLY these):
{block_agent1_lines}

Layers (ascending WORLD y). Each layer is a set of rectangles in WORLD (x,z) coords:
- Format: y=ABS_Y: {{(x1, z1, x2, z2 block_id), (x1, z1, x2, z2 block_id)}}
{layers_text}

WORLD bbox (inclusive):
- from: {world_bbox_from}
- to:   {world_bbox_to}

Constraints:
- Output ONLY Minecraft commands, one per line.
- Allowed commands: /fill only
- Fill format: /fill x1 y1 z1 x2 y2 z2 block
- Use absolute integer coordinates only (no ~).
- Use ONLY blocks from the legend.
- Every coordinate must be within the bbox.
"""

DEFAULT_USER_TEMPLATE_AGENT1 = """You are Agent 1 in a 2-person Minecraft building team. You will place SOME of the blocks for the final build.

Task: Build the 3D structure from the provided y-axis slices.

Available blocks (use ONLY these):
{block_agent1_lines}

Layers (ascending WORLD y). Each layer is a set of rectangles in WORLD (x,z) coords:
- Format: y=ABS_Y: {{(x1, z1, x2, z2 block_id), (x1, z1, x2, z2 block_id)}}
{layers_text}

WORLD bbox (inclusive):
- from: {world_bbox_from}
- to:   {world_bbox_to}

Constraints:
- Output ONLY Minecraft commands, one per line.
- Allowed commands: /fill only
- Fill format: /fill x1 y1 z1 x2 y2 z2 block
- Use absolute integer coordinates only (no ~).
- Use ONLY blocks from the legend.
- Every coordinate must be within the bbox.
"""

DEFAULT_USER_TEMPLATE_AGENT2 = """You are Agent 2 in a 2-person Minecraft building team. You will place SOME of the blocks for the final build.

Task: Build the 3D structure from the provided y-axis slices.

Available blocks (use ONLY these):
{block_agent2_lines}

Layers (ascending WORLD y). Each layer is a set of rectangles in WORLD (x,z) coords:
- Format: y=ABS_Y: {{(x1, z1, x2, z2 block_id), (x1, z1, x2, z2 block_id)}}
{layers_text}

WORLD bbox (inclusive):
- from: {world_bbox_from}
- to:   {world_bbox_to}

Constraints:
- Output ONLY Minecraft commands, one per line.
- Allowed commands: /fill only
- Fill format: /fill x1 y1 z1 x2 y2 z2 block
- Use absolute integer coordinates only (no ~).
- Use ONLY blocks from the legend.
- Every coordinate must be within the bbox.
"""

DEFAULT_PROMPT_CONFIG = {
    "use_chat_template": False,
    "include_air_rects": False,
    "turn1_hint": False,
    "system": DEFAULT_SYSTEM_PROMPT,
    "user_template": DEFAULT_USER_TEMPLATE,
    "user_template_agent1": DEFAULT_USER_TEMPLATE_AGENT1,
    "user_template_agent2": DEFAULT_USER_TEMPLATE_AGENT2,
}


def apply_prompt_defaults(cfg: Dict[str, Any]) -> None:
    prompt_cfg = cfg.get("prompt")
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
        cfg["prompt"] = prompt_cfg
    for key, value in DEFAULT_PROMPT_CONFIG.items():
        if key not in prompt_cfg:
            prompt_cfg[key] = value


def _infer_size_x(item: Dict[str, Any]) -> int:
    size = item.get("target_spec_size") or item.get("target_spec_size_x")
    if isinstance(size, (list, tuple)) and size:
        try:
            return int(size[0])
        except Exception:
            pass
    if isinstance(size, int):
        return int(size)
    layers = item.get("layers_by_y") or {}
    if isinstance(layers, dict):
        for rows in layers.values():
            if isinstance(rows, list) and rows:
                return len(str(rows[0]))
    bbox_from = item.get("local_bbox_from") or [0, 0, 0]
    bbox_to = item.get("local_bbox_to") or [0, 0, 0]
    try:
        min_x = min(int(bbox_from[0]), int(bbox_to[0]))
        max_x = max(int(bbox_from[0]), int(bbox_to[0]))
        return max(0, max_x - min_x + 1)
    except Exception:
        return 0


def _turn1_hint_text(item: Dict[str, Any], world_from: list[int] | None, agent_idx: int) -> str:
    size_x = _infer_size_x(item)
    if size_x < 2:
        return ""
    bbox_from = item.get("local_bbox_from") or [0, 0, 0]
    bbox_to = item.get("local_bbox_to") or [0, 0, 0]
    try:
        min_x = min(int(bbox_from[0]), int(bbox_to[0]))
    except Exception:
        min_x = 0
    world_min_x = int(world_from[0]) if world_from is not None else min_x
    mid = size_x // 2
    if mid <= 0:
        return ""
    if agent_idx == 0:
        limit = world_min_x + mid - 1
        return f"(hint: prioritize rectangles with x2 <= {limit})"
    limit = world_min_x + mid
    return f"(hint: prioritize rectangles with x2 >= {limit})"


def append_turn1_hint(
    user_prompt: str,
    *,
    item: Dict[str, Any],
    world_from: list[int] | None,
    agent_idx: int | None,
    num_agents: int,
    prompt_cfg: Dict[str, Any],
) -> str:
    if not bool(prompt_cfg.get("turn1_hint", False)):
        return user_prompt
    if num_agents <= 1 or agent_idx is None:
        return user_prompt
    hint = _turn1_hint_text(item, world_from, agent_idx)
    if not hint:
        return user_prompt
    return user_prompt.rstrip() + "\n\n" + hint
