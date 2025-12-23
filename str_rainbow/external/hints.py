from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _infer_spacing(rows: List[str], text_len: int, spacing_default: Any) -> int:
    if text_len <= 1:
        return 0
    width = len(rows[0]) if rows else 0
    base = text_len * 5
    rem = width - base
    if rem >= 0 and (text_len - 1) > 0 and rem % (text_len - 1) == 0:
        return int(rem // (text_len - 1))
    try:
        return max(0, int(spacing_default))
    except Exception:
        return 0


def _format_positions(points: List[Tuple[int, int, int]]) -> str:
    if not points:
        return ""
    return "\n".join(f"- ({x}, {y}, {z})" for x, y, z in points)


def _select_one_letter_positions(
    *,
    text: str,
    rows: List[str],
    local_bbox_from: List[int],
    spacing_default: Any,
) -> List[Tuple[int, int, int]]:
    text = str(text or "")
    if not text or not rows:
        return []
    height = len(rows)
    spacing = _infer_spacing(rows, len(text), spacing_default)
    import secrets

    letter_idx = secrets.randbelow(len(text))
    start = letter_idx * (5 + spacing)
    end = start + 4
    positions: List[Tuple[int, int, int]] = []
    for r, row in enumerate(rows):
        if start >= len(row):
            break
        for x in range(start, min(end + 1, len(row))):
            if row[x] != "#":
                continue
            wx = int(local_bbox_from[0]) + x
            wy = int(local_bbox_from[1]) + (height - 1 - r)
            wz = int(local_bbox_from[2])
            positions.append((wx, wy, wz))
    return positions


def append_turn1_hint(base_user: str, ctx: Dict[str, Any]) -> str:
    turn1_hint = ctx.get("turn1_hint") or {}
    if not isinstance(turn1_hint, dict):
        return base_user
    if not bool(turn1_hint.get("one_letter", False)):
        return base_user
    try:
        difficulty = int(ctx.get("difficulty") or 0)
    except Exception:
        difficulty = 0
    if difficulty <= 1:
        return base_user

    text = str(ctx.get("text") or "")
    rows = [str(r) for r in (ctx.get("target_rows_topdown") or [])]
    local_bbox_from = [int(v) for v in (ctx.get("local_bbox_from") or [0, 0, 0])]
    spacing_default = ctx.get("hint_spacing", 1)

    positions = _select_one_letter_positions(
        text=text,
        rows=rows,
        local_bbox_from=local_bbox_from,
        spacing_default=spacing_default,
    )
    if not positions:
        return base_user

    hint_block = "\n".join(
        [
            "Turn 1 hint (one-letter positions):",
            _format_positions(positions),
            "You must infer the remaining characters' shapes and positions yourself.",
        ]
    ).strip()
    if not hint_block:
        return base_user

    if base_user.strip():
        return base_user.rstrip() + "\n\n" + hint_block
    return hint_block
