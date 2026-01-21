"""
Shared model aliases and helpers for str_rainbow evaluation scripts.

Environment overrides:
- STR_RAINBOW_BASE_MODEL: base/fallback model
- STR_RAINBOW_GRPO_MODEL: GRPO-trained model path
- STR_RAINBOW_PPO_MODEL: PPO-trained model path
- STR_RAINBOW_DEFAULT_MODEL: default if config/model arg is missing
"""

import os
from typing import Optional

DEFAULT_BASE_MODEL = os.environ.get("STR_RAINBOW_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

MODEL_ALIASES = {
    "base": DEFAULT_BASE_MODEL,
    "grpo": os.environ.get("STR_RAINBOW_GRPO_MODEL", DEFAULT_BASE_MODEL),
    "grpo-best": os.environ.get("STR_RAINBOW_GRPO_MODEL", DEFAULT_BASE_MODEL),
    "ppo": os.environ.get("STR_RAINBOW_PPO_MODEL", DEFAULT_BASE_MODEL),
    "ppo-best": os.environ.get("STR_RAINBOW_PPO_MODEL", DEFAULT_BASE_MODEL),
}

DEFAULT_MODEL = os.environ.get("STR_RAINBOW_DEFAULT_MODEL", MODEL_ALIASES["base"])


def resolve_model_name(name: Optional[str]) -> Optional[str]:
    """Resolve short alias to full HF path; pass through unknown names."""
    if name is None:
        return None
    key = str(name).strip()
    if not key:
        return key
    return MODEL_ALIASES.get(key, key)
