#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


def _read_payload() -> Dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"prompt": raw}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_torch_and_transformers():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from transformers.utils import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
        return torch, AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime env dependent
        msg = (
            "ERROR: failed to import transformers/torch. "
            "If you see a CUDA library error, install a CPU-only torch build. "
            f"Details: {exc}"
        )
        print(msg, file=sys.stderr)
        return None, None, None


def _pick_device(torch_mod, requested: str) -> str:
    req = (requested or "auto").lower()
    if req == "cpu":
        return "cpu"
    if torch_mod.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_dtype(torch_mod, device: str):
    if device == "cuda":
        if hasattr(torch_mod.cuda, "is_bf16_supported") and torch_mod.cuda.is_bf16_supported():
            return torch_mod.bfloat16
        return torch_mod.float16
    return torch_mod.float32


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    payload = _read_payload()
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        print("ERROR: prompt is empty", file=sys.stderr)
        return 2

    system = str(payload.get("system") or "").strip()
    if not system:
        system = (
            "You are a Minecraft building agent. "
            "Output only /fill and /setblock commands, one per line. No extra text."
        )

    model_id = str(payload.get("model") or args.model)
    temperature = _to_float(payload.get("temperature"), args.temperature)
    top_p = _to_float(payload.get("top_p"), args.top_p)
    max_new_tokens = _to_int(payload.get("max_new_tokens"), args.max_new_tokens)
    device_req = str(payload.get("device") or args.device)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    torch_mod, AutoModelForCausalLM, AutoTokenizer = _load_torch_and_transformers()
    if torch_mod is None:
        return 3

    device = _pick_device(torch_mod, device_req)
    dtype = _pick_dtype(torch_mod, device)

    hf_token = (
        payload.get("hf_token")
        or args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=hf_token,
    )

    try:
        model.to(device)
    except Exception as exc:
        if device != "cpu":
            print(
                f"WARN: failed to move model to {device} ({exc}); falling back to cpu",
                file=sys.stderr,
            )
            device = "cpu"
            model.to(device)
        else:
            raise

    model.eval()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"{system}\n\n{prompt}\n"

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = temperature > 0

    with torch_mod.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    out_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    result = {
        "text": out_text,
        "model": model_id,
        "device": device,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
