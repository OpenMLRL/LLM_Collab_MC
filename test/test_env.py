#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test: start a mineflayer bot, run a few hard-coded /fill commands, "
            "then verify blocks via bot.blockAt."
        )
    )
    parser.add_argument("--host", default=os.environ.get("MC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MC_PORT", "25565")))
    parser.add_argument("--username", default=os.environ.get("MC_USERNAME", "executor_bot"))
    parser.add_argument("--version", default=os.environ.get("MC_VERSION"))
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--step-delay-ms", type=int, default=600)
    args = parser.parse_args()

    this_dir = Path(__file__).resolve().parent
    repo_dir = this_dir.parent
    bot_js = this_dir / "bot_executor.cjs"

    cmd = [
        "node",
        str(bot_js),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--username",
        args.username,
        "--timeout-ms",
        str(args.timeout_ms),
        "--step-delay-ms",
        str(args.step_delay_ms),
    ]
    if args.version:
        cmd += ["--version", args.version]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print("ERROR: node not found. Install Node.js first.", file=sys.stderr)
        return 127

    if proc.stderr.strip():
        print(proc.stderr, end="", file=sys.stderr)

    # The node script prints exactly one JSON line to stdout.
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not line:
        print("ERROR: no stdout from node bot runner.", file=sys.stderr)
        return proc.returncode or 1

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        print("ERROR: failed to parse node stdout as JSON.", file=sys.stderr)
        print(proc.stdout, end="", file=sys.stderr)
        return proc.returncode or 1

    ok = bool(payload.get("ok"))
    if ok:
        origin = payload.get("origin")
        print(f"OK: bot executed commands and verified blocks. origin={origin}")
        return 0

    print("FAILED: bot runner returned ok=false", file=sys.stderr)
    for e in payload.get("errors", []) or []:
        print(f"- {e}", file=sys.stderr)
    print("raw_result:", json.dumps(payload, ensure_ascii=False), file=sys.stderr)

    if "Cannot find module 'mineflayer'" in proc.stderr:
        print(
            f"hint: install deps: cd {repo_dir} && npm i mineflayer vec3",
            file=sys.stderr,
        )

    return proc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
