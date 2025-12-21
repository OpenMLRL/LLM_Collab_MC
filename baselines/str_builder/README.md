# String Builder Baseline

`baselines/str_builder/main.py` reads tasks from `../../dataset/str_builder/data.csv`, prompts an LLM once per task to output **Minecraft command lines** (one command per line), then scores the final build result and writes an eval JSON/JSONL file.

The task: given an uppercase string (e.g. `ICML`), build its 5x5 block-letter mask on a constrained 2D plane. Only the letters should be built (no background). Adjacent blocks (sharing a side) must not be the same color.

## Prereqs

- Python env with: `pyyaml`, `torch`, `transformers`, `accelerate`
- Node.js >= 18 + repo node deps: run once from repo root: `cd LLM_Collab_MC && npm install`

Notes:

- Default (`minecraft.enabled=false`) uses an **offline simulator** for `/fill` and `/setblock` to compute metrics (no server needed).
- If you enable real Minecraft execution (`minecraft.enabled=true`), you also need a running server and the bot username must be OP (`op executor_bot`).

## Run locally

From repo root:

- Dry-run (no model load, prints first prompts):
  - `python3 baselines/str_builder/main.py --config baselines/str_builder/config.yaml --dry-run --limit 1`
- Real run:
  - `python3 baselines/str_builder/main.py --config baselines/str_builder/config.yaml`

Useful flags:

- `--limit N`: only run first N tasks
- `--dry-run`: don’t load model, just show prompts

## Output

This baseline writes **two** JSONL files per run:

- Main output: `output.path` from `baselines/str_builder/config.yaml`
- Simplified output: `output.simple_path` (optional); defaults to `{output.path.stem}.simple.jsonl`

The simplified JSONL keeps only: `task_id/string/difficulty`, `model_id`, and the 3 scores (`score_shape_overlap`, `score_components`, `score_s3`).

## Multi-agent (num_agents=2)

Set `agents.num_agents: 2` in `baselines/str_builder/config.yaml` to run two agents per task:

- Agent1 can only use `task.block_agent1` (list of allowed blocks).
- Agent2 can only use `task.block_agent2` (list of allowed blocks).
- Their validated commands are merged and executed in the same world, then scored with the same metrics.

## Metrics (offline + optional MC scan)

Each record writes:

- `metrics.score_shape_overlap`: `IoU`
- `metrics.score_components`: `min(num_8cc / len(string), 1)`
- `metrics.score_s3`: adjacency score for same-color neighbors (`1 - num/difficulty` if `num <= difficulty`, else `-1`)
- `metrics.score_mean`: mean of the three scores above

If Minecraft execution is enabled and scan succeeds, the same metrics are also written with `mc_` prefixes.
