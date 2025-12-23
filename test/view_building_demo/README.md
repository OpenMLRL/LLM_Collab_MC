# View Building Demo

This demo connects a mineflayer bot to a local Minecraft server and builds a small house with furniture.
It also starts a browser viewer (prismarine-viewer) so you can watch the result.

## Prereqs

- A running Minecraft server (Paper 1.19.2 works). Offline mode must be enabled.
- The bot username must be OP (example: `op executor_bot`).
- Node.js >= 18 and repo deps installed.

## Install deps

From repo root:

```bash
cd LLM_Collab_MC
npm install
```

## Run demo

From repo root:

```bash
node test/view_building_demo/build_house_demo.cjs \
  --host 127.0.0.1 \
  --port 25565 \
  --username executor_bot \
  --viewer-port 3000
```

Open the viewer in a browser:

- http://127.0.0.1:3000/

## Notes

- The "LLM" prompt is in `test/view_building_demo/llm_prompt.txt`.
- The demo uses a deterministic LLM stub inside `build_house_demo.cjs`. Swap it with a real model call if needed.
- The demo writes the last prompt + command list to `test/view_building_demo/llm_output.txt`.
- Use `--keep-alive false` if you want the process to exit after building.
