/* eslint-disable no-console */
'use strict';

const fs = require('fs');
const path = require('path');
const mineflayer = require('mineflayer');

let viewerFn = null;
let viewerLoadError = null;
try {
  const viewerPath = require.resolve('prismarine-viewer', {
    paths: [path.join(__dirname, '..', '..')],
  });
  const viewerMod = require(viewerPath);
  viewerFn = viewerMod && typeof viewerMod.mineflayer === 'function' ? viewerMod.mineflayer : null;
  if (!viewerFn) {
    viewerLoadError = new Error('prismarine-viewer loaded but mineflayer export missing');
  }
} catch (err) {
  // Viewer is optional; the bot can still build without it.
  viewerFn = null;
  viewerLoadError = err;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseArgs(argv) {
  /** @type {Record<string, string | boolean>} */
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      out[key] = true;
      continue;
    }
    out[key] = next;
    i += 1;
  }
  return out;
}

function toInt(value, fallback) {
  const n = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(n) ? n : fallback;
}

function toBool(value, fallback) {
  if (value === undefined || value === null) return fallback;
  if (typeof value === 'boolean') return value;
  const v = String(value).toLowerCase().trim();
  if (v === 'true' || v === '1' || v === 'yes') return true;
  if (v === 'false' || v === '0' || v === 'no') return false;
  return fallback;
}

function normalizeCmd(line) {
  const s = String(line ?? '').trim();
  if (!s) return null;
  return s.startsWith('/') ? s : `/${s}`;
}

function loadPrompt(promptPath) {
  try {
    return fs.readFileSync(promptPath, 'utf8').trim();
  } catch {
    return '';
  }
}

function llmAgent(promptText, origin) {
  // Deterministic "LLM" stub: generates a small house + furniture.
  const x0 = origin.x;
  const y0 = origin.y;
  const z0 = origin.z;
  const x1 = x0 + 6;
  const z1 = z0 + 6;

  const cmds = [];

  // Clear the area first.
  cmds.push(`/fill ${x0 - 1} ${y0} ${z0 - 1} ${x1 + 1} ${y0 + 6} ${z1 + 1} air`);

  // Floor.
  cmds.push(`/fill ${x0} ${y0} ${z0} ${x1} ${y0} ${z1} oak_planks`);

  // Walls (solid), then hollow the interior.
  cmds.push(`/fill ${x0} ${y0 + 1} ${z0} ${x1} ${y0 + 3} ${z1} spruce_planks`);
  cmds.push(`/fill ${x0 + 1} ${y0 + 1} ${z0 + 1} ${x1 - 1} ${y0 + 3} ${z1 - 1} air`);

  // Door (front wall, centered).
  const doorX = x0 + 3;
  const doorZ = z0;
  cmds.push(`/setblock ${doorX} ${y0 + 1} ${doorZ} oak_door[facing=south,half=lower]`);
  cmds.push(`/setblock ${doorX} ${y0 + 2} ${doorZ} oak_door[facing=south,half=upper]`);

  // Window (right wall).
  const winX = x1;
  const winZ = z0 + 3;
  cmds.push(`/setblock ${winX} ${y0 + 2} ${winZ} glass_pane`);
  cmds.push(`/setblock ${winX} ${y0 + 3} ${winZ} glass_pane`);

  // Roof.
  cmds.push(`/fill ${x0} ${y0 + 4} ${z0} ${x1} ${y0 + 4} ${z1} spruce_planks`);

  // Furniture inside.
  cmds.push(`/setblock ${x0 + 2} ${y0 + 1} ${z0 + 2} crafting_table`);
  cmds.push(`/setblock ${x0 + 4} ${y0 + 1} ${z0 + 2} bookshelf`);
  cmds.push(`/setblock ${x0 + 2} ${y0 + 1} ${z0 + 4} furnace`);
  cmds.push(`/setblock ${x0 + 4} ${y0 + 1} ${z0 + 4} jack_o_lantern`);

  return { prompt: promptText, commands: cmds };
}

function writeLlmOutput(pathname, payload, origin) {
  const lines = [];
  if (payload.prompt) {
    lines.push('# LLM prompt');
    lines.push(payload.prompt);
    lines.push('');
  }
  lines.push(`# origin: ${origin.x} ${origin.y} ${origin.z}`);
  lines.push('# LLM output commands');
  for (const cmd of payload.commands) {
    lines.push(cmd);
  }
  fs.writeFileSync(pathname, `${lines.join('\n')}\n`, 'utf8');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const host = String(args.host ?? '127.0.0.1');
  const port = toInt(args.port, 25565);
  const username = String(args.username ?? 'executor_bot');
  const version = typeof args.version === 'string' ? args.version : undefined;
  const stepDelayMs = toInt(args.stepDelayMs ?? args['step-delay-ms'], 250);
  const timeoutMs = toInt(args.timeoutMs ?? args['timeout-ms'], 60_000);
  const offsetX = toInt(args.offsetX ?? args['offset-x'], 6);
  const offsetZ = toInt(args.offsetZ ?? args['offset-z'], 6);
  const viewPort = toInt(args.viewerPort ?? args['viewer-port'], 3000);
  const enableViewer = toBool(args.viewer ?? args['enable-viewer'], true);
  const keepAlive = toBool(args.keepAlive ?? args['keep-alive'], true);
  const dumpPath = typeof args['dump-path'] === 'string'
    ? String(args['dump-path'])
    : path.join(__dirname, 'llm_output.txt');
  const dumpOutput = toBool(args['dump'] ?? args['dump-output'], true);

  const promptPath = path.join(__dirname, 'llm_prompt.txt');
  const promptText = loadPrompt(promptPath);

  const bot = mineflayer.createBot({
    host,
    port,
    username,
    auth: 'offline',
    ...(version ? { version } : {}),
  });

  let finished = false;
  const timeout = setTimeout(() => {
    if (finished) return;
    finished = true;
    console.error(`timeout after ${timeoutMs}ms (server not responding?)`);
    try {
      bot.quit();
    } catch {
      // ignore
    }
    process.exit(2);
  }, timeoutMs);

  bot.on('kicked', (reason) => {
    if (finished) return;
    finished = true;
    console.error(`kicked: ${String(reason)}`);
    clearTimeout(timeout);
    process.exit(3);
  });

  bot.on('error', (err) => {
    if (finished) return;
    finished = true;
    console.error(`error: ${err?.message ?? String(err)}`);
    clearTimeout(timeout);
    process.exit(4);
  });

  bot.once('spawn', async () => {
    try {
      clearTimeout(timeout);
      const base = bot.entity.position.floored();
      const origin = base.offset(offsetX, 0, offsetZ);
      console.log(`spawned at ${base.x} ${base.y} ${base.z}`);
      console.log(`building origin ${origin.x} ${origin.y} ${origin.z}`);

      if (enableViewer && viewerFn) {
        viewerFn(bot, { port: viewPort, firstPerson: false });
        console.log(`viewer: http://127.0.0.1:${viewPort}/`);
      } else if (enableViewer) {
        const reason = viewerLoadError ? `: ${viewerLoadError.message}` : '';
        console.warn(`viewer disabled (prismarine-viewer not available${reason})`);
      }

      const payload = llmAgent(promptText, origin);
      if (dumpOutput) {
        writeLlmOutput(dumpPath, payload, origin);
      }

      for (const raw of payload.commands) {
        const cmd = normalizeCmd(raw);
        if (!cmd) continue;
        bot.chat(cmd);
        await sleep(stepDelayMs);
      }

      console.log('build complete');
      if (!keepAlive) {
        finished = true;
        bot.quit();
        process.exit(0);
      }

      // Keep the process alive for viewing.
      setInterval(() => {}, 10_000);
    } catch (e) {
      console.error(`exception: ${e?.stack ?? e?.message ?? String(e)}`);
      process.exit(5);
    }
  });
}

main().catch((e) => {
  process.stderr.write(`${e?.stack ?? e?.message ?? String(e)}\n`);
  process.exit(10);
});
