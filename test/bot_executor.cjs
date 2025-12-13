/* eslint-disable no-console */
'use strict';

const mineflayer = require('mineflayer');
const { Vec3 } = require('vec3');

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

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const host = String(args.host ?? '127.0.0.1');
  const port = toInt(args.port, 25565);
  const username = String(args.username ?? 'executor_bot');
  const version = typeof args.version === 'string' ? args.version : undefined;
  const stepDelayMs = toInt(args.stepDelayMs ?? args['step-delay-ms'], 600);
  const timeoutMs = toInt(args.timeoutMs ?? args['timeout-ms'], 60_000);

  const startMs = Date.now();
  const result = {
    ok: false,
    host,
    port,
    username,
    version: version ?? null,
    origin: null,
    checks: [],
    errors: [],
    duration_ms: null,
  };

  const bot = mineflayer.createBot({
    host,
    port,
    username,
    auth: 'offline',
    ...(version ? { version } : {}),
  });

  let finished = false;

  function finish(exitCode) {
    if (finished) return;
    finished = true;
    result.duration_ms = Date.now() - startMs;
    process.stdout.write(`${JSON.stringify(result)}\n`);
    process.exitCode = exitCode;
    try {
      bot.quit();
    } catch {
      // ignore
    }
  }

  const timeout = setTimeout(() => {
    result.errors.push(`timeout after ${timeoutMs}ms (did the server accept the bot connection?)`);
    finish(2);
  }, timeoutMs);

  bot.on('kicked', (reason) => {
    result.errors.push(`kicked: ${String(reason)}`);
    clearTimeout(timeout);
    finish(3);
  });

  bot.on('error', (err) => {
    result.errors.push(`error: ${err?.message ?? String(err)}`);
    clearTimeout(timeout);
    finish(4);
  });

  bot.once('spawn', async () => {
    try {
      const base = bot.entity.position.floored();
      const origin = base.offset(2, 0, 2);
      const x1 = origin.x;
      const y = origin.y;
      const z1 = origin.z;
      const x2 = x1 + 2;
      const y2 = y + 2;
      const z2 = z1 + 2;

      result.origin = [x1, y, z1];

      const cmds = [
        `/fill ${x1} ${y} ${z1} ${x2} ${y2} ${z2} air`,
        `/fill ${x1} ${y} ${z1} ${x2} ${y} ${z2} stone`,
        `/setblock ${x1} ${y + 1} ${z1} gold_block`,
        `/fill ${x2} ${y + 1} ${z2} ${x2} ${y + 3} ${z2} oak_planks`,
      ];

      for (const cmd of cmds) {
        bot.chat(cmd);
        await sleep(stepDelayMs);
      }

      const checks = [
        { pos: [x1, y, z1], want: 'stone' },
        { pos: [x1, y + 1, z1], want: 'gold_block' },
        { pos: [x2, y + 2, z2], want: 'oak_planks' },
      ];

      for (const c of checks) {
        const [cx, cy, cz] = c.pos;
        const block = bot.blockAt(new Vec3(cx, cy, cz));
        const got = block?.name ?? null;
        const ok = got === c.want;
        result.checks.push({ ...c, got, ok });
      }

      const allOk = result.checks.every((c) => c.ok === true);
      if (!allOk) {
        result.errors.push(
          'block checks failed (common causes: bot not OP so /fill is denied; or chunk not loaded yet)'
        );
        result.errors.push(
          'fix: in the server console run: op <username>  (and ensure gamerule sendCommandFeedback true)'
        );
        clearTimeout(timeout);
        finish(1);
        return;
      }

      result.ok = true;
      clearTimeout(timeout);
      finish(0);
    } catch (e) {
      result.errors.push(`exception: ${e?.stack ?? e?.message ?? String(e)}`);
      clearTimeout(timeout);
      finish(5);
    }
  });
}

main().catch((e) => {
  process.stderr.write(`${e?.stack ?? e?.message ?? String(e)}\n`);
  process.exitCode = 10;
});
