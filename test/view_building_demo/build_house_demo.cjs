/* eslint-disable no-console */
'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const mineflayer = require('mineflayer');
const { Vec3 } = require('vec3');

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

function toFloat(value, fallback) {
  const n = Number.parseFloat(String(value ?? ''));
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

function buildPrompt(promptText, origin) {
  const originLine = `Origin (x0, y0, z0): ${origin.x} ${origin.y} ${origin.z}`;
  if (!promptText) return originLine;
  return `${promptText}\n\n${originLine}`;
}

function findSurfaceY(bot, x, z, startY) {
  const radius = 80;
  let bestY = null;
  for (let y = startY + radius; y >= startY - radius; y -= 1) {
    const block = bot.blockAt(new Vec3(x, y, z));
    if (!block || block.name === 'air') continue;
    if (bestY === null || y > bestY) bestY = y;
  }
  return bestY === null ? startY : bestY + 1;
}

function makeChatUrl(baseUrl) {
  const trimmed = String(baseUrl || '').replace(/\/+$/, '');
  if (!trimmed) return 'https://api.openai.com/v1/chat/completions';
  if (trimmed.endsWith('/v1')) return `${trimmed}/chat/completions`;
  return `${trimmed}/v1/chat/completions`;
}

function extractCommands(text) {
  if (!text) return [];
  const cleaned = String(text)
    .replace(/```[a-zA-Z0-9_-]*\n/g, '')
    .replace(/```/g, '')
    .trim();
  const lines = cleaned.split(/\r?\n/);
  const cmds = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const match = trimmed.match(/(\/?(setblock|fill)\b.*)$/i);
    if (!match) continue;
    const cmd = normalizeCmd(match[1]);
    if (cmd) cmds.push(cmd);
  }
  return cmds;
}

async function callChatCompletion({
  baseUrl,
  apiKey,
  model,
  messages,
  temperature,
  topP,
  maxTokens,
  timeoutMs,
}) {
  if (typeof fetch !== 'function') {
    throw new Error('fetch is not available in this Node version (need Node.js >= 18).');
  }
  const url = makeChatUrl(baseUrl);
  const headers = {
    'Content-Type': 'application/json',
  };
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }

  const payload = {
    model,
    messages,
    temperature,
    top_p: topP,
    max_tokens: maxTokens,
  };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`LLM request failed (${res.status}): ${errText}`);
    }
    const json = await res.json();
    const content = json?.choices?.[0]?.message?.content ?? '';
    return { content, raw: json };
  } finally {
    clearTimeout(timeout);
  }
}

function callHfModel({
  scriptPath,
  prompt,
  system,
  model,
  temperature,
  topP,
  maxTokens,
  device,
  timeoutMs,
}) {
  return new Promise((resolve, reject) => {
    const args = [
      scriptPath,
      '--model',
      model,
      '--max-new-tokens',
      String(maxTokens),
      '--temperature',
      String(temperature),
      '--top-p',
      String(topP),
      '--device',
      device,
    ];

    const child = spawn('python3', args, { stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      try {
        child.kill('SIGKILL');
      } catch {
        // ignore
      }
      reject(new Error(`HF LLM timeout after ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`HF LLM failed (code ${code}): ${stderr.trim() || stdout.trim()}`));
        return;
      }
      const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
      const last = lines[lines.length - 1] || '';
      try {
        const payload = JSON.parse(last);
        resolve({ payload, stderr: stderr.trim() });
      } catch (e) {
        reject(
          new Error(
            `HF LLM returned non-JSON output: ${stderr.trim() || ''}\nstdout:\n${stdout.trim()}`
          )
        );
      }
    });

    const inputPayload = {
      prompt,
      system,
      model,
      temperature,
      top_p: topP,
      max_new_tokens: maxTokens,
      device,
    };
    child.stdin.write(`${JSON.stringify(inputPayload)}\n`);
    child.stdin.end();
  });
}

function llmAgentStub(promptText, origin) {
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

  return { prompt: buildPrompt(promptText, origin), commands: cmds, raw_output: cmds.join('\n') };
}

async function llmAgentOpenAI(promptText, origin, llmCfg) {
  const prompt = buildPrompt(promptText, origin);
  const messages = [
    {
      role: 'system',
      content:
        'You are a Minecraft building agent. Output only /fill and /setblock commands, one per line. No extra text.',
    },
    { role: 'user', content: prompt },
  ];

  const { content } = await callChatCompletion({
    baseUrl: llmCfg.baseUrl,
    apiKey: llmCfg.apiKey,
    model: llmCfg.model,
    messages,
    temperature: llmCfg.temperature,
    topP: llmCfg.topP,
    maxTokens: llmCfg.maxTokens,
    timeoutMs: llmCfg.timeoutMs,
  });

  const commands = extractCommands(content);
  if (!commands.length) {
    throw new Error('LLM returned no /fill or /setblock commands.');
  }
  return { prompt, commands, raw_output: content, model: llmCfg.model };
}

async function llmAgentHf(promptText, origin, llmCfg) {
  const prompt = buildPrompt(promptText, origin);
  const system =
    'You are a Minecraft building agent. Output only /fill and /setblock commands, one per line. No extra text.';
  if (!fs.existsSync(llmCfg.hfScriptPath)) {
    throw new Error(`HF helper not found: ${llmCfg.hfScriptPath}`);
  }
  const { payload } = await callHfModel({
    scriptPath: llmCfg.hfScriptPath,
    prompt,
    system,
    model: llmCfg.model,
    temperature: llmCfg.temperature,
    topP: llmCfg.topP,
    maxTokens: llmCfg.maxTokens,
    device: llmCfg.device,
    timeoutMs: llmCfg.timeoutMs,
  });

  const content = payload?.text ?? '';
  const commands = extractCommands(content);
  if (!commands.length) {
    throw new Error('HF model returned no /fill or /setblock commands.');
  }
  return {
    prompt,
    commands,
    raw_output: content,
    model: payload?.model ?? llmCfg.model,
    backend: 'hf',
  };
}

async function llmAgent(promptText, origin, llmCfg) {
  const mode = String(llmCfg.mode || '').toLowerCase();
  if (mode === 'stub') {
    return llmAgentStub(promptText, origin);
  }
  if (!llmCfg.model) {
    throw new Error('LLM model is required. Set LLM_MODEL or pass --llm-model.');
  }
  if (mode === 'hf' || mode === 'huggingface') {
    return llmAgentHf(promptText, origin, llmCfg);
  }
  if (!llmCfg.apiKey && !llmCfg.allowNoKey) {
    throw new Error('LLM API key is required. Set LLM_API_KEY or pass --llm-api-key.');
  }
  return llmAgentOpenAI(promptText, origin, llmCfg);
}

function writeLlmOutput(pathname, payload, origin) {
  const lines = [];
  if (payload.prompt) {
    lines.push('# LLM prompt');
    lines.push(payload.prompt);
    lines.push('');
  }
  lines.push(`# origin: ${origin.x} ${origin.y} ${origin.z}`);
  if (payload.raw_output) {
    lines.push('# LLM raw output');
    lines.push(payload.raw_output);
    lines.push('');
  }
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
  const llmMode = String(args.llmMode ?? args['llm-mode'] ?? process.env.LLM_MODE ?? 'hf').toLowerCase();
  const llmBaseUrl = String(
    args.llmBaseUrl ??
      args['llm-base-url'] ??
      process.env.LLM_BASE_URL ??
      process.env.OPENAI_BASE_URL ??
      process.env.OPENAI_API_BASE ??
      'https://api.openai.com/v1'
  );
  const llmApiKey = String(
    args.llmApiKey ?? args['llm-api-key'] ?? process.env.LLM_API_KEY ?? process.env.OPENAI_API_KEY ?? ''
  );
  const defaultModel =
    llmMode === 'hf' || llmMode === 'huggingface' ? 'Qwen/Qwen3-4B-Instruct-2507' : 'gpt-4o-mini';
  const llmModel = String(args.llmModel ?? args['llm-model'] ?? process.env.LLM_MODEL ?? defaultModel);
  const llmTemperature = toFloat(
    args.llmTemperature ?? args['llm-temperature'] ?? process.env.LLM_TEMPERATURE,
    0.2
  );
  const llmTopP = toFloat(args.llmTopP ?? args['llm-top-p'] ?? process.env.LLM_TOP_P, 0.95);
  const llmDevice = String(args.llmDevice ?? args['llm-device'] ?? process.env.LLM_DEVICE ?? 'auto');
  const llmMaxTokens = toInt(
    args.llmMaxTokens ?? args['llm-max-tokens'] ?? process.env.LLM_MAX_TOKENS,
    800
  );
  const llmTimeoutMs = toInt(
    args.llmTimeoutMs ?? args['llm-timeout-ms'] ?? process.env.LLM_TIMEOUT_MS,
    180_000
  );
  const llmAllowNoKey = toBool(
    args.llmAllowNoKey ?? args['llm-allow-no-key'] ?? process.env.LLM_ALLOW_NO_KEY,
    false
  );
  const llmHfScriptPath = path.join(__dirname, 'llm_agent_hf.py');

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
      const originXZ = base.offset(offsetX, 0, offsetZ);
      const originY = findSurfaceY(bot, originXZ.x, originXZ.z, base.y);
      const origin = new Vec3(originXZ.x, originY, originXZ.z);
      console.log(`spawned at ${base.x} ${base.y} ${base.z}`);
      console.log(`building origin ${origin.x} ${origin.y} ${origin.z}`);

      if (enableViewer && viewerFn) {
        viewerFn(bot, { port: viewPort, firstPerson: false });
        console.log(`viewer: http://127.0.0.1:${viewPort}/`);
      } else if (enableViewer) {
        const reason = viewerLoadError ? `: ${viewerLoadError.message}` : '';
        console.warn(`viewer disabled (prismarine-viewer not available${reason})`);
      }

      const payload = await llmAgent(promptText, origin, {
        mode: llmMode,
        baseUrl: llmBaseUrl,
        apiKey: llmApiKey,
        model: llmModel,
        temperature: llmTemperature,
        topP: llmTopP,
        device: llmDevice,
        maxTokens: llmMaxTokens,
        timeoutMs: llmTimeoutMs,
        allowNoKey: llmAllowNoKey,
        hfScriptPath: llmHfScriptPath,
      });
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
