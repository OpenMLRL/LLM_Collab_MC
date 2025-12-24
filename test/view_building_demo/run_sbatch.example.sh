#!/usr/bin/env bash
set -euo pipefail

# Copy this file to run_sbatch.sh and fill in your own Slurm settings:
#   cp test/view_building_demo/run_sbatch.example.sh test/view_building_demo/run_sbatch.sh
#   chmod +x test/view_building_demo/run_sbatch.sh
#   vim test/view_building_demo/run_sbatch.sh
#
# Then submit:
#   ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION ./test/view_building_demo/run_sbatch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Slurm resources (set these for your cluster)
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NTASKS="${NTASKS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-64g}"
TIME="${TIME:-02:00:00}"
JOB_NAME="${JOB_NAME:-view_building_demo}"

SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SCRIPT_DIR}/slurm_logs}"
mkdir -p "${SLURM_LOG_DIR}"

# Conda env
CONDA_ENV="${CONDA_ENV:-comlrl}"
BASHRC_PATH="${BASHRC_PATH:-$HOME/.bashrc}"

# Minecraft server (must exist on the compute node filesystem, e.g. shared $HOME)
MC_DIR="${MC_DIR:-$HOME/mc-server}"
MC_JAR="${MC_JAR:-paper-1.19.2-307.jar}"
MC_PORT="${MC_PORT:-25565}"
BOT_USERNAME="${BOT_USERNAME:-executor_bot}"
MC_XMS="${MC_XMS:-1G}"
MC_XMX="${MC_XMX:-1G}"
LEVEL_NAME="${LEVEL_NAME:-view_building_demo_flat}"
LEVEL_TYPE="${LEVEL_TYPE:-flat}"

# Demo options
VIEW_PORT="${VIEW_PORT:-3000}"
LLM_MODE="${LLM_MODE:-hf}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
LLM_DEVICE="${LLM_DEVICE:-cuda}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-256}"
LLM_TIMEOUT_MS="${LLM_TIMEOUT_MS:-900000}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.2}"
LLM_TOP_P="${LLM_TOP_P:-0.95}"
ALLOW_CPU="${ALLOW_CPU:-0}"
EXTRA_NODE_ARGS="${EXTRA_NODE_ARGS:-}"
SRUN_ARGS="${SRUN_ARGS:---ntasks=1 --gpus-per-task=1}"

if [[ -z "${ACCOUNT}" || -z "${PARTITION}" ]]; then
  echo "ERROR: set ACCOUNT and PARTITION for sbatch." >&2
  echo "Example:" >&2
  echo "  ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION ./test/view_building_demo/run_sbatch.sh" >&2
  exit 2
fi

echo "Submitting Slurm job (server + demo in one job):"
echo "  job name:    ${JOB_NAME}"
echo "  logs:        ${SLURM_LOG_DIR}"

sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes="${NODES}" \
  --gpus-per-node="${GPUS_PER_NODE}" \
  --ntasks="${NTASKS}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --job-name="${JOB_NAME}" \
  --output="${SLURM_LOG_DIR}/%x-%j.out" \
  --export=ALL,REPO_DIR="${REPO_DIR}",CONDA_ENV="${CONDA_ENV}",BASHRC_PATH="${BASHRC_PATH}",MC_DIR="${MC_DIR}",MC_JAR="${MC_JAR}",MC_PORT="${MC_PORT}",BOT_USERNAME="${BOT_USERNAME}",MC_XMS="${MC_XMS}",MC_XMX="${MC_XMX}",LEVEL_NAME="${LEVEL_NAME}",LEVEL_TYPE="${LEVEL_TYPE}",VIEW_PORT="${VIEW_PORT}",LLM_MODE="${LLM_MODE}",LLM_MODEL="${LLM_MODEL}",LLM_DEVICE="${LLM_DEVICE}",LLM_MAX_TOKENS="${LLM_MAX_TOKENS}",LLM_TIMEOUT_MS="${LLM_TIMEOUT_MS}",LLM_TEMPERATURE="${LLM_TEMPERATURE}",LLM_TOP_P="${LLM_TOP_P}",ALLOW_CPU="${ALLOW_CPU}",EXTRA_NODE_ARGS="${EXTRA_NODE_ARGS}",SRUN_ARGS="${SRUN_ARGS}" \
  <<'SBATCH'
#!/usr/bin/env bash
set -eo pipefail

echo "job_id=${SLURM_JOB_ID:-}"
echo "node=$(hostname)"
echo "repo=${REPO_DIR}"

# Load user env (conda/java/node). This may be cluster-specific.
if [ -f "${BASHRC_PATH}" ]; then
  # shellcheck disable=SC1090
  source "${BASHRC_PATH}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found (set BASHRC_PATH or load conda module)" >&2
  exit 1
fi
if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: node not found in PATH" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH" >&2
  exit 1
fi

# Ensure deps exist (node_modules on shared filesystem)
if [ ! -d "${REPO_DIR}/node_modules" ]; then
  echo "ERROR: ${REPO_DIR}/node_modules not found. Run once on login node:" >&2
  echo "  cd ${REPO_DIR} && npm install" >&2
  exit 1
fi

# Resolve java
JAVA_BIN=""
if [ -n "${JAVA_HOME:-}" ] && [ -x "${JAVA_HOME}/bin/java" ]; then
  JAVA_BIN="${JAVA_HOME}/bin/java"
else
  JAVA_BIN="$(command -v java || true)"
fi
if [ -z "${JAVA_BIN}" ]; then
  echo "ERROR: java not found in PATH (set JAVA_HOME or load java module)" >&2
  exit 1
fi

# Sanity-check MC server files
if [ ! -d "${MC_DIR}" ]; then
  echo "ERROR: MC_DIR not found: ${MC_DIR}" >&2
  echo "Create it on a shared filesystem (e.g. $HOME) and put the server jar there." >&2
  exit 1
fi
if [ ! -f "${MC_DIR}/${MC_JAR}" ]; then
  echo "ERROR: Minecraft server jar not found: ${MC_DIR}/${MC_JAR}" >&2
  echo "Example setup (Paper 1.19.2 build 307):" >&2
  echo "  mkdir -p \"${MC_DIR}\" && cd \"${MC_DIR}\"" >&2
  echo "  curl -fsSL https://api.papermc.io/v2/projects/paper/versions/1.19.2/builds/307/downloads/paper-1.19.2-307.jar -o \"${MC_JAR}\"" >&2
  exit 1
fi

# Prepare MC server dir and config
mkdir -p "${MC_DIR}/logs"
if [ ! -f "${MC_DIR}/eula.txt" ]; then
  echo "eula=true" > "${MC_DIR}/eula.txt"
fi
if [ ! -f "${MC_DIR}/server.properties" ]; then
  cat > "${MC_DIR}/server.properties" <<EOF
online-mode=false
level-name=${LEVEL_NAME}
level-type=${LEVEL_TYPE}
server-port=${MC_PORT}
EOF
else
  sed -i -E "s/^online-mode=.*/online-mode=false/" "${MC_DIR}/server.properties" || true
  sed -i -E "s/^level-name=.*/level-name=${LEVEL_NAME}/" "${MC_DIR}/server.properties" || true
  sed -i -E "s/^level-type=.*/level-type=${LEVEL_TYPE}/" "${MC_DIR}/server.properties" || true
  sed -i -E "s/^server-port=.*/server-port=${MC_PORT}/" "${MC_DIR}/server.properties" || true
fi

# Ensure bot is OP (offline UUID)
OPS_JSON="${MC_DIR}/ops.json" MC_USERNAME="${BOT_USERNAME}" python3 - <<'PY'
import os, json, hashlib, uuid
ops_path = os.environ["OPS_JSON"]
username = os.environ["MC_USERNAME"]

data = []
if os.path.exists(ops_path):
    try:
        data = json.load(open(ops_path, "r", encoding="utf-8"))
    except Exception:
        data = []
if not isinstance(data, list):
    data = []

s = ("OfflinePlayer:" + username).encode("utf-8")
d = bytearray(hashlib.md5(s).digest())
d[6] = (d[6] & 0x0F) | 0x30
d[8] = (d[8] & 0x3F) | 0x80
uid = str(uuid.UUID(bytes=bytes(d)))

found = False
for e in data:
    if isinstance(e, dict) and e.get("name") == username:
        e["uuid"] = uid
        e["level"] = 4
        e.setdefault("bypassesPlayerLimit", False)
        found = True
if not found:
    data.append({"uuid": uid, "name": username, "level": 4, "bypassesPlayerLimit": False})

json.dump(data, open(ops_path, "w", encoding="utf-8"), indent=2)
print("ops.json updated:", username, uid)
PY

SERVER_LOG="${MC_DIR}/logs/view_building_demo_server_${SLURM_JOB_ID:-$$}.log"
cd "${MC_DIR}"
"${JAVA_BIN}" -Xms"${MC_XMS}" -Xmx"${MC_XMX}" -jar "${MC_JAR}" --nogui < /dev/null > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "server_pid=${SERVER_PID}"
echo "server_log=${SERVER_LOG}"

cleanup() {
  echo "stopping server..."
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for port
for i in {1..180}; do
  if (echo > /dev/tcp/127.0.0.1/"${MC_PORT}") >/dev/null 2>&1; then
    echo "server_ready"
    break
  fi
  sleep 1
  if [ "${i}" -eq 180 ]; then
    echo "server_not_ready; tail log:" >&2
    tail -n 120 "${SERVER_LOG}" || true
    exit 1
  fi
done

conda activate "${CONDA_ENV}"

# Confirm CUDA torch is available (otherwise generation can take a long time).
python3 - <<'PY'
import os
import torch

print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())

allow_cpu = os.environ.get("ALLOW_CPU", "0").lower() in ("1", "true", "yes")
if not torch.cuda.is_available() and not allow_cpu:
    raise SystemExit(
        "ERROR: torch.cuda.is_available() is False in this job. "
        "Install a CUDA-enabled PyTorch build or set ALLOW_CPU=1 to proceed on CPU."
    )
PY

NODE_HOST="$(hostname)"
echo "viewer_url=http://127.0.0.1:${VIEW_PORT}/"
echo "viewer_ssh_cmd=ssh -L ${VIEW_PORT}:127.0.0.1:${VIEW_PORT} ${USER}@${NODE_HOST}"

cd "${REPO_DIR}"
srun ${SRUN_ARGS} node test/view_building_demo/build_house_demo.cjs \
  --host 127.0.0.1 \
  --port "${MC_PORT}" \
  --username "${BOT_USERNAME}" \
  --viewer-port "${VIEW_PORT}" \
  --llm-mode "${LLM_MODE}" \
  --llm-model "${LLM_MODEL}" \
  --llm-device "${LLM_DEVICE}" \
  --llm-max-tokens "${LLM_MAX_TOKENS}" \
  --llm-timeout-ms "${LLM_TIMEOUT_MS}" \
  --llm-temperature "${LLM_TEMPERATURE}" \
  --llm-top-p "${LLM_TOP_P}" \
  ${EXTRA_NODE_ARGS}
SBATCH
