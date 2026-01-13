#!/usr/bin/env bash
#
# str_rainbow Single-agent evaluation (IoU metric)
#

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CONFIG_FILE="${CONFIG_FILE:-evals/configs/single_agent_config.yaml}"

echo "=============================================="
echo "str_rainbow Single-Agent Evaluation"
echo "Started at: $(date)"
echo "Config: ${CONFIG_FILE}"
echo "=============================================="

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate comlrl || echo "Warning: failed to activate conda env 'comlrl'."
else
  echo "Warning: conda not found; assuming environment already active."
fi

mkdir -p evals/results

python evals/eval_single_agent.py --config "${CONFIG_FILE}" "$@"
STATUS=$?

echo ""
echo "=============================================="
echo "Done at: $(date) | Exit status: ${STATUS}"
echo "Results: evals/results"
echo "=============================================="

exit ${STATUS}
