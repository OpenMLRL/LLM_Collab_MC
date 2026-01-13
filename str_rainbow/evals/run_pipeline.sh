#!/usr/bin/env bash
#
# str_rainbow Pipeline evaluation (Agent 1 then Agent 2 sees Agent 1 commands)
#

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CONFIG_FILE="${CONFIG_FILE:-evals/configs/pipeline_config.yaml}"

echo "=============================================="
echo "str_rainbow Pipeline Evaluation"
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

python evals/eval_pipeline.py --config "${CONFIG_FILE}" "$@"
STATUS=$?

echo ""
echo "=============================================="
echo "Done at: $(date) | Exit status: ${STATUS}"
echo "Results: evals/results"
echo "=============================================="

exit ${STATUS}
