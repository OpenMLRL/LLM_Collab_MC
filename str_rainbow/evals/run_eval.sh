#!/usr/bin/env bash
#
# str_rainbow evaluation runner (IoU metric, multi-turn, multi-agent)
#
# Usage:
#   bash evals/run_eval.sh
#   bash evals/run_eval.sh --eval-split "[:8]" --verbose
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

CONFIG_FILE="${CONFIG_FILE:-evals/configs/str_rainbow_eval.yaml}"

echo "=============================================="
echo "str_rainbow Evaluation (IoU)"
echo "Started at: $(date)"
echo "Project dir: ${PROJECT_DIR}"
echo "Config: ${CONFIG_FILE}"
echo "=============================================="

# Activate conda env (same as training)
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate comlrl || {
    echo "Warning: failed to activate conda env 'comlrl'."
  }
else
  echo "Warning: conda not found; assuming environment already active."
fi

mkdir -p evals/results

python evals/eval_str_rainbow.py --config "${CONFIG_FILE}" "$@"

STATUS=$?

echo ""
echo "=============================================="
echo "str_rainbow eval completed at: $(date)"
echo "Exit status: ${STATUS}"
echo "Results dir: evals/results"
echo "=============================================="

exit ${STATUS}
