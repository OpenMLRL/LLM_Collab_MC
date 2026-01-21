#!/usr/bin/env bash
#
# str_rainbow GRPO-trained model evaluation (IoU metric)
#
# Usage:
#   bash evals/run_grpo_eval.sh
#   bash evals/run_grpo_eval.sh --eval-split "[:16]" --verbose
#
# Model resolution:
#   - Uses aliases from evals/constants.py (default alias: grpo)
#   - Override via MODEL env or STR_RAINBOW_GRPO_MODEL for a full HF path
#

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CONFIG_FILE="${CONFIG_FILE:-evals/configs/str_rainbow_eval.yaml}"
MODEL_ALIAS="${MODEL:-${STR_RAINBOW_GRPO_MODEL_ALIAS:-grpo}}"

echo "=============================================="
echo "str_rainbow GRPO Evaluation"
echo "Started at: $(date)"
echo "Project dir: ${PROJECT_DIR}"
echo "Config: ${CONFIG_FILE}"
echo "Model alias/path: ${MODEL_ALIAS}"
echo "=============================================="

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate comlrl || echo "Warning: failed to activate conda env 'comlrl'."
else
  echo "Warning: conda not found; assuming environment already active."
fi

mkdir -p evals/results

python evals/eval_str_rainbow.py --config "${CONFIG_FILE}" --model "${MODEL_ALIAS}" "$@"
STATUS=$?

echo ""
echo "=============================================="
echo "Done at: $(date) | Exit status: ${STATUS}"
echo "Results: evals/results"
echo "=============================================="

exit ${STATUS}
