#!/usr/bin/env bash
# Source this file to initialize or restore one paper-v2 experiment batch.

paper_v2_exp_id="${1:-paper_v2_20260719}"
paper_v2_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${paper_v2_project_root}" || return 1

if [ -f socialvenv/bin/activate ]; then
  # shellcheck disable=SC1091
  source socialvenv/bin/activate
elif [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[ERROR] No virtual environment found. Run: python3 -m venv socialvenv" >&2
  return 1 2>/dev/null || exit 1
fi

export EXP_ID="${paper_v2_exp_id}"
export RUN_ROOT="results/${EXP_ID}"
export MODEL_DIR="${RUN_ROOT}/models"
export LOG_DIR="${RUN_ROOT}/logs"
export NRC_EN="data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
export NRC_ZH="data/lexicons/NRC-Emotion-Lexicon-ZH.tsv"

mkdir -p \
  "${MODEL_DIR}" "${LOG_DIR}" \
  "${RUN_ROOT}/lstm" \
  "${RUN_ROOT}/bert_adv" \
  "${RUN_ROOT}/surface_adv" \
  "${RUN_ROOT}/ablation_adv" \
  "${RUN_ROOT}/explanation" \
  "${RUN_ROOT}/llm_baseline" \
  "${RUN_ROOT}/human_eval" \
  "${RUN_ROOT}/manifests"

if [ ! -f "${RUN_ROOT}/manifests/git_commit.txt" ]; then
  git rev-parse HEAD > "${RUN_ROOT}/manifests/git_commit.txt"
  git status --short > "${RUN_ROOT}/manifests/git_status.txt"
  python --version > "${RUN_ROOT}/manifests/python_version.txt" 2>&1
  python -m pip freeze > "${RUN_ROOT}/manifests/pip_freeze.txt"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${RUN_ROOT}/manifests/nvidia_smi.txt"
  fi
fi

echo "[paper-v2] environment ready"
echo "  project=${paper_v2_project_root}"
echo "  EXP_ID=${EXP_ID}"
echo "  RUN_ROOT=${RUN_ROOT}"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  LOG_DIR=${LOG_DIR}"
