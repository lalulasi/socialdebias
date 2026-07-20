#!/usr/bin/env bash
# Source this file to initialize or restore one paper-v2 experiment batch.

paper_v2_exp_id="${1:-paper_v2_20260719}"
paper_v2_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${paper_v2_project_root}" || return 1

paper_v2_requested_venv="${SOCIALDEBIAS_VENV_DIR:-}"
paper_v2_python_env=""

if [ -n "${paper_v2_requested_venv}" ]; then
  if [ ! -f "${paper_v2_requested_venv}/bin/activate" ]; then
    echo "[ERROR] SOCIALDEBIAS_VENV_DIR has no bin/activate: ${paper_v2_requested_venv}" >&2
    return 1 2>/dev/null || exit 1
  fi
  # shellcheck disable=SC1090
  source "${paper_v2_requested_venv}/bin/activate"
  paper_v2_python_env="${paper_v2_requested_venv}"
elif [ -f socialvenv/bin/activate ]; then
  # shellcheck disable=SC1091
  source socialvenv/bin/activate
  paper_v2_python_env="${paper_v2_project_root}/socialvenv"
elif [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  paper_v2_python_env="${paper_v2_project_root}/.venv"
elif [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
  paper_v2_python_env="${paper_v2_project_root}/venv"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  paper_v2_python_env="${VIRTUAL_ENV} (already active)"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  paper_v2_python_env="${CONDA_PREFIX} (active conda)"
elif command -v python >/dev/null 2>&1; then
  paper_v2_python_env="$(command -v python) (unmanaged current Python)"
  echo "[WARN] No project virtual environment found; using current Python." >&2
  echo "       Before further training, create socialvenv or set SOCIALDEBIAS_VENV_DIR." >&2
else
  echo "[ERROR] No usable Python environment found. Run: python3 -m venv socialvenv" >&2
  return 1 2>/dev/null || exit 1
fi

export EXP_ID="${paper_v2_exp_id}"
# Keep result paths stable after callers cd into an external baseline repo.
export RUN_ROOT="${paper_v2_project_root}/results/${EXP_ID}"
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
echo "  python_env=${paper_v2_python_env}"
echo "  EXP_ID=${EXP_ID}"
echo "  RUN_ROOT=${RUN_ROOT}"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  LOG_DIR=${LOG_DIR}"
