#!/usr/bin/env bash
# Create/repair a complete ENDEF Python 3.10 environment in one command.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENDEF_ROOT="${1:-/autodl-fs/data/ENDEF-SIGIR2022}"
VENV_DIR="${ENDEF_VENV_DIR:-${ENDEF_ROOT}/endefvenv}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"

if [[ ! -d "${ENDEF_ROOT}/ENDEF_en" || ! -d "${ENDEF_ROOT}/ENDEF_ch" ]]; then
  echo "[ERROR] ENDEF source tree not found: ${ENDEF_ROOT}" >&2
  exit 1
fi

# Upstream grid_search.py creates logs/param but writes the final result to
# logs/json without creating that directory first.  Initialise all runtime
# output directories here so a completed training run does not fail at the
# final JSON write.
for ENDEF_CODE_DIR in "${ENDEF_ROOT}/ENDEF_en" "${ENDEF_ROOT}/ENDEF_ch"; do
  mkdir -p \
    "${ENDEF_CODE_DIR}/logs/json" \
    "${ENDEF_CODE_DIR}/logs/param" \
    "${ENDEF_CODE_DIR}/logs/event" \
    "${ENDEF_CODE_DIR}/param_model" \
    "${ENDEF_CODE_DIR}/backup_ckpt"
done

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  "${PYTHON_BOOTSTRAP}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --upgrade --force-reinstall \
  -r "${PROJECT_ROOT}/requirements-endef-py310.txt"

if ! python -c "import torch" >/dev/null 2>&1; then
  if [[ -n "${PYTORCH_INDEX_URL:-}" ]]; then
    python -m pip install 'torch>=2.1,<3' --index-url "${PYTORCH_INDEX_URL}"
  else
    python -m pip install 'torch>=2.1,<3'
  fi
fi

python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python -m pip check
python "${PROJECT_ROOT}/scripts/check_endef_environment.py" \
  --endef_root "${ENDEF_ROOT}" \
  --require_cuda

echo
echo "ENDEF environment installed: ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
