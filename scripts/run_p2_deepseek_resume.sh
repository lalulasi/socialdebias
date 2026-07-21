#!/usr/bin/env bash
# Resume-safe DeepSeek zero-shot baseline. Inherits DEEPSEEK_API_KEY from launcher.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
batch_root=/autodl-fs/data/socialdebias/results/paper_v2_20260719
output_dir=${batch_root}/llm_baseline
model=${DEEPSEEK_MODEL:-deepseek-v4-flash}

cd "${project_root}"
: "${DEEPSEEK_API_KEY:?Run export DEEPSEEK_API_KEY before starting P2 DeepSeek}"
mkdir -p "${output_dir}"
OUT_DIR="${output_dir}" DEEPSEEK_MODEL="${model}" bash run_llm_baseline.sh
[[ -s "${output_dir}/summary.csv" ]] || { echo "[ERROR] Missing DeepSeek summary" >&2; exit 1; }
[[ "$(find "${output_dir}" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 11 ]] || {
  echo "[ERROR] DeepSeek JSON count is not 11" >&2
  exit 1
}
echo "P2_DEEPSEEK_READY=True"
