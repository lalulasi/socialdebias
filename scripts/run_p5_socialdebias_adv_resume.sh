#!/usr/bin/env bash
# Resume-safe P5 generation, filtering and three-model evaluation.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
batch_root=/autodl-fs/data/socialdebias/results/paper_v2_20260719
model_dir=${batch_root}/models
log_dir=${batch_root}/logs/socialdebias_adv
raw_dir=${project_root}/data/socialdebias_adv
filtered_dir=${raw_dir}/filtered
qwen_model=${QWEN_MODEL:-qwen3.6-plus-2026-04-02}
qwen_region=${QWEN_REGION:-intl}
deepseek_model=${DEEPSEEK_MODEL:-deepseek-v4-flash}

cd "${project_root}"
: "${DASHSCOPE_API_KEY:?Run export DASHSCOPE_API_KEY before starting P5}"
: "${DEEPSEEK_API_KEY:?Run export DEEPSEEK_API_KEY before starting P5}"
mkdir -p "${log_dir}" "${filtered_dir}"

SOCIALDEBIAS_ADV_OUT_DIR="${raw_dir}" SOCIALDEBIAS_ADV_LOG_DIR="${log_dir}" \
QWEN_MODEL="${qwen_model}" QWEN_REGION="${qwen_region}" DEEPSEEK_MODEL="${deepseek_model}" \
  bash run_socialdebias_adv.sh
[[ "$(find "${raw_dir}" -maxdepth 1 -type f -name '*_test_adv_*.pkl' | wc -l)" -eq 24 ]]

python -u scripts/filter_socialdebias_adv.py \
  --input_dir "${raw_dir}" --output_dir "${filtered_dir}" \
  2>&1 | tee "${log_dir}/filter_socialdebias_adv.log"
[[ -s "${filtered_dir}/filter_report.csv" ]]
[[ "$(find "${filtered_dir}" -maxdepth 1 -type f -name '*.pkl' | wc -l)" -eq 24 ]]

python -u scripts/evaluate_socialdebias_adv.py \
  --filtered_dir "${filtered_dir}" \
  --orig_dir data/sheepdog/news_articles --weibo21_orig data/weibo21_repo/data/test.pkl \
  --ckpt_dir "${model_dir}" --seeds 42 2024 3407 --sd_suffix surface_all \
  --deepseek_model "${deepseek_model}" --output "${batch_root}/socialdebias_adv_eval.csv" \
  2>&1 | tee "${log_dir}/socialdebias_adv_eval.log"
[[ -s "${batch_root}/socialdebias_adv_eval.csv" ]]
echo "P5_SOCIALDEBIAS_ADV_READY=True"
