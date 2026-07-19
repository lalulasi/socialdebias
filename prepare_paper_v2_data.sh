#!/usr/bin/env bash
# Rebuild the three filtered + NLI-weighted adversarial training datasets.
# This script reuses existing raw rewrites and never calls a generation API.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-paper_v2_data}"
REPORT_DIR="results/${RUN_TAG}/data_preparation"
mkdir -p "${REPORT_DIR}"

required_files=(
  "data/sheepdog/news_articles/politifact_train.pkl"
  "data/sheepdog/news_articles/gossipcop_train.pkl"
  "data/weibo21_repo/data/train.pkl"
  "data/qwen_adv/politifact_train_adv.pkl"
  "data/qwen_adv/gossipcop_train_adv.pkl"
  "data/qwen_adv/weibo21_train_adv_deepseek.pkl"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] 缺少原始输入: ${path}"
    exit 1
  fi
done

"${PYTHON_BIN}" scripts/audit_experiment_data.py \
  --json_output "${REPORT_DIR}/before.json" \
  2>&1 | tee "${REPORT_DIR}/before.log"

for dataset in politifact gossipcop; do
  "${PYTHON_BIN}" scripts/filter_adversarial_v3.py \
    --original "data/sheepdog/news_articles/${dataset}_train.pkl" \
    --rewritten "data/qwen_adv/${dataset}_train_adv.pkl" \
    --output "data/qwen_adv/${dataset}_train_adv_filtered_paper_v2.pkl" \
    --entity_recall_threshold 0.7 \
    --semantic_threshold 0.65 \
    2>&1 | tee "${REPORT_DIR}/${dataset}_filter.log"
done

"${PYTHON_BIN}" scripts/filter_adversarial_v4_zh.py \
  --original data/weibo21_repo/data/train.pkl \
  --rewritten data/qwen_adv/weibo21_train_adv_deepseek.pkl \
  --output data/qwen_adv/weibo21_train_adv_filtered_paper_v2.pkl \
  --entity_recall_threshold 0.6 \
  --semantic_threshold 0.65 \
  2>&1 | tee "${REPORT_DIR}/weibo21_filter.log"

for dataset in politifact gossipcop; do
  "${PYTHON_BIN}" scripts/compute_nli_p_entail.py \
    --original "data/sheepdog/news_articles/${dataset}_train.pkl" \
    --rewritten "data/qwen_adv/${dataset}_train_adv_filtered_paper_v2.pkl" \
    --output "data/qwen_adv/${dataset}_p_entail_paper_v2.pkl" \
    --orig_format pkl_dict --max_length 512 --batch_size 8 \
    2>&1 | tee "${REPORT_DIR}/${dataset}_nli.log"
done

"${PYTHON_BIN}" scripts/compute_nli_p_entail.py \
  --original data/weibo21_repo/data/train.pkl \
  --rewritten data/qwen_adv/weibo21_train_adv_filtered_paper_v2.pkl \
  --output data/qwen_adv/weibo21_p_entail_paper_v2.pkl \
  --orig_format pkl_dataframe --max_length 512 --batch_size 8 \
  2>&1 | tee "${REPORT_DIR}/weibo21_nli.log"

"${PYTHON_BIN}" scripts/audit_experiment_data.py \
  --json_output "${REPORT_DIR}/after.json" \
  2>&1 | tee "${REPORT_DIR}/after.log"

echo ""
echo "训练对抗数据已补齐。报告目录: ${REPORT_DIR}"
echo "注意：training_ready 仍要求另外准备 NRC_EN 与 NRC_ZH。"
