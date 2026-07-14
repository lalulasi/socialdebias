#!/bin/bash
# 评测 B1 消融模型在对抗集上的表现。
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SEEDS=(42 2024 3407)
VARIANTS=("full" "no_grl" "no_consist" "no_both")
DATASETS=("politifact")

CKPT_DIR="./results/ablation_b1"
OUTPUT_DIR="results/ablation_adv_b1"
LOG_ROOT="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_ROOT}"

COUNT=0
TOTAL=$((${#SEEDS[@]} * ${#VARIANTS[@]} * ${#DATASETS[@]}))
echo "B1 对抗评估开始: $(date)"

for DS in "${DATASETS[@]}"; do
  for VAR in "${VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      COUNT=$((COUNT + 1))
      TAG="${DS}_${VAR}_seed${SEED}"
      CKPT="${CKPT_DIR}/socialdebias_${DS}_en_seed${SEED}_b1_${VAR}.pt"
      LOG_FILE="${LOG_ROOT}/${TAG}.log"

      echo "[${COUNT}/${TOTAL}] ${TAG}"
      if [ ! -f "${CKPT}" ]; then
        echo "  检查点不存在"
        continue
      fi

      python scripts/evaluate_ablation_adv.py \
        --ckpt "${CKPT}" \
        --dataset "${DS}" \
        --save_suffix "b1_${VAR}" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        > "${LOG_FILE}" 2>&1

      if [ $? -eq 0 ]; then
        echo "  完成"
      else
        echo "  失败，日志：${LOG_FILE}"
      fi
    done
  done
done
echo "B1 对抗评估完成: $(date)"
