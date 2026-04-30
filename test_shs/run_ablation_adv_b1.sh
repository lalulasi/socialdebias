#!/bin/bash
# B1 版消融 ckpt 的对抗评估
cd /root/autodl-tmp/socialdebias

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
      CKPT="${CKPT_DIR}/ablation_${DS}_${VAR}_seed${SEED}.pt"
      LOG_FILE="${LOG_ROOT}/${TAG}.log"

      echo "[${COUNT}/${TOTAL}] ${TAG}"
      if [ ! -f "${CKPT}" ]; then
        echo "  [x] ckpt 不存在"
        continue
      fi

      python scripts/evaluate_ablation_adv.py \
        --ckpt "${CKPT}" \
        --dataset "${DS}" \
        --variant "${VAR}" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        > "${LOG_FILE}" 2>&1

      if [ $? -eq 0 ]; then
        echo "  [OK]"
      else
        echo "  [FAIL] 日志 ${LOG_FILE}"
      fi
    done
  done
done
echo "B1 对抗评估完成: $(date)"
