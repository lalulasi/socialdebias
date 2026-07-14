#!/bin/bash
# B1 消融设置：训练五轮，并将启用偏置分支的设置改为较小权重。

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SEEDS=(42 2024 3407)
VARIANTS=("full" "no_grl" "no_consist" "no_both")
DATASETS=("politifact")

EPOCHS=5
OVERRIDE_BIAS=0.3
OUTPUT_DIR="results/ablation_b1"

LOG_ROOT="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_ROOT}"

echo "========================================"
echo "消融 B1 开始: $(date)"
echo "配置: epochs=${EPOCHS}, λ_bias=${OVERRIDE_BIAS}"
echo "========================================"

COUNT=0
TOTAL=$((${#SEEDS[@]} * ${#VARIANTS[@]} * ${#DATASETS[@]}))

for DS in "${DATASETS[@]}"; do
  for VAR in "${VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      COUNT=$((COUNT + 1))
      TAG="${DS}_${VAR}_seed${SEED}"
      LOG_FILE="${LOG_ROOT}/${TAG}.log"
      BIAS=0
      CONSIST=0
      if [ "${VAR}" = "full" ] || [ "${VAR}" = "no_consist" ]; then
        BIAS=${OVERRIDE_BIAS}
      fi
      if [ "${VAR}" = "full" ] || [ "${VAR}" = "no_grl" ]; then
        CONSIST=0.3
      fi

      echo ""
      echo "[${COUNT}/${TOTAL}] ${TAG}"
      echo "  开始: $(date)"

      python scripts/train_ablation.py \
        --dataset ${DS} \
        --variant ${VAR} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size 16 \
        --lambda_fact 1.0 \
        --lambda_bias ${BIAS} \
        --lambda_consist ${CONSIST} \
        --save_dir ${OUTPUT_DIR} \
        --save_suffix "b1_${VAR}" \
        > "${LOG_FILE}" 2>&1

      STATUS=$?
      if [ ${STATUS} -eq 0 ]; then
        echo "  完成 $(date)"
      else
        echo "  失败 $(date)，日志：${LOG_FILE}"
      fi
    done
  done
done

echo ""
echo "========================================"
echo "消融 B1 完成: $(date)"
echo "========================================"
