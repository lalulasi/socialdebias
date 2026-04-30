#!/bin/bash
# 消融实验 B1 版（延长训练 + 减小 λ_bias）
# 4 变体 × 3 种子 × epoch=5，约 1.5-2 小时
# 用独立输出目录避免覆盖 B0 结果
# 使用: bash run_ablation_b1.sh

cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
VARIANTS=("full" "no_grl" "no_consist" "no_both")
DATASETS=("politifact")

EPOCHS=5
OVERRIDE_BIAS=0.3       # 从 0.5 降到 0.3
OUTPUT_DIR="results/ablation_b1"   # 新目录，不覆盖 B0

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

      echo ""
      echo "[${COUNT}/${TOTAL}] ${TAG}"
      echo "  开始: $(date)"

      python scripts/train_ablation.py \
        --dataset ${DS} \
        --variant ${VAR} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size 16 \
        --override_lambda_bias ${OVERRIDE_BIAS} \
        --output_dir ${OUTPUT_DIR} \
        > "${LOG_FILE}" 2>&1

      STATUS=$?
      if [ ${STATUS} -eq 0 ]; then
        echo "  [✓] 成功 $(date)"
      else
        echo "  [✗] 失败 $(date)，日志: ${LOG_FILE}"
      fi
    done
  done
done

echo ""
echo "========================================"
echo "消融 B1 完成: $(date)"
echo "========================================"