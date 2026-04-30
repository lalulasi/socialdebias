#!/bin/bash
# 批量评估表层特征 ckpt 在对抗集上的鲁棒性
cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
DATASETS=("politifact" "gossipcop")
SUFFIXES=("surface" "surface_contrast")  # 12 个 ckpt

LOG_ROOT="/root/autodl-tmp/socialdebias/results/surface_adv/logs"
mkdir -p "${LOG_ROOT}"

echo "对抗评估开始: $(date)"

COUNT=0
TOTAL=$((${#SEEDS[@]} * ${#DATASETS[@]} * ${#SUFFIXES[@]}))

for SUFFIX in "${SUFFIXES[@]}"; do
  for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      COUNT=$((COUNT + 1))
      TAG="${DS}_seed${SEED}_${SUFFIX}"
      LOG_FILE="${LOG_ROOT}/${TAG}.log"
      
      echo ""
      echo "[${COUNT}/${TOTAL}] ${TAG}"
      
      python scripts/evaluate_surface_adv.py \
        --dataset ${DS} \
        --seed ${SEED} \
        --save_suffix ${SUFFIX} \
        > "${LOG_FILE}" 2>&1
      
      if [ $? -eq 0 ]; then
          echo "  [✓]"
      else
          echo "  [✗] 看日志: ${LOG_FILE}"
      fi
    done
  done
done
echo ""
echo "评估完成: $(date)"