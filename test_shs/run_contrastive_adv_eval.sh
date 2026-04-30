#!/bin/bash
# 批量评估对比学习 ckpt 在对抗集上的鲁棒性
cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
LAMBDA_CONTRAST=(0.1 0.3 0.5)
DATASET="politifact"

LOG_ROOT="./results/contrastive_adv/logs"
mkdir -p "${LOG_ROOT}"

echo "对比学习对抗评估开始: $(date)"

COUNT=0
for LAMBDA in "${LAMBDA_CONTRAST[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    TAG="${DATASET}_lc${LAMBDA}_seed${SEED}"
    LOG_FILE="${LOG_ROOT}/${TAG}.log"
    
    echo ""
    echo "[${COUNT}/9] ${TAG}"
    
    python scripts/evaluate_contrastive_adv.py \
      --dataset ${DATASET} \
      --seed ${SEED} \
      --lambda_contrast ${LAMBDA} \
      > "${LOG_FILE}" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  [✓]"
    else
        echo "  [✗] 看日志: ${LOG_FILE}"
    fi
  done
done

echo ""
echo "对抗评估完成: $(date)"