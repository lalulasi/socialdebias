#!/bin/bash
# GossipCop 对比学习 ckpt 的对抗评估
cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
LAMBDA_CONTRAST=(0.1 0.3 0.5)
DATASET="gossipcop"

LOG_ROOT="./results/contrastive_adv_gc/logs"
mkdir -p "${LOG_ROOT}"

echo "GossipCop 对抗评估开始: $(date)"

COUNT=0
TOTAL=$((${#SEEDS[@]} * ${#LAMBDA_CONTRAST[@]}))

for LAMBDA in "${LAMBDA_CONTRAST[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    TAG="${DATASET}_lc${LAMBDA}_seed${SEED}"
    LOG_FILE="${LOG_ROOT}/${TAG}.log"
    
    echo "[${COUNT}/${TOTAL}] ${TAG}"
    
    python scripts/evaluate_contrastive_adv.py \
      --dataset ${DATASET} \
      --seed ${SEED} \
      --lambda_contrast ${LAMBDA} \
      --output_dir results/contrastive_adv_gc \
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