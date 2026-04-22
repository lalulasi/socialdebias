#!/bin/bash
cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
DATASETS=("politifact" "gossipcop")

LOG_ROOT="./results/lstm/logs"
mkdir -p "${LOG_ROOT}"

echo "========================================"
echo "LSTM 批量实验开始: $(date)"
echo "========================================"

COUNT=0
TOTAL=$((${#SEEDS[@]} * ${#DATASETS[@]}))

for SEED in "${SEEDS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    COUNT=$((COUNT + 1))
    TAG="${DS}_seed${SEED}"
    LOG_FILE="${LOG_ROOT}/${TAG}.log"

    echo ""
    echo "[${COUNT}/${TOTAL}] LSTM ${TAG}"
    echo "  开始: $(date)"

    python scripts/train_lstm.py \
      --dataset ${DS} \
      --seed ${SEED} \
      --epochs 10 \
      > "${LOG_FILE}" 2>&1

    STATUS=$?
    if [ ${STATUS} -eq 0 ]; then
      echo "  [✓] 成功 $(date)"
    else
      echo "  [✗] 失败 $(date)，看日志：${LOG_FILE}"
    fi
  done
done

echo ""
echo "========================================"
echo "LSTM 全部完成: $(date)"
echo "========================================"
