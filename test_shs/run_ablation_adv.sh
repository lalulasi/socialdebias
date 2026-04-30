#!/bin/bash
# 批量评估 12 个消融 ckpt 在对抗集上的鲁棒性
# 使用: cd /root/autodl-tmp/socialdebias && bash run_ablation_adv.sh

cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
VARIANTS=("full" "no_grl" "no_consist" "no_both")
DATASETS=("politifact")

LOG_ROOT="./results/ablation_adv/logs"
mkdir -p "${LOG_ROOT}"

echo "========================================"
echo "消融对抗集评估开始: $(date)"
echo "========================================"

COUNT=0
TOTAL=$((${#SEEDS[@]} * ${#VARIANTS[@]} * ${#DATASETS[@]}))
echo "总任务数: ${TOTAL}"

for DS in "${DATASETS[@]}"; do
  for VAR in "${VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      COUNT=$((COUNT + 1))
      TAG="${DS}_${VAR}_seed${SEED}"
      CKPT="./results/ablation/ablation_${DS}_${VAR}_seed${SEED}.pt"
      LOG_FILE="${LOG_ROOT}/${TAG}.log"

      echo ""
      echo "[${COUNT}/${TOTAL}] ${TAG}"

      if [ ! -f "${CKPT}" ]; then
        echo "  [✗] ckpt 不存在: ${CKPT}"
        continue
      fi

      python scripts/evaluate_ablation_adv.py \
        --ckpt "${CKPT}" \
        --dataset "${DS}" \
        --variant "${VAR}" \
        --seed "${SEED}" \
        > "${LOG_FILE}" 2>&1

      STATUS=$?
      if [ ${STATUS} -eq 0 ]; then
        echo "  [✓] 成功"
      else
        echo "  [✗] 失败，看日志: ${LOG_FILE}"
      fi
    done
  done
done

echo ""
echo "========================================"
echo "消融对抗集评估完成: $(date)"
echo "========================================"