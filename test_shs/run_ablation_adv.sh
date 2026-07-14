#!/bin/bash
# 批量评测 PolitiFact 消融模型在对抗集上的表现。

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

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
      CKPT="./results/ablation/socialdebias_${DS}_en_seed${SEED}_abl_${VAR}.pt"
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
        --save_suffix "abl_${VAR}" \
        --seed "${SEED}" \
        > "${LOG_FILE}" 2>&1

      STATUS=$?
      if [ ${STATUS} -eq 0 ]; then
        echo "  完成"
      else
        echo "  失败，日志：${LOG_FILE}"
      fi
    done
  done
done

echo ""
echo "========================================"
echo "消融对抗集评估完成: $(date)"
echo "========================================"
