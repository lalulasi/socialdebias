#!/bin/bash
# PolitiFact 消融实验：四种设置各跑三个随机种子。

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SEEDS=(42 2024 3407)
VARIANTS=("full" "no_grl" "no_consist" "no_both")
DATASETS=("politifact")

LOG_ROOT="./results/ablation/logs"
mkdir -p "${LOG_ROOT}"

echo "========================================"
echo "消融实验批量开始: $(date)"
echo "========================================"

COUNT=0
TOTAL=$((${#SEEDS[@]} * ${#VARIANTS[@]} * ${#DATASETS[@]}))
echo "总任务数: ${TOTAL}"

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
        --epochs 3 \
        --batch_size 16 \
        --save_dir results/ablation \
        --save_suffix "abl_${VAR}" \
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
echo "消融全部完成: $(date)"
echo "========================================"
