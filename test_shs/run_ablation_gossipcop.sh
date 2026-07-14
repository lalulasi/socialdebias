#!/bin/bash
# GossipCop 消融实验：四种设置各跑三个随机种子。

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SEEDS=(42 2024 3407)
VARIANTS=("full" "no_grl" "no_consist" "no_both")
DATASETS=("gossipcop")

EPOCHS=3
BATCH_SIZE=16
OUTPUT_DIR="results/ablation_gossipcop"

LOG_ROOT="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_ROOT}"

echo "========================================"
echo "GossipCop 消融实验开始: $(date)"
echo "数据集: ${DATASETS[*]}"
echo "变体: ${VARIANTS[*]}"
echo "种子: ${SEEDS[*]}"
echo "训练轮数: ${EPOCHS}，batch size: ${BATCH_SIZE}"
echo "输出: ${OUTPUT_DIR}"
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

      START_TS=$(date +%s)

      python scripts/train_ablation.py \
        --dataset ${DS} \
        --variant ${VAR} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --save_dir ${OUTPUT_DIR} \
        --save_suffix "abl_${VAR}" \
        > "${LOG_FILE}" 2>&1

      STATUS=$?
      END_TS=$(date +%s)
      ELAPSED=$((END_TS - START_TS))

      if [ ${STATUS} -eq 0 ]; then
        echo "  完成 $(date)，用时 ${ELAPSED}s"
      else
        echo "  失败 $(date)，日志：${LOG_FILE}"
        tail -5 "${LOG_FILE}" | sed 's/^/    /'
      fi
    done
  done
done

echo ""
echo "========================================"
echo "GossipCop 消融完成: $(date)"
echo "结果目录: ${OUTPUT_DIR}"
echo "========================================"
