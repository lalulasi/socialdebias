#!/bin/bash
# 消融实验在 SheepDog GossipCop 上重做
# 4 变体 × 3 种子 × 3 epoch，预计 4-6 小时
# 使用: nohup bash run_ablation_gossipcop.sh > run.out 2>&1 &

cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
VARIANTS=("full" "no_grl" "no_consist" "no_both")
DATASETS=("gossipcop")

EPOCHS=3
BATCH_SIZE=16
# 不加 --override_lambda_bias，使用 LAMBDA_MAP 默认（full 时 bias=0.5，和 exp002 一致）
OUTPUT_DIR="results/ablation_gossipcop"

LOG_ROOT="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_ROOT}"

echo "========================================"
echo "GossipCop 消融实验开始: $(date)"
echo "数据集: ${DATASETS[*]}"
echo "变体: ${VARIANTS[*]}"
echo "种子: ${SEEDS[*]}"
echo "训练: ${EPOCHS} epochs, batch=${BATCH_SIZE}"
echo "λ: 使用 LAMBDA_MAP 默认 (full=0.5/0.3, no_grl=0/0.3, no_consist=0.5/0, no_both=0/0)"
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
        --output_dir ${OUTPUT_DIR} \
        > "${LOG_FILE}" 2>&1

      STATUS=$?
      END_TS=$(date +%s)
      ELAPSED=$((END_TS - START_TS))

      if [ ${STATUS} -eq 0 ]; then
        echo "  [✓] 成功 $(date)  用时 ${ELAPSED}s"
      else
        echo "  [✗] 失败 $(date)  日志: ${LOG_FILE}"
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