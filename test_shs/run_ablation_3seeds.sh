#!/bin/bash
# 消融实验批量跑：4 变体 × 3 种子 × 1 数据集（politifact）
# 总共 12 次实验，每次 15-25 分钟，总时长 3-5 小时
#
# 使用: cd /root/autodl-tmp/socialdebias && bash run_ablation_3seeds.sh

cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
VARIANTS=("full" "no_grl" "no_consist" "no_both")
DATASETS=("politifact")      # 先只跑 politifact（效果最明显）

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
        > "${LOG_FILE}" 2>&1

      STATUS=$?
      if [ ${STATUS} -eq 0 ]; then
        echo "  [✓] 成功 $(date)"
      else
        echo "  [✗] 失败 $(date)，看日志：${LOG_FILE}"
      fi
    done
  done
done

echo ""
echo "========================================"
echo "消融全部完成: $(date)"
echo "========================================"