#!/bin/bash
# PolitiFact 对比学习实验：三个随机种子与三组对比损失权重。

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SEEDS=(42 2024 3407)
LAMBDA_CONTRAST=(0.1 0.3 0.5)
DATASET="politifact"
LANGUAGE="en"

ADV_PKL="data/qwen_adv/politifact_train_adv_filtered_v2.pkl"

LOG_ROOT="./results/contrastive_logs"
mkdir -p "${LOG_ROOT}"

# 训练前确认两份配对数据都在
if [ ! -f "${ADV_PKL}" ]; then
    echo "ERROR: 对抗数据不存在: ${ADV_PKL}"
    exit 1
fi

echo "========================================"
echo "对比学习实验开始: $(date)"
echo "数据: ${ADV_PKL}"
echo "总任务: $((${#SEEDS[@]} * ${#LAMBDA_CONTRAST[@]}))"
echo "========================================"

COUNT=0
for LAMBDA in "${LAMBDA_CONTRAST[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    TAG="${DATASET}_lc${LAMBDA}_seed${SEED}"
    LOG_FILE="${LOG_ROOT}/${TAG}.log"
    
    echo ""
    echo "[${COUNT}/9] λ_contrast=${LAMBDA} seed=${SEED}"
    echo "  开始: $(date)"
    
    python scripts/train_socialdebias_contrastive.py \
      --dataset ${DATASET} \
      --language ${LANGUAGE} \
      --seed ${SEED} \
      --lambda_fact 1.0 \
      --lambda_bias 0.5 \
      --lambda_consist 0.3 \
      --lambda_contrast ${LAMBDA} \
      --temperature 0.07 \
      --adv_pkl ${ADV_PKL} \
      --save_suffix "lc${LAMBDA}" \
      > "${LOG_FILE}" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  [✓] 成功 $(date)"
    else
        echo "  [✗] 失败 $(date)，日志：${LOG_FILE}"
        tail -10 "${LOG_FILE}" | sed 's/^/    /'
    fi
  done
done

echo ""
echo "========================================"
echo "对比学习训练完成: $(date)"
echo "========================================"
echo ""
echo "下一步：跑对抗评估"
echo "  bash run_contrastive_adv_eval.sh"
