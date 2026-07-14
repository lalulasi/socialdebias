#!/bin/bash
# GossipCop 对比学习实验：三个随机种子与三组对比损失权重。

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SEEDS=(42 2024 3407)
LAMBDA_CONTRAST=(0.1 0.3 0.5)
DATASET="gossipcop"
LANGUAGE="en"

# 使用与对抗改写配对的训练集
ORIG_PKL="data/sheepdog/news_articles/gossipcop_train.pkl"
ADV_PKL="data/qwen_adv/gossipcop_train_adv_filtered_v2.pkl"

LOG_ROOT="./results/contrastive_logs_gc"
mkdir -p "${LOG_ROOT}"

# 训练前确认两份配对数据都在
if [ ! -f "${ORIG_PKL}" ]; then
    echo "ERROR: 原数据不存在: ${ORIG_PKL}"
    exit 1
fi
if [ ! -f "${ADV_PKL}" ]; then
    echo "ERROR: 对抗数据不存在: ${ADV_PKL}"
    exit 1
fi

echo "========================================"
echo "GossipCop 对比学习实验开始: $(date)"
echo "原数据: ${ORIG_PKL}"
echo "对抗数据: ${ADV_PKL}"
echo "总任务数: $((${#SEEDS[@]} * ${#LAMBDA_CONTRAST[@]}))"
echo "========================================"

COUNT=0
TOTAL=$((${#SEEDS[@]} * ${#LAMBDA_CONTRAST[@]}))

for LAMBDA in "${LAMBDA_CONTRAST[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    TAG="${DATASET}_lc${LAMBDA}_seed${SEED}"
    LOG_FILE="${LOG_ROOT}/${TAG}.log"
    
    echo ""
    echo "[${COUNT}/${TOTAL}] λ_contrast=${LAMBDA} seed=${SEED}"
    echo "  开始: $(date)"
    
    START_TS=$(date +%s)
    
    python scripts/train_socialdebias_contrastive.py \
      --dataset ${DATASET} \
      --language ${LANGUAGE} \
      --seed ${SEED} \
      --epochs 3 \
      --batch_size 4 \
      --lambda_fact 1.0 \
      --lambda_bias 0.5 \
      --lambda_consist 0.3 \
      --lambda_contrast ${LAMBDA} \
      --temperature 0.07 \
      --orig_pkl ${ORIG_PKL} \
      --adv_pkl ${ADV_PKL} \
      --save_suffix "lc${LAMBDA}" \
      > "${LOG_FILE}" 2>&1
    
    STATUS=$?
    ELAPSED=$(($(date +%s) - START_TS))
    
    if [ ${STATUS} -eq 0 ]; then
        echo "  [✓] 成功 用时 ${ELAPSED}s"
    else
        echo "  [✗] 失败 用时 ${ELAPSED}s 看日志：${LOG_FILE}"
        tail -10 "${LOG_FILE}" | sed 's/^/    /'
    fi
  done
done

echo ""
echo "========================================"
echo "GossipCop 对比学习训练完成: $(date)"
echo "========================================"
echo ""
echo "下一步：跑对抗评估"
echo "  bash run_contrastive_adv_eval_gossipcop.sh"
