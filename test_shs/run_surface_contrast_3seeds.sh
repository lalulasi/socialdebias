#!/bin/bash
# SocialDebias + 17 维表层特征 + InfoNCE 对比学习（论文最终方法）
# 6 次训练 × ~6-8 分钟（含 InfoNCE）≈ 1-1.5 小时

cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
LANGUAGE="en"

LOG_ROOT="./results/surface_contrast_logs"
mkdir -p "${LOG_ROOT}"

echo "========================================"
echo "SD + Surface + InfoNCE 实验开始: $(date)"
echo "========================================"

COUNT=0
TOTAL=6

# PolitiFact
DATASET="politifact"
ORIG_PKL="data/sheepdog/news_articles/politifact_train.pkl"
ADV_PKL="data/qwen_adv/politifact_train_adv_filtered_v2.pkl"

for SEED in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    TAG="${DATASET}_seed${SEED}_surface_contrast"
    LOG_FILE="${LOG_ROOT}/${TAG}.log"
    
    echo ""
    echo "[${COUNT}/${TOTAL}] ${TAG}"
    
    START_TS=$(date +%s)
    
    python scripts/train_socialdebias_surface.py \
      --dataset ${DATASET} \
      --language ${LANGUAGE} \
      --seed ${SEED} \
      --epochs 3 \
      --batch_size 4 \
      --lambda_fact 1.0 \
      --lambda_bias 0.5 \
      --lambda_consist 0.3 \
      --surface_feat_dim 8 \
      --use_contrastive \
      --lambda_contrast 0.3 \
      --temperature 0.07 \
      --orig_pkl ${ORIG_PKL} \
      --adv_pkl ${ADV_PKL} \
      --save_suffix surface_contrast \
      > "${LOG_FILE}" 2>&1
    
    STATUS=$?
    ELAPSED=$(($(date +%s) - START_TS))
    
    if [ ${STATUS} -eq 0 ]; then
        echo "  [✓] 成功 用时 ${ELAPSED}s"
    else
        echo "  [✗] 失败 看日志：${LOG_FILE}"
        tail -10 "${LOG_FILE}" | sed 's/^/    /'
    fi
done

# GossipCop（用采样数据）
DATASET="gossipcop"
ORIG_PKL="data/sheepdog/news_articles/gossipcop_train_sampled1000.pkl"
ADV_PKL="data/qwen_adv/gossipcop_train_adv_filtered.pkl"

for SEED in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    TAG="${DATASET}_seed${SEED}_surface_contrast"
    LOG_FILE="${LOG_ROOT}/${TAG}.log"
    
    echo ""
    echo "[${COUNT}/${TOTAL}] ${TAG}"
    
    START_TS=$(date +%s)
    
    python scripts/train_socialdebias_surface.py \
      --dataset ${DATASET} \
      --language ${LANGUAGE} \
      --seed ${SEED} \
      --epochs 3 \
      --batch_size 4 \
      --lambda_fact 1.0 \
      --lambda_bias 0.5 \
      --lambda_consist 0.3 \
      --surface_feat_dim 8 \
      --use_contrastive \
      --lambda_contrast 0.3 \
      --temperature 0.07 \
      --orig_pkl ${ORIG_PKL} \
      --adv_pkl ${ADV_PKL} \
      --save_suffix surface_contrast \
      > "${LOG_FILE}" 2>&1
    
    STATUS=$?
    ELAPSED=$(($(date +%s) - START_TS))
    
    if [ ${STATUS} -eq 0 ]; then
        echo "  [✓] 成功 用时 ${ELAPSED}s"
    else
        echo "  [✗] 失败 看日志：${LOG_FILE}"
        tail -10 "${LOG_FILE}" | sed 's/^/    /'
    fi
done

echo ""
echo "========================================"
echo "完成: $(date)"
echo "========================================"