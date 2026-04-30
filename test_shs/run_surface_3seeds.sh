#!/bin/bash
# SocialDebias + 17 维表层特征（意见 1 + 意见 8 实现）
# 6 次训练 × ~5 分钟（PolitiFact）/ 25 分钟（GossipCop）≈ 1.5-2 小时
#
# 使用: cd /root/autodl-tmp/socialdebias && bash run_surface_3seeds.sh

cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
DATASETS=("politifact")
LANGUAGE="en"

LOG_ROOT="./results/surface_logs"
mkdir -p "${LOG_ROOT}"

echo "========================================"
echo "SD + 17 维表层特征实验开始: $(date)"
echo "========================================"

COUNT=0
TOTAL=$((${#SEEDS[@]} * ${#DATASETS[@]}))

for DS in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    TAG="${DS}_seed${SEED}_surface"
    LOG_FILE="${LOG_ROOT}/${TAG}.log"
    
    echo ""
    echo "[${COUNT}/${TOTAL}] ${TAG}"
    echo "  开始: $(date)"
    
    START_TS=$(date +%s)
    
    python scripts/train_socialdebias_surface.py \
      --dataset ${DS} \
      --language ${LANGUAGE} \
      --seed ${SEED} \
      --epochs 3 \
      --batch_size 4 \
      --lambda_fact 1.0 \
      --lambda_bias 0.5 \
      --lambda_consist 0.3 \
      --surface_feat_dim 8 \
      --save_suffix surface \
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
echo "完成: $(date)"
echo "========================================"