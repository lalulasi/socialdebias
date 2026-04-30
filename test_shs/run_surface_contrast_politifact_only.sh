#!/bin/bash
cd /root/autodl-tmp/socialdebias
SEEDS=(42 2024 3407)
LOG_ROOT="./results/surface_contrast_logs"
mkdir -p "${LOG_ROOT}"
echo "PolitiFact only: $(date)"
for SEED in "${SEEDS[@]}"; do
    TAG="politifact_seed${SEED}_surface_contrast"
    echo "[$SEED] $(date)"
    python scripts/train_socialdebias_surface.py \
      --dataset politifact --language en --seed ${SEED} \
      --epochs 3 --batch_size 4 \
      --lambda_fact 1.0 --lambda_bias 0.5 --lambda_consist 0.3 \
      --surface_feat_dim 8 \
      --use_contrastive --lambda_contrast 0.3 --temperature 0.07 \
      --orig_pkl data/sheepdog/news_articles/politifact_train.pkl \
      --adv_pkl data/qwen_adv/politifact_train_adv_filtered_v2.pkl \
      --save_suffix surface_contrast \
      > "${LOG_ROOT}/${TAG}.log" 2>&1
    [ $? -eq 0 ] && echo "  [✓]" || echo "  [✗]"
done
echo "Done: $(date)"
