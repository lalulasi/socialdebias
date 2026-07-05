#!/bin/bash
# 解释一致性(X) × 鲁棒性(Y) 逐样本联合提取 —— PolitiFact / adv_A,B,C,D 四变体 / surface seed42
set -e
cd /root/autodl-tmp/socialdebias   # ← 按你的实际路径改（Mac 本地也可，走 CPU，约2小时）
mkdir -p results/logs
python scripts/extract_expl_robust_xy.py \
    --orig_pkl data/sheepdog/news_articles/politifact_test.pkl \
    --adv_tpl  "data/sheepdog/adversarial_test/politifact_test_adv_{v}.pkl" \
    --variants A,B,C,D \
    --ckpt     results/models/socialdebias_politifact_en_seed42_surface.pt \
    --output   results/expl_robust_xy_politifact_ALL.csv \
    2>&1 | tee results/logs/expl_robust_xy_all.log
