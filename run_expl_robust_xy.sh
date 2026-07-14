#!/bin/bash
# 逐样本提取 PolitiFact 原文与四种对抗改写的归因一致性和预测变化。
set -e
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p results/logs
python scripts/extract_expl_robust_xy.py \
    --orig_pkl data/sheepdog/news_articles/politifact_test.pkl \
    --adv_tpl  "data/sheepdog/adversarial_test/politifact_test_adv_{v}.pkl" \
    --variants A,B,C,D \
    --ckpt     results/models/socialdebias_politifact_en_seed42_surface.pt \
    --output   results/expl_robust_xy_politifact_ALL.csv \
    2>&1 | tee results/logs/expl_robust_xy_all.log
