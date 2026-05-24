#!/bin/bash
# 采样 PolitiFact 50 条用于意见 17 人工评估
# 输出：results/human_eval/politifact_human_eval_template.csv（101 行）

set -e

mkdir -p results/human_eval logs

python scripts/sample_human_eval.py \
    --test_pkl data/sheepdog/news_articles/politifact_test.pkl \
    --adv_pkl  data/sheepdog/adversarial_test/politifact_test_adv_C.pkl \
    --n_samples 50 \
    --seed 42 \
    --output results/human_eval/politifact_human_eval_template.csv \
    2>&1 | tee logs/sample_human_eval.log

echo ""
echo "下一步：用 Excel / Numbers 打开 CSV，填 human_keywords / human_judgment / confidence / notes 四列"
