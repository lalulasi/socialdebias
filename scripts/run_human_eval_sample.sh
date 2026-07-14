#!/bin/bash
# 从 PolitiFact 采样 50 条原文和 50 条对抗文本，生成标注表。

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
