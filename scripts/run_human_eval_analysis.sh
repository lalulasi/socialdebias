#!/bin/bash
# 分析人工标注结果。
# 输入为完成标注的 xlsx，结果写入 results/human_eval/。
# 传入 --ig_json 时，额外比较模型归因和人工关键词。

set -e

mkdir -p results/human_eval logs

INPUT_FILE="results/human_eval/politifact_pre_annotated_task.xlsx"
IG_JSON=""  # 可选：results/explanation/politifact_surface_seed42.json

CMD="python scripts/analyze_human_eval.py \
    --input ${INPUT_FILE} \
    --output_dir results/human_eval/ \
    --uncertain_score 0.5"

if [ -n "${IG_JSON}" ] && [ -f "${IG_JSON}" ]; then
    CMD="${CMD} --ig_json ${IG_JSON} --topk 10"
fi

eval ${CMD} 2>&1 | tee logs/analyze_human_eval.log
