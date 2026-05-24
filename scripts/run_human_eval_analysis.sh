#!/bin/bash
# 分析人工评估标注结果（意见 17）
# 输入：标注完成的 xlsx
# 输出：results/human_eval/ 下三个文件 + 终端打印聚合指标
#
# 可选参数：--ig_json，若提供则额外输出"模型 IG 归因 vs 人类关键词"对齐分析

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
