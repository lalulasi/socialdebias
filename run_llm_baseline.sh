#!/bin/bash
# DeepSeek 零样本基线：评测干净集和 SheepDog 对抗测试集。

set -e
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "[ERROR] 请先设置 DEEPSEEK_API_KEY"
    exit 1
fi

OUT_DIR="results/llm_baseline"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
mkdir -p "$OUT_DIR"

run_one () {
    local DATASET=$1
    local VARIANT=$2
    local PKL=$3
    local LANG=$4
    local SAMPLE=$5
    local OUT="$OUT_DIR/${DATASET}_${VARIANT}.json"

    echo "=== ${DATASET} ${VARIANT} ==="
    python scripts/eval_llm_baseline.py \
        --pkl "$PKL" \
        --lang "$LANG" \
        --sample "$SAMPLE" \
        --seed 42 \
        --model "$DEEPSEEK_MODEL" \
        --output "$OUT"
}

run_one politifact clean "data/sheepdog/news_articles/politifact_test.pkl" en 0
for V in A B C D; do
    run_one politifact "adv_${V}" "data/sheepdog/adversarial_test/politifact_test_adv_${V}.pkl" en 0
done

run_one gossipcop clean "data/sheepdog/news_articles/gossipcop_test.pkl" en 200
for V in A B C D; do
    run_one gossipcop "adv_${V}" "data/sheepdog/adversarial_test/gossipcop_test_adv_${V}.pkl" en 200
done

run_one weibo21 clean "data/weibo21_repo/data/test.pkl" zh 200

python scripts/aggregate_llm_baseline.py \
    --input_dir "$OUT_DIR" \
    --output "$OUT_DIR/summary.csv"
