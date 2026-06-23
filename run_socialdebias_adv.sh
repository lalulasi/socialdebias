#!/bin/bash
# SocialDebias-Adv 完整生成: 3 dataset × 4 tone × 2 source = 24 个 pkl
# 总调用 ~3920 次 (PolitiFact 90 + GossipCop 200 + Weibo21 200 = 490 条 × 4 tone × 2 source)

set -e
cd /root/autodl-tmp/socialdebias

if [ -z "$DASHSCOPE_API_KEY" ] || [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "[ERROR] 需要 export DASHSCOPE_API_KEY 和 DEEPSEEK_API_KEY"
    exit 1
fi

OUT_DIR="data/socialdebias_adv"
LOG_DIR="logs/socialdebias_adv"
mkdir -p "$OUT_DIR" "$LOG_DIR"

TONES=("neutral" "objective" "sensational" "emotionally_triggering")
SOURCES=("qwen" "deepseek")

run_one () {
    local DATASET=$1 PKL=$2 LANG=$3 SAMPLE=$4 TONE=$5 SRC=$6
    local OUT="$OUT_DIR/${DATASET}_test_adv_${TONE}_${SRC}.pkl"
    local LOG="$LOG_DIR/${DATASET}_${TONE}_${SRC}.log"
    echo "--- ${DATASET} ${TONE} ${SRC} $(date +%H:%M:%S) ---"
    python scripts/gen_socialdebias_adv.py \
        --pkl "$PKL" --lang "$LANG" --tone "$TONE" --source "$SRC" \
        --sample "$SAMPLE" --seed 42 \
        --output "$OUT" 2>&1 | tee "$LOG"
}

# PolitiFact 90 全量
for TONE in "${TONES[@]}"; do
    for SRC in "${SOURCES[@]}"; do
        run_one politifact "data/sheepdog/news_articles/politifact_test.pkl" en 0 "$TONE" "$SRC"
    done
done

# GossipCop 200 抽样
for TONE in "${TONES[@]}"; do
    for SRC in "${SOURCES[@]}"; do
        run_one gossipcop "data/sheepdog/news_articles/gossipcop_test.pkl" en 200 "$TONE" "$SRC"
    done
done

# Weibo21 200 抽样
for TONE in "${TONES[@]}"; do
    for SRC in "${SOURCES[@]}"; do
        run_one weibo21 "data/weibo21_repo/data/test.pkl" zh 200 "$TONE" "$SRC"
    done
done

echo "===== 全部完成 ====="
ls -la "$OUT_DIR"/
