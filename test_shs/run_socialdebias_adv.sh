#!/bin/bash
# 生成 SocialDebias-Adv 测试集：三个数据集、四种语气、两个模型来源。

set -e
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

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

# PolitiFact：全部测试样本
for TONE in "${TONES[@]}"; do
    for SRC in "${SOURCES[@]}"; do
        run_one politifact "data/sheepdog/news_articles/politifact_test.pkl" en 0 "$TONE" "$SRC"
    done
done

# GossipCop：固定随机种子抽取 200 条
for TONE in "${TONES[@]}"; do
    for SRC in "${SOURCES[@]}"; do
        run_one gossipcop "data/sheepdog/news_articles/gossipcop_test.pkl" en 200 "$TONE" "$SRC"
    done
done

# Weibo21：固定随机种子抽取 200 条
for TONE in "${TONES[@]}"; do
    for SRC in "${SOURCES[@]}"; do
        run_one weibo21 "data/weibo21_repo/data/test.pkl" zh 200 "$TONE" "$SRC"
    done
done

echo "===== 全部完成 ====="
ls -la "$OUT_DIR"/
