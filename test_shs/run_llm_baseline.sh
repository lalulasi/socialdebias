#!/bin/bash
# 意见 19 兑现：DeepSeek LLM 零样本基线 完整评估
# 11 个测试集 × 串行调用，估时 ~80 分钟，估费 ~¥2-3

set -e
cd /root/autodl-tmp/socialdebias

# === 检查环境 ===
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "[ERROR] 请先 export DEEPSEEK_API_KEY=sk-xxx"
    exit 1
fi

OUT_DIR="results/llm_baseline"
LOG_DIR="logs/llm_baseline"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# === PolitiFact 5 集（全量 90 条/集）===
echo "===== PolitiFact (全量 90 条/集) ====="
for VARIANT in clean adv_A adv_B adv_C adv_D; do
    if [ "$VARIANT" = "clean" ]; then
        PKL="data/sheepdog/news_articles/politifact_test.pkl"
    else
        PKL="data/sheepdog/adversarial_test/politifact_test_${VARIANT}.pkl"
    fi
    OUT="$OUT_DIR/politifact_${VARIANT}.json"
    LOG="$LOG_DIR/politifact_${VARIANT}.log"
    echo "--- politifact $VARIANT $(date +%H:%M:%S) ---"
    python scripts/eval_llm_baseline.py \
        --pkl "$PKL" --lang en --sample 0 --seed 42 \
        --output "$OUT" 2>&1 | tee "$LOG"
done

# === GossipCop 5 集（采样 200 条/集）===
echo ""
echo "===== GossipCop (采样 200 条/集, seed=42) ====="
for VARIANT in clean adv_A adv_B adv_C adv_D; do
    if [ "$VARIANT" = "clean" ]; then
        PKL="data/sheepdog/news_articles/gossipcop_test.pkl"
    else
        PKL="data/sheepdog/adversarial_test/gossipcop_test_${VARIANT}.pkl"
    fi
    OUT="$OUT_DIR/gossipcop_${VARIANT}.json"
    LOG="$LOG_DIR/gossipcop_${VARIANT}.log"
    echo "--- gossipcop $VARIANT $(date +%H:%M:%S) ---"
    python scripts/eval_llm_baseline.py \
        --pkl "$PKL" --lang en --sample 200 --seed 42 \
        --output "$OUT" 2>&1 | tee "$LOG"
done

# === Weibo21 1 集（采样 200 条）===
echo ""
echo "===== Weibo21 (采样 200 条, seed=42) ====="
PKL="data/weibo21_repo/data/test.pkl"
OUT="$OUT_DIR/weibo21_clean.json"
LOG="$LOG_DIR/weibo21_clean.log"
echo "--- weibo21 clean $(date +%H:%M:%S) ---"
python scripts/eval_llm_baseline.py \
    --pkl "$PKL" --lang zh --sample 200 --seed 42 \
    --output "$OUT" 2>&1 | tee "$LOG"

# === 汇总 CSV ===
echo ""
echo "===== 汇总 CSV ====="
python scripts/aggregate_llm_baseline.py \
    --input_dir "$OUT_DIR" \
    --output "$OUT_DIR/summary.csv"

echo ""
echo "===== 完成 ====="
echo "结果目录: $OUT_DIR"
echo "汇总 CSV: $OUT_DIR/summary.csv"
