#!/bin/bash
# BERT 基线的三种子训练、对抗评测与结果汇总。
# 请按实际部署位置修改下面的项目路径。

set -e
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p results/models results/bert_adv results/baseline_logs

echo "================================================================"
echo "阶段 1: 训练 BERT baseline 6 次 (PolitiFact + GossipCop × 3 seed)"
echo "================================================================"

for DATASET in politifact gossipcop; do
  for SEED in 42 2024 3407; do
    TAG="${DATASET}_seed${SEED}"
    CKPT="./results/models/socialdebias_${DATASET}_en_seed${SEED}_bert_baseline.pt"
    
    if [ -f "$CKPT" ]; then
        echo "[SKIP TRAIN] 已存在: $CKPT"
        continue
    fi
    
    echo ""
    echo "---------------- 训练 $TAG ----------------"
    python scripts/train_bert_baseline.py \
        --dataset $DATASET --seed $SEED 2>&1 | tee results/baseline_logs/${TAG}_train.log
  done
done

echo ""
echo "================================================================"
echo "阶段 2: 评估对抗鲁棒性 6 次"
echo "================================================================"

for DATASET in politifact gossipcop; do
  for SEED in 42 2024 3407; do
    TAG="${DATASET}_seed${SEED}"
    JSON="results/bert_adv/bert_adv_${DATASET}_seed${SEED}.json"
    
    if [ -f "$JSON" ]; then
        echo "[SKIP EVAL] 已存在: $JSON"
        continue
    fi
    
    echo ""
    echo "---------------- 评估 $TAG ----------------"
    python scripts/evaluate_bert_adv.py \
        --dataset $DATASET --seed $SEED 2>&1 | tee results/baseline_logs/${TAG}_eval.log
  done
done

echo ""
echo "================================================================"
echo "阶段 3: 3 seed 汇总 (mean ± std)"
echo "================================================================"

python3 << 'PYEOF'
import json, statistics, os

OUT_DIR = "results/bert_adv"
print(f"{'数据集':<14}{'clean F1':<18}{'adv F1':<18}{'F1 Drop':<16}{'avg ASR':<16}")
print("-" * 82)

for DS in ["politifact", "gossipcop"]:
    cleans, advs, drops, asrs = [], [], [], []
    for SEED in [42, 2024, 3407]:
        p = f"{OUT_DIR}/bert_adv_{DS}_seed{SEED}.json"
        if not os.path.exists(p):
            print(f"  [MISS] {p}")
            continue
        s = json.load(open(p))["results"]["summary"]
        cleans.append(s["clean_f1"])
        advs.append(s["avg_adv_f1"])
        drops.append(s["f1_drop"])
        asrs.append(s["avg_asr"])
    if cleans:
        cm = statistics.mean(cleans); cs = statistics.stdev(cleans) if len(cleans)>1 else 0
        am = statistics.mean(advs);  as_ = statistics.stdev(advs)   if len(advs)>1   else 0
        dm = statistics.mean(drops); ds = statistics.stdev(drops)   if len(drops)>1  else 0
        rm = statistics.mean(asrs);  rs = statistics.stdev(asrs)    if len(asrs)>1   else 0
        print(f"{DS:<14}{cm:.4f}±{cs:.4f}    {am:.4f}±{as_:.4f}    {dm*100:.2f}±{ds*100:.2f}pp    {rm*100:.2f}±{rs*100:.2f}%")
print()
print("→ 用这些数字替换论文表 5-3 / 5-4 的 BERT-base 行（4 列均含 std）")
PYEOF
