#!/bin/bash
# ============================================================
# 17 维表层特征消融（对照 8 维 surface 基础配置）
# 配方与复现指南 B.1（PolitiFact 最优 surface, 4.85pp）完全一致，
# 唯一区别：--surface_feat_dim 8 → 17，--save_suffix surface → surface17
# PolitiFact × 3 seeds，训练 + 对抗评估 + 汇总。极小数据集，很快。
# ============================================================
set -e
cd /root/autodl-fs/socialdebias      # ← 按你的实际路径改

SEEDS=(42 2024 3407)
DS="politifact"
SUFFIX="surface17"
LOG_ROOT="./results/surface17_logs"
mkdir -p "${LOG_ROOT}"

echo "==== [1/2] 训练 17 维 × 3 seeds : $(date) ===="
for SEED in "${SEEDS[@]}"; do
  echo "  [train] ${DS} seed=${SEED} dim=17"
  python scripts/train_socialdebias_surface.py \
    --dataset ${DS} --language en --seed ${SEED} \
    --epochs 3 --batch_size 4 \
    --lambda_fact 1.0 --lambda_bias 0.5 --lambda_consist 0.3 \
    --surface_feat_dim 17 \
    --save_suffix ${SUFFIX} \
    > "${LOG_ROOT}/train_${DS}_seed${SEED}.log" 2>&1
  echo "    done ($(tail -1 ${LOG_ROOT}/train_${DS}_seed${SEED}.log 2>/dev/null | head -c 60))"
done

echo "==== [2/2] 对抗评估 17 维 × 3 seeds ===="
for SEED in "${SEEDS[@]}"; do
  echo "  [eval] seed=${SEED}"
  python scripts/evaluate_surface_adv.py \
    --dataset ${DS} --seed ${SEED} --save_suffix ${SUFFIX} \
    > "${LOG_ROOT}/eval_${DS}_seed${SEED}.log" 2>&1
  grep "F1 Drop" "${LOG_ROOT}/eval_${DS}_seed${SEED}.log" | tail -1 | sed 's/^/    /'
done

echo "==== 汇总 8 维 vs 17 维 ===="
python scripts/parse_surface17.py
echo "==== 完成: $(date) ===="
