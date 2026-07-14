#!/bin/bash
# 两个英文数据集的三种子主实验。
# 依次训练 BERT 基线和不使用表层特征的 SocialDebias，并分别完成对抗评测。

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SEEDS=(42 2024 3407)
DATASETS=("politifact" "gossipcop")
LANGUAGE="en"

# SocialDebias 损失权重
LAMBDA_FACT=1.0
LAMBDA_BIAS=0.5
LAMBDA_CONSIST=0.3

LOG_ROOT="./results/main_3seeds_logs"
mkdir -p "${LOG_ROOT}"

RESULT_CSV="${LOG_ROOT}/batch_status.csv"
echo "phase,dataset,model,seed,status,start,end,elapsed_sec" > "${RESULT_CSV}"

run_task() {
    local PHASE=$1 CMD=$2 TAG=$3 LOG_FILE=$4 DS=$5 MODEL=$6 SEED=$7
    local START=$(date '+%Y-%m-%d %H:%M:%S')
    local START_TS=$(date +%s)
    echo ""
    echo "[${PHASE}] ${TAG}   开始: ${START}"
    eval "${CMD}" > "${LOG_FILE}" 2>&1
    local STATUS=$?
    local END=$(date '+%Y-%m-%d %H:%M:%S')
    local ELAPSED=$(($(date +%s) - START_TS))
    if [ ${STATUS} -eq 0 ]; then
        echo "  [✓] 成功  用时 ${ELAPSED}s"
        echo "${PHASE},${DS},${MODEL},${SEED},success,${START},${END},${ELAPSED}" >> "${RESULT_CSV}"
    else
        echo "  [✗] 失败  日志: ${LOG_FILE}"
        echo "${PHASE},${DS},${MODEL},${SEED},failed,${START},${END},${ELAPSED}" >> "${RESULT_CSV}"
        tail -10 "${LOG_FILE}" | sed 's/^/    /'
    fi
}

echo "========================================"
echo "主实验 3 种子批量开始: $(date)"
echo "λ_fact=${LAMBDA_FACT}  λ_bias=${LAMBDA_BIAS}  λ_consist=${LAMBDA_CONSIST}"
echo "Seeds: ${SEEDS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "========================================"

# 第一阶段：BERT 基线
echo ""
echo "=== 阶段 1/3：BERT 基线 (6 次) ==="
for DS in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TAG="baseline_${DS}_seed${SEED}"
    CMD="python scripts/train_bert_baseline.py --dataset ${DS} --seed ${SEED}"
    run_task "phase1" "${CMD}" "${TAG}" "${LOG_ROOT}/${TAG}.log" "${DS}" "baseline" "${SEED}"
  done
done

# 第二阶段：不使用表层特征的 SocialDebias
echo ""
echo "=== 阶段 2/3：SocialDebias (6 次) ==="
for DS in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TAG="socialdebias_${DS}_seed${SEED}"
    CMD="python scripts/train_socialdebias_surface.py --dataset ${DS} --language ${LANGUAGE} --seed ${SEED} --epochs 3 --batch_size 4 --lambda_fact ${LAMBDA_FACT} --lambda_bias ${LAMBDA_BIAS} --lambda_consist ${LAMBDA_CONSIST} --surface_feat_dim 0 --save_suffix main"
    run_task "phase2" "${CMD}" "${TAG}" "${LOG_ROOT}/${TAG}.log" "${DS}" "socialdebias" "${SEED}"
  done
done

# 第三阶段：分别评测 BERT 基线和 SocialDebias
echo ""
echo "=== 阶段 3/3：对抗评估 (6 次) ==="
for DS in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TAG="bert_adv_${DS}_seed${SEED}"
    CMD="python scripts/evaluate_bert_adv.py --dataset ${DS} --seed ${SEED}"
    run_task "phase3" "${CMD}" "${TAG}" "${LOG_ROOT}/${TAG}.log" "${DS}" "bert_adv" "${SEED}"
    TAG="socialdebias_adv_${DS}_seed${SEED}"
    CMD="python scripts/evaluate_surface_adv.py --dataset ${DS} --seed ${SEED} --save_suffix main"
    run_task "phase3" "${CMD}" "${TAG}" "${LOG_ROOT}/${TAG}.log" "${DS}" "socialdebias_adv" "${SEED}"
  done
done

echo ""
echo "========================================"
echo "全部完成: $(date)"
echo "状态汇总: ${RESULT_CSV}"
echo "========================================"
cat "${RESULT_CSV}" | column -t -s,
