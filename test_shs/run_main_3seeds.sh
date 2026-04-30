#!/bin/bash
# 主实验 3 种子批量
# BERT 基线 × 2 数据集 × 3 种子 + SocialDebias × 2 数据集 × 3 种子 + 6 次对抗评估
# λ 配置与 exp001/exp002 完全一致，保证 3 种子结果可和单次 exp 对比
#
# 使用: cd /root/autodl-tmp/socialdebias && nohup bash run_main_3seeds.sh > run_main_3seeds.out 2>&1 &

cd /root/autodl-tmp/socialdebias

SEEDS=(42 2024 3407)
DATASETS=("politifact" "gossipcop")
LANGUAGE="en"

# === λ 与 exp001/exp002 对齐 ===
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

# === 阶段 1：BERT 基线 × 6 ===
echo ""
echo "=== 阶段 1/3：BERT 基线 (6 次) ==="
for DS in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TAG="baseline_${DS}_seed${SEED}"
    CMD="python scripts/train_baseline.py --dataset ${DS} --language ${LANGUAGE} --seed ${SEED}"
    run_task "phase1" "${CMD}" "${TAG}" "${LOG_ROOT}/${TAG}.log" "${DS}" "baseline" "${SEED}"
  done
done

# === 阶段 2：SocialDebias × 6 ===
echo ""
echo "=== 阶段 2/3：SocialDebias (6 次) ==="
for DS in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TAG="socialdebias_${DS}_seed${SEED}"
    CMD="python scripts/train_socialdebias.py --dataset ${DS} --language ${LANGUAGE} --seed ${SEED} --lambda_fact ${LAMBDA_FACT} --lambda_bias ${LAMBDA_BIAS} --lambda_consist ${LAMBDA_CONSIST}"
    run_task "phase2" "${CMD}" "${TAG}" "${LOG_ROOT}/${TAG}.log" "${DS}" "socialdebias" "${SEED}"
  done
done

# === 阶段 3：对抗评估 × 6（每次评估 baseline + SD 两个 ckpt）===
echo ""
echo "=== 阶段 3/3：对抗评估 (6 次) ==="
for DS in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TAG="adv_${DS}_seed${SEED}"
    CMD="python scripts/evaluate_adversarial.py --dataset ${DS} --language ${LANGUAGE} --seed ${SEED}"
    run_task "phase3" "${CMD}" "${TAG}" "${LOG_ROOT}/${TAG}.log" "${DS}" "adv_eval" "${SEED}"
  done
done

echo ""
echo "========================================"
echo "全部完成: $(date)"
echo "状态汇总: ${RESULT_CSV}"
echo "========================================"
cat "${RESULT_CSV}" | column -t -s,