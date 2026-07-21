#!/usr/bin/env bash
# P1 data filtering and NLI preparation with fixed paper-v2 paths.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
report_dir=${project_root}/results/paper_v2_20260719/data_preparation
outputs=(
  data/qwen_adv/politifact_train_adv_filtered_paper_v2.pkl
  data/qwen_adv/gossipcop_train_adv_filtered_paper_v2.pkl
  data/qwen_adv/weibo21_train_adv_filtered_paper_v2.pkl
  data/qwen_adv/politifact_p_entail_paper_v2.pkl
  data/qwen_adv/gossipcop_p_entail_paper_v2.pkl
  data/qwen_adv/weibo21_p_entail_paper_v2.pkl
)

cd "${project_root}"
mkdir -p "${report_dir}"
all_present=true
for output in "${outputs[@]}"; do
  [[ -s "${output}" ]] || all_present=false
done
if [[ "${all_present}" == true ]]; then
  echo "[SKIP] Six P1 outputs already exist; running strict audit only."
else
  RUN_TAG=paper_v2_20260719 bash prepare_paper_v2_data.sh
fi
python -u scripts/audit_experiment_data.py \
  --strict_paper_v2 --strict_training --json_output "${report_dir}/after.json"
echo "P1_DATA_READY=True"
