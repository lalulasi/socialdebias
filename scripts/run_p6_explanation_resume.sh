#!/usr/bin/env bash
# Resume-safe P6 three-seed IG explanation evaluation.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
batch_root=/autodl-fs/data/socialdebias/results/paper_v2_20260719
model_dir=${batch_root}/models
output_dir=${batch_root}/explanation
log_dir=${batch_root}/logs

cd "${project_root}"
mkdir -p "${output_dir}" "${log_dir}"

is_complete_explanation() {
  python - "$1" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        result = json.load(f)
except (OSError, ValueError):
    raise SystemExit(1)

rows = result.get("rows")
summary = result.get("summary")
raise SystemExit(0 if isinstance(rows, list) and len(rows) == 90 and summary else 1)
PY
}

for seed in 42 2024 3407; do
  surface_checkpoint="${model_dir}/socialdebias_politifact_en_seed${seed}_surface_all.pt"
  bert_checkpoint="${model_dir}/socialdebias_politifact_en_seed${seed}_bert_baseline.pt"
  output="${output_dir}/politifact_surface_all_seed${seed}.json"
  [[ -s "${surface_checkpoint}" && -s "${bert_checkpoint}" ]] || {
    echo "[ERROR] Missing P6 checkpoint for seed${seed}" >&2
    exit 1
  }
  if [[ -s "${output}" ]] \
      && [[ "${output}" -nt "${surface_checkpoint}" ]] \
      && [[ "${output}" -nt "${bert_checkpoint}" ]] \
      && is_complete_explanation "${output}"; then
    echo "[SKIP] explanation/seed${seed}"
    continue
  fi
  if [[ -e "${output}" ]]; then
    echo "[RESTART] incomplete/stale output will be recomputed: ${output}"
  fi
  python -u scripts/run_explanation_metrics.py \
    --dataset politifact --language en --variant C \
    --ckpt "${surface_checkpoint}" --bert_ckpt "${bert_checkpoint}" \
    --topk 10 --n_steps 50 --ig_internal_batch_size 4 \
    --max_length 512 --surface_feat_dim 8 \
    --output "${output}" 2>&1 | tee "${log_dir}/explanation_seed${seed}.log"
  [[ -s "${output}" ]] || { echo "[ERROR] Missing output: ${output}" >&2; exit 1; }
done

python -u scripts/aggregate_explanation_metrics.py \
  --input_dir "${output_dir}" --pattern 'politifact_surface_all_seed*.json' \
  --expected_n 3 --output "${output_dir}/explanation_3seed_summary.csv"
echo "P6_EXPLANATION_READY=True"
