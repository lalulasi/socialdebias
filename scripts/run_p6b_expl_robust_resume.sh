#!/usr/bin/env bash
# Resume-safe Table 5-10 / Figure 5.4 explanation-robustness correlation run.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
batch_root=/autodl-fs/data/socialdebias/results/paper_v2_20260719
model_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/models
output_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/explanation_robustness
log_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/logs

cd "${project_root}"
mkdir -p "${output_dir}" "${log_dir}"

is_complete_pairs() {
  python - "$1" <<'PY'
import csv
import sys
from collections import Counter

try:
    with open(sys.argv[1], encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
except (OSError, UnicodeError):
    raise SystemExit(1)

required = {
    "variant", "idx", "label", "attribution_target", "sd_spearman",
    "delta_p", "correct_orig", "correct_adv",
}
counts = Counter(row.get("variant") for row in rows)
valid = (
    len(rows) == 360
    and rows
    and required.issubset(rows[0])
    and counts == {"A": 90, "B": 90, "C": 90, "D": 90}
    and all(row["label"] == row["attribution_target"] for row in rows)
)
raise SystemExit(0 if valid else 1)
PY
}

for seed in 42 2024 3407; do
  checkpoint="${model_dir}/socialdebias_politifact_en_seed${seed}_surface_all.pt"
  output="${output_dir}/expl_robust_xy_politifact_seed${seed}.csv"
  test -s "${checkpoint}" || { echo "[ERROR] Missing checkpoint: ${checkpoint}" >&2; exit 1; }

  if [[ -s "${output}" && "${output}" -nt "${checkpoint}" ]] && is_complete_pairs "${output}"; then
    echo "[SKIP] explanation-robustness/seed${seed}"
    continue
  fi
  if [[ -e "${output}" ]]; then
    echo "[RESTART] incomplete/stale output will be recomputed: ${output}"
  fi

  python -u scripts/extract_expl_robust_xy.py \
    --orig_pkl data/sheepdog/news_articles/politifact_test.pkl \
    --adv_tpl 'data/sheepdog/adversarial_test/politifact_test_adv_{v}.pkl' \
    --variants A,B,C,D \
    --ckpt "${checkpoint}" \
    --output "${output}" \
    --top_k 10 --n_steps 50 --max_length 512 \
    --ig_internal_batch_size 2 \
    2>&1 | tee "${log_dir}/expl_robust_seed${seed}.log"

  is_complete_pairs "${output}" || {
    echo "[ERROR] Invalid pair output: ${output}" >&2
    exit 1
  }
done

python -u scripts/analyze_expl_robust_correlation.py \
  --inputs \
    "${output_dir}/expl_robust_xy_politifact_seed42.csv" \
    "${output_dir}/expl_robust_xy_politifact_seed2024.csv" \
    "${output_dir}/expl_robust_xy_politifact_seed3407.csv" \
  --output_dir "${output_dir}" \
  --primary_seed 42 --expected_rows 360 \
  --bootstrap_iters 5000 --random_seed 42 --figure_dpi 300 \
  2>&1 | tee "${log_dir}/expl_robust_analysis.log"

test -s "${output_dir}/explanation_robustness_summary.json"
test -s "${output_dir}/explanation_robustness_3seed_stats.csv"
test -s "${output_dir}/table5_10_seed42.csv"
test -s "${output_dir}/figure5_4_explanation_robustness_seed42.png"
test -s "${output_dir}/figure5_4_explanation_robustness_seed42.pdf"
echo "P6B_EXPLANATION_ROBUSTNESS_READY=True"
