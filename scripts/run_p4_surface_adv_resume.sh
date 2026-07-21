#!/usr/bin/env bash
# Resume-safe P4 SheepDog A/B/C/D evaluation for surface_all.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
batch_root=/autodl-fs/data/socialdebias/results/paper_v2_20260719
model_dir=${batch_root}/models
evaluation_dir=${batch_root}/surface_adv
bert_dir=${batch_root}/bert_adv
log_dir=${batch_root}/logs
english_nrc=${project_root}/data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

cd "${project_root}"
mkdir -p "${evaluation_dir}" "${log_dir}"
test -s "${english_nrc}"

for dataset in politifact gossipcop; do
  if [[ "${dataset}" == politifact ]]; then batch_size=4; else batch_size=16; fi
  for seed in 42 2024 3407; do
    checkpoint="${model_dir}/socialdebias_${dataset}_en_seed${seed}_surface_all.pt"
    history="${model_dir}/socialdebias_${dataset}_en_seed${seed}_surface_all_history.json"
    output="${evaluation_dir}/surface_adv_${dataset}_seed${seed}_surface_all.json"
    [[ -s "${checkpoint}" && -s "${history}" ]] || {
      echo "[ERROR] Missing P3 output: ${dataset}/seed${seed}" >&2
      exit 1
    }
    if [[ -s "${output}" && "${output}" -nt "${checkpoint}" && "${output}" -nt "${history}" ]]; then
      echo "[SKIP] ${dataset}/seed${seed}"
      continue
    fi
    python -u scripts/evaluate_surface_adv.py \
      --dataset "${dataset}" --language en --seed "${seed}" \
      --save_suffix surface_all --variants A,B,C,D \
      --batch_size "${batch_size}" --max_length 512 \
      --surface_lexicon_path "${english_nrc}" \
      --ckpt_dir "${model_dir}" --output_dir "${evaluation_dir}" \
      2>&1 | tee "${log_dir}/surface_adv_${dataset}_seed${seed}.log"
    [[ -s "${output}" ]] || { echo "[ERROR] Missing output: ${output}" >&2; exit 1; }
  done
done

count="$(find "${evaluation_dir}" -maxdepth 1 -type f -name 'surface_adv_*_surface_all.json' | wc -l)"
[[ "${count}" -eq 6 ]] || { echo "[ERROR] P4 incomplete: ${count}/6" >&2; exit 1; }
python -u scripts/parse_main_3seeds.py \
  --datasets politifact gossipcop --seeds 42 2024 3407 \
  --surface_suffix surface_all --bert_dir "${bert_dir}" \
  --surface_dir "${evaluation_dir}" --output "${batch_root}/main_summary.csv"
echo "P4_SURFACE_ADV_READY=True"
