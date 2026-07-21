#!/usr/bin/env bash
# Resume-safe P10 adaptive-lambda training and English adversarial evaluation.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
batch_root=/autodl-fs/data/socialdebias/results/paper_v2_20260719
model_dir=${batch_root}/models
evaluation_dir=${batch_root}/surface_adv
bert_dir=${batch_root}/bert_adv
log_dir=${batch_root}/logs
english_nrc=${project_root}/data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
chinese_nrc=${project_root}/data/lexicons/NRC-Emotion-Lexicon-ZH.tsv
configs=(surface_fixed surface_adaptive)

cd "${project_root}"
mkdir -p "${model_dir}" "${evaluation_dir}" "${log_dir}"
test -s "${english_nrc}" && test -s "${chinese_nrc}"

for dataset in politifact gossipcop weibo21; do
  if [[ "${dataset}" == politifact ]]; then
    batch_size=4; language=en; lexicon=${english_nrc}; data_args=(--dataset politifact --language en)
  elif [[ "${dataset}" == gossipcop ]]; then
    batch_size=16; language=en; lexicon=${english_nrc}; data_args=(--dataset gossipcop --language en)
  else
    batch_size=16; language=zh; lexicon=${chinese_nrc}; data_args=(--use_weibo21)
  fi
  for config in "${configs[@]}"; do
    adaptive_args=()
    [[ "${config}" == surface_adaptive ]] && adaptive_args=(--adaptive_lambda --adaptive_size_thresh 1000 --adaptive_f1_thresh 0.85 --adaptive_scale 0.1)
    for seed in 42 2024 3407; do
      checkpoint="${model_dir}/socialdebias_${dataset}_${language}_seed${seed}_${config}.pt"
      history="${model_dir}/socialdebias_${dataset}_${language}_seed${seed}_${config}_history.json"
      if [[ -s "${checkpoint}" && -s "${history}" ]]; then
        echo "[SKIP TRAIN] ${dataset}/${config}/seed${seed}"
        continue
      fi
      python -u scripts/train_socialdebias_surface.py \
        "${data_args[@]}" --seed "${seed}" \
        --epochs 3 --batch_size "${batch_size}" --max_length 512 \
        --lr 2e-5 --weight_decay 0.01 --label_smoothing 0.1 \
        --hidden_dim 384 --bottleneck_dim 128 \
        --lambda_fact 1.0 --lambda_bias 0.5 --lambda_consist 0.3 \
        --surface_feat_dim 8 --surface_lexicon_path "${lexicon}" \
        "${adaptive_args[@]}" --save_dir "${model_dir}" --save_suffix "${config}" \
        2>&1 | tee "${log_dir}/${config}_${dataset}_seed${seed}.log"
      [[ -s "${checkpoint}" && -s "${history}" ]] || {
        echo "[ERROR] P10 training output missing: ${dataset}/${config}/seed${seed}" >&2
        exit 1
      }
    done
  done
done

count="$(find "${model_dir}" -maxdepth 1 -type f \( -name '*_surface_fixed_history.json' -o -name '*_surface_adaptive_history.json' \) | wc -l)"
[[ "${count}" -eq 18 ]] || { echo "[ERROR] P10 training incomplete: ${count}/18" >&2; exit 1; }
python -u scripts/aggregate_training_histories.py \
  --model_dir "${model_dir}" --suffixes surface_fixed surface_adaptive \
  --expected_datasets politifact gossipcop weibo21 \
  --expected_seeds 42 2024 3407 --require_complete \
  --output "${batch_root}/adaptive_clean_summary.csv"

for config in "${configs[@]}"; do
  for dataset in politifact gossipcop; do
    if [[ "${dataset}" == politifact ]]; then batch_size=4; else batch_size=16; fi
    for seed in 42 2024 3407; do
      checkpoint="${model_dir}/socialdebias_${dataset}_en_seed${seed}_${config}.pt"
      history="${model_dir}/socialdebias_${dataset}_en_seed${seed}_${config}_history.json"
      output="${evaluation_dir}/surface_adv_${dataset}_seed${seed}_${config}.json"
      if [[ -s "${output}" && "${output}" -nt "${checkpoint}" && "${output}" -nt "${history}" ]]; then
        echo "[SKIP EVAL] ${dataset}/${config}/seed${seed}"
        continue
      fi
      python -u scripts/evaluate_surface_adv.py \
        --dataset "${dataset}" --language en --seed "${seed}" \
        --save_suffix "${config}" --variants A,B,C,D \
        --batch_size "${batch_size}" --max_length 512 \
        --surface_lexicon_path "${english_nrc}" \
        --ckpt_dir "${model_dir}" --output_dir "${evaluation_dir}" \
        2>&1 | tee "${log_dir}/${config}_adv_${dataset}_seed${seed}.log"
      [[ -s "${output}" ]] || { echo "[ERROR] Missing eval: ${output}" >&2; exit 1; }
    done
  done
  python -u scripts/parse_main_3seeds.py \
    --datasets politifact gossipcop --seeds 42 2024 3407 \
    --surface_suffix "${config}" --bert_dir "${bert_dir}" \
    --surface_dir "${evaluation_dir}" --output "${batch_root}/${config}_adv_summary.csv"
done

count="$(find "${evaluation_dir}" -maxdepth 1 -type f \( -name '*_surface_fixed.json' -o -name '*_surface_adaptive.json' \) | wc -l)"
[[ "${count}" -eq 12 ]] || { echo "[ERROR] P10 eval incomplete: ${count}/12" >&2; exit 1; }
echo "P10_ADAPTIVE_READY=True"
