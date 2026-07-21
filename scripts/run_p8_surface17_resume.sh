#!/usr/bin/env bash
# Resume-safe P8 comparison of 8-D abl_full against 17-D surface features.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
model_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/models
evaluation_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/ablation_adv
log_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/logs
english_nrc=/autodl-fs/data/socialdebias/data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

cd "${project_root}"
mkdir -p "${model_dir}" "${evaluation_dir}" "${log_dir}"
python -c 'import numpy, sklearn, torch, transformers'
test -s "${english_nrc}" || {
  echo "[ERROR] Missing NRC lexicon: ${english_nrc}" >&2
  exit 1
}

for seed in 42 2024 3407; do
  checkpoint="${model_dir}/socialdebias_politifact_en_seed${seed}_surface17_full.pt"
  history="${model_dir}/socialdebias_politifact_en_seed${seed}_surface17_full_history.json"
  evaluation="${evaluation_dir}/ablation_adv_politifact_surface17_full_seed${seed}.json"

  if [[ -s "${checkpoint}" && -s "${history}" ]]; then
    echo "[SKIP TRAIN] politifact/surface17_full/seed${seed}"
  else
    echo "[TRAIN] politifact/surface17_full/seed${seed}"
    python -u scripts/train_ablation.py \
      --dataset politifact --language en --seed "${seed}" \
      --epochs 3 --batch_size 4 --max_length 512 \
      --lr 2e-5 --weight_decay 0.01 --label_smoothing 0.1 \
      --hidden_dim 384 --bottleneck_dim 128 \
      --lambda_fact 1.0 --lambda_bias 0.5 --lambda_consist 0.3 \
      --surface_feat_dim 17 --surface_lexicon_path "${english_nrc}" \
      --use_contrastive --lambda_contrast 0.3 --temperature 0.07 \
      --use_soft_labels --alpha_floor 0.5 \
      --lambda_fact_soft 0.5 --soft_label_floor 0.5 \
      --orig_pkl data/sheepdog/news_articles/politifact_train.pkl \
      --adv_pkl data/qwen_adv/politifact_p_entail_paper_v2.pkl \
      --save_dir "${model_dir}" --save_suffix surface17_full \
      2>&1 | tee "${log_dir}/surface17_full_politifact_seed${seed}.log"
    [[ -s "${checkpoint}" && -s "${history}" ]] || {
      echo "[ERROR] P8 training produced incomplete outputs for seed${seed}" >&2
      exit 1
    }
  fi

  if [[ -s "${evaluation}" && "${evaluation}" -nt "${checkpoint}" && "${evaluation}" -nt "${history}" ]]; then
    echo "[SKIP EVAL] politifact/surface17_full/seed${seed}"
  else
    echo "[EVAL] politifact/surface17_full/seed${seed}"
    python -u scripts/evaluate_ablation_adv.py \
      --dataset politifact --language en --seed "${seed}" \
      --save_suffix surface17_full --variants A,B,C,D \
      --batch_size 4 --max_length 512 \
      --surface_lexicon_path "${english_nrc}" \
      --ckpt "${checkpoint}" --output_dir "${evaluation_dir}" \
      2>&1 | tee "${log_dir}/surface17_full_adv_politifact_seed${seed}.log"
    [[ -s "${evaluation}" ]] || {
      echo "[ERROR] P8 evaluation produced no JSON for seed${seed}" >&2
      exit 1
    }
  fi
done

surface17_history_count="$(find "${model_dir}" -maxdepth 1 -type f -name '*_surface17_full_history.json' | wc -l)"
surface17_evaluation_count="$(find "${evaluation_dir}" -maxdepth 1 -type f -name '*_surface17_full_seed*.json' | wc -l)"
if [[ "${surface17_history_count}" -ne 3 || "${surface17_evaluation_count}" -ne 3 ]]; then
  echo "[ERROR] P8 incomplete: history=${surface17_history_count}/3 eval=${surface17_evaluation_count}/3" >&2
  exit 1
fi

python -u scripts/aggregate_training_histories.py \
  --model_dir "${model_dir}" \
  --suffixes abl_full surface17_full \
  --expected_datasets politifact \
  --expected_seeds 42 2024 3407 --require_complete \
  --output /autodl-fs/data/socialdebias/results/paper_v2_20260719/surface_8_vs_17_clean_summary.csv

python -u scripts/parse_ablation_adv.py "${evaluation_dir}"
echo "P8_SURFACE17_READY=True"
