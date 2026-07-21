#!/usr/bin/env bash
# Resume-safe P3 surface_all training for all datasets/seeds.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
model_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/models
log_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/logs
english_nrc=/autodl-fs/data/socialdebias/data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
chinese_nrc=/autodl-fs/data/socialdebias/data/lexicons/NRC-Emotion-Lexicon-ZH.tsv

cd "${project_root}"
mkdir -p "${model_dir}" "${log_dir}"
python -c 'import numpy, sklearn, torch, transformers'
test -s "${english_nrc}" && test -s "${chinese_nrc}"

for dataset in politifact gossipcop weibo21; do
  if [[ "${dataset}" == politifact ]]; then
    batch_size=4
    language=en
    lexicon="${english_nrc}"
    data_args=(--dataset politifact --language en)
    original=data/sheepdog/news_articles/politifact_train.pkl
  elif [[ "${dataset}" == gossipcop ]]; then
    batch_size=16
    language=en
    lexicon="${english_nrc}"
    data_args=(--dataset gossipcop --language en)
    original=data/sheepdog/news_articles/gossipcop_train.pkl
  else
    batch_size=16
    language=zh
    lexicon="${chinese_nrc}"
    data_args=(--use_weibo21)
    original=data/weibo21_repo/data/train.pkl
  fi
  adversarial="data/qwen_adv/${dataset}_p_entail_paper_v2.pkl"

  for seed in 42 2024 3407; do
    checkpoint="${model_dir}/socialdebias_${dataset}_${language}_seed${seed}_surface_all.pt"
    history="${model_dir}/socialdebias_${dataset}_${language}_seed${seed}_surface_all_history.json"
    if [[ -s "${checkpoint}" && -s "${history}" ]]; then
      echo "[SKIP] ${dataset}/surface_all/seed${seed}"
      continue
    fi
    python -u scripts/train_socialdebias_surface.py \
      "${data_args[@]}" --seed "${seed}" \
      --epochs 3 --batch_size "${batch_size}" --max_length 512 \
      --lr 2e-5 --weight_decay 0.01 --label_smoothing 0.1 \
      --hidden_dim 384 --bottleneck_dim 128 \
      --lambda_fact 1.0 --lambda_bias 0.5 --lambda_consist 0.3 \
      --surface_feat_dim 8 --surface_lexicon_path "${lexicon}" \
      --adaptive_lambda --adaptive_size_thresh 1000 \
      --adaptive_f1_thresh 0.85 --adaptive_scale 0.1 \
      --use_contrastive --lambda_contrast 0.3 --temperature 0.07 \
      --use_soft_labels --alpha_floor 0.5 \
      --lambda_fact_soft 0.5 --soft_label_floor 0.5 \
      --orig_pkl "${original}" --adv_pkl "${adversarial}" \
      --save_dir "${model_dir}" --save_suffix surface_all \
      2>&1 | tee "${log_dir}/surface_all_${dataset}_seed${seed}.log"
    [[ -s "${checkpoint}" && -s "${history}" ]] || {
      echo "[ERROR] P3 incomplete output: ${dataset}/seed${seed}" >&2
      exit 1
    }
  done
done

history_count="$(find "${model_dir}" -maxdepth 1 -type f -name '*_surface_all_history.json' | wc -l)"
checkpoint_count="$(find "${model_dir}" -maxdepth 1 -type f -name '*_surface_all.pt' | wc -l)"
[[ "${history_count}" -eq 9 && "${checkpoint_count}" -eq 9 ]] || {
  echo "[ERROR] P3 incomplete: checkpoint=${checkpoint_count}/9 history=${history_count}/9" >&2
  exit 1
}

python -u scripts/aggregate_training_histories.py \
  --model_dir "${model_dir}" --suffixes surface_all \
  --expected_datasets politifact gossipcop weibo21 \
  --expected_seeds 42 2024 3407 --require_complete \
  --output /autodl-fs/data/socialdebias/results/paper_v2_20260719/surface_all_clean_summary.csv
echo "P3_SURFACE_ALL_READY=True"
