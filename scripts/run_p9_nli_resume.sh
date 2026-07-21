#!/usr/bin/env bash
# Resume-safe P9 five NLI configurations x three datasets x three seeds.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
batch_root=/autodl-fs/data/socialdebias/results/paper_v2_20260719
model_dir=${batch_root}/models
log_dir=${batch_root}/logs
english_nrc=${project_root}/data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
chinese_nrc=${project_root}/data/lexicons/NRC-Emotion-Lexicon-ZH.tsv
configs=(nli_hard nli_contrast_only nli_cls_only nli_dual_floor0 nli_soft14)

cd "${project_root}"
mkdir -p "${model_dir}" "${log_dir}"
test -s "${english_nrc}" && test -s "${chinese_nrc}"

for dataset in politifact gossipcop weibo21; do
  if [[ "${dataset}" == politifact ]]; then
    batch_size=4; language=en; lexicon=${english_nrc}
    data_args=(--dataset politifact --language en)
    original=data/sheepdog/news_articles/politifact_train.pkl
  elif [[ "${dataset}" == gossipcop ]]; then
    batch_size=16; language=en; lexicon=${english_nrc}
    data_args=(--dataset gossipcop --language en)
    original=data/sheepdog/news_articles/gossipcop_train.pkl
  else
    batch_size=16; language=zh; lexicon=${chinese_nrc}
    data_args=(--use_weibo21)
    original=data/weibo21_repo/data/train.pkl
  fi
  adversarial="data/qwen_adv/${dataset}_p_entail_paper_v2.pkl"

  for config in "${configs[@]}"; do
    case "${config}" in
      nli_hard)
        nli_args=(--use_contrastive --lambda_contrast 0.3 --temperature 0.07) ;;
      nli_contrast_only)
        nli_args=(--use_contrastive --lambda_contrast 0.3 --temperature 0.07 --use_soft_labels --alpha_floor 0.5) ;;
      nli_cls_only)
        nli_args=(--lambda_fact_soft 0.5 --soft_label_floor 0.5) ;;
      nli_dual_floor0)
        nli_args=(--use_contrastive --lambda_contrast 0.3 --temperature 0.07 --use_soft_labels --alpha_floor 0 --lambda_fact_soft 0.5 --soft_label_floor 0) ;;
      nli_soft14)
        nli_args=(--use_contrastive --lambda_contrast 0.3 --temperature 0.07 --use_soft_labels --alpha_floor 0.5 --lambda_fact_soft 0.5 --soft_label_floor 0.5) ;;
    esac
    for seed in 42 2024 3407; do
      checkpoint="${model_dir}/socialdebias_${dataset}_${language}_seed${seed}_${config}.pt"
      history="${model_dir}/socialdebias_${dataset}_${language}_seed${seed}_${config}_history.json"
      if [[ -s "${checkpoint}" && -s "${history}" ]]; then
        echo "[SKIP] ${dataset}/${config}/seed${seed}"
        continue
      fi
      python -u scripts/train_socialdebias_surface.py \
        "${data_args[@]}" --seed "${seed}" \
        --epochs 3 --batch_size "${batch_size}" --max_length 512 \
        --lr 2e-5 --weight_decay 0.01 --label_smoothing 0.1 \
        --hidden_dim 384 --bottleneck_dim 128 \
        --lambda_fact 1.0 --lambda_bias 0.5 --lambda_consist 0.3 \
        --surface_feat_dim 8 --surface_lexicon_path "${lexicon}" \
        "${nli_args[@]}" --orig_pkl "${original}" --adv_pkl "${adversarial}" \
        --save_dir "${model_dir}" --save_suffix "${config}" \
        2>&1 | tee "${log_dir}/${config}_${dataset}_seed${seed}.log"
      [[ -s "${checkpoint}" && -s "${history}" ]] || {
        echo "[ERROR] P9 incomplete output: ${dataset}/${config}/seed${seed}" >&2
        exit 1
      }
    done
  done
done

count="$(find "${model_dir}" -maxdepth 1 -type f -name '*_nli_*_history.json' | wc -l)"
[[ "${count}" -eq 45 ]] || { echo "[ERROR] P9 incomplete: ${count}/45" >&2; exit 1; }
python -u scripts/aggregate_training_histories.py \
  --model_dir "${model_dir}" \
  --suffixes nli_hard nli_contrast_only nli_cls_only nli_dual_floor0 nli_soft14 \
  --expected_datasets politifact gossipcop weibo21 \
  --expected_seeds 42 2024 3407 --require_complete \
  --output "${batch_root}/nli_mechanism_summary.csv"
echo "P9_NLI_READY=True"
