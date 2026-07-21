#!/usr/bin/env bash
# Resume-safe P7 ablation training and evaluation for the fixed paper-v2 batch.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
model_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/models
evaluation_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/ablation_adv
log_dir=/autodl-fs/data/socialdebias/results/paper_v2_20260719/logs
english_nrc=/autodl-fs/data/socialdebias/data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

cd "${project_root}"
mkdir -p "${model_dir}" "${evaluation_dir}" "${log_dir}"

command -v python >/dev/null 2>&1 || {
  echo "[ERROR] python command not found" >&2
  exit 1
}
python -c 'import numpy, sklearn, torch, transformers' || {
  echo "[ERROR] Current Python is missing SocialDebias dependencies" >&2
  exit 1
}
test -s "${english_nrc}" || {
  echo "[ERROR] Missing NRC lexicon: ${english_nrc}" >&2
  exit 1
}

configs=(full no_grl no_infonce no_consist no_surface no_advaug no_labelsmooth)

for dataset in politifact gossipcop; do
  if [[ "${dataset}" == politifact ]]; then
    batch_size=4
  else
    batch_size=16
  fi
  original="data/sheepdog/news_articles/${dataset}_train.pkl"
  adversarial="data/qwen_adv/${dataset}_p_entail_paper_v2.pkl"

  for config in "${configs[@]}"; do
    for seed in 42 2024 3407; do
      lambda_bias=0.5
      lambda_consist=0.3
      surface_dim=8
      label_smoothing=0.1
      lexicon_args=(--surface_lexicon_path "${english_nrc}")
      paired_args=(--orig_pkl "${original}" --adv_pkl "${adversarial}")
      nli_args=(
        --use_contrastive --lambda_contrast 0.3 --temperature 0.07
        --use_soft_labels --alpha_floor 0.5
        --lambda_fact_soft 0.5 --soft_label_floor 0.5
      )

      case "${config}" in
        full) ;;
        no_grl) lambda_bias=0 ;;
        no_infonce)
          nli_args=(--lambda_fact_soft 0.5 --soft_label_floor 0.5)
          ;;
        no_consist) lambda_consist=0 ;;
        no_surface)
          surface_dim=0
          lexicon_args=()
          ;;
        no_advaug)
          paired_args=()
          nli_args=()
          ;;
        no_labelsmooth)
          label_smoothing=0
          nli_args=(
            --use_contrastive --lambda_contrast 0.3 --temperature 0.07
            --use_soft_labels --alpha_floor 0.5
          )
          ;;
      esac

      suffix="abl_${config}"
      checkpoint="${model_dir}/socialdebias_${dataset}_en_seed${seed}_${suffix}.pt"
      history="${model_dir}/socialdebias_${dataset}_en_seed${seed}_${suffix}_history.json"
      evaluation="${evaluation_dir}/ablation_adv_${dataset}_${suffix}_seed${seed}.json"

      training_is_current=false
      if [[ -s "${checkpoint}" && -s "${history}" ]]; then
        if [[ "${config}" != no_infonce ]]; then
          training_is_current=true
        elif python -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if p.get("args", {}).get("paired_forward_version") == "skip_unused_orig_v1" else 1)' "${history}"; then
          training_is_current=true
        else
          echo "[STALE TRAIN] ${dataset}/${suffix}/seed${seed}: paired forward implementation changed"
        fi
      fi

      if [[ "${training_is_current}" == true ]]; then
        echo "[SKIP TRAIN] ${dataset}/${suffix}/seed${seed}"
      else
        echo "[TRAIN] ${dataset}/${suffix}/seed${seed}"
        python -u scripts/train_ablation.py \
          --dataset "${dataset}" --language en --seed "${seed}" \
          --epochs 3 --batch_size "${batch_size}" --max_length 512 \
          --lr 2e-5 --weight_decay 0.01 --label_smoothing "${label_smoothing}" \
          --hidden_dim 384 --bottleneck_dim 128 \
          --lambda_fact 1.0 --lambda_bias "${lambda_bias}" \
          --lambda_consist "${lambda_consist}" \
          --surface_feat_dim "${surface_dim}" "${lexicon_args[@]}" \
          "${nli_args[@]}" "${paired_args[@]}" \
          --save_dir "${model_dir}" --save_suffix "${suffix}" \
          2>&1 | tee "${log_dir}/${suffix}_${dataset}_seed${seed}.log"
        [[ -s "${checkpoint}" && -s "${history}" ]] || {
          echo "[ERROR] Training produced incomplete outputs: ${dataset}/${suffix}/seed${seed}" >&2
          exit 1
        }
      fi

      if [[ -s "${evaluation}" && "${evaluation}" -nt "${checkpoint}" && "${evaluation}" -nt "${history}" ]]; then
        echo "[SKIP EVAL] ${dataset}/${suffix}/seed${seed}"
      else
        echo "[EVAL] ${dataset}/${suffix}/seed${seed}"
        python -u scripts/evaluate_ablation_adv.py \
          --dataset "${dataset}" --language en --seed "${seed}" \
          --save_suffix "${suffix}" --variants A,B,C,D \
          --batch_size "${batch_size}" --max_length 512 \
          "${lexicon_args[@]}" \
          --ckpt "${checkpoint}" --output_dir "${evaluation_dir}" \
          2>&1 | tee "${log_dir}/${suffix}_adv_${dataset}_seed${seed}.log"
        [[ -s "${evaluation}" ]] || {
          echo "[ERROR] Evaluation produced no JSON: ${evaluation}" >&2
          exit 1
        }
      fi
    done
  done
done

history_count="$(find "${model_dir}" -maxdepth 1 -type f -name '*_abl_*_history.json' | wc -l)"
evaluation_count="$(find "${evaluation_dir}" -maxdepth 1 -type f -name 'ablation_adv_*_abl_*.json' | wc -l)"
if [[ "${history_count}" -ne 42 || "${evaluation_count}" -ne 42 ]]; then
  echo "[ERROR] P7 incomplete: history=${history_count}/42 eval=${evaluation_count}/42" >&2
  exit 1
fi

python -u scripts/aggregate_training_histories.py \
  --model_dir "${model_dir}" \
  --suffixes abl_full abl_no_grl abl_no_infonce abl_no_consist \
             abl_no_surface abl_no_advaug abl_no_labelsmooth \
  --expected_datasets politifact gossipcop \
  --expected_seeds 42 2024 3407 --require_complete \
  --output /autodl-fs/data/socialdebias/results/paper_v2_20260719/ablation_clean_summary.csv

python -u scripts/parse_ablation_adv.py "${evaluation_dir}"
echo "P7_ABLATION_READY=True"
