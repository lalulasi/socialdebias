#!/usr/bin/env bash
# Resume-safe BiLSTM, BERT and BERT SheepDog baseline jobs.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
batch_root=/autodl-fs/data/socialdebias/results/paper_v2_20260719
model_dir=${batch_root}/models
lstm_dir=${batch_root}/lstm
bert_adv_dir=${batch_root}/bert_adv
log_dir=${batch_root}/logs

cd "${project_root}"
mkdir -p "${model_dir}" "${lstm_dir}" "${bert_adv_dir}" "${log_dir}"

for dataset in politifact gossipcop; do
  if [[ "${dataset}" == politifact ]]; then batch_size=4; else batch_size=16; fi
  for seed in 42 2024 3407; do
    result="${lstm_dir}/lstm_${dataset}_seed${seed}_result.json"
    checkpoint="${lstm_dir}/lstm_${dataset}_seed${seed}.pt"
    if [[ -s "${result}" && -s "${checkpoint}" ]]; then
      echo "[SKIP LSTM] ${dataset}/seed${seed}"
    else
      python -u scripts/train_lstm.py \
        --dataset "${dataset}" --seed "${seed}" --epochs 3 --max_len 512 \
        --batch_size "${batch_size}" --lr 1e-3 --embed_dim 300 \
        --hidden_dim 256 --num_layers 2 --output_dir "${lstm_dir}" \
        2>&1 | tee "${log_dir}/lstm_${dataset}_seed${seed}.log"
      [[ -s "${result}" && -s "${checkpoint}" ]] || { echo "[ERROR] LSTM output missing" >&2; exit 1; }
    fi
  done
done
python -u scripts/parse_lstm_results.py --input_dir "${lstm_dir}" --output "${batch_root}/lstm_summary.csv"

for dataset in politifact gossipcop weibo21; do
  if [[ "${dataset}" == politifact ]]; then
    batch_size=4; language=en; data_args=(--dataset politifact)
  elif [[ "${dataset}" == gossipcop ]]; then
    batch_size=16; language=en; data_args=(--dataset gossipcop)
  else
    batch_size=16; language=zh; data_args=(--use_weibo21)
  fi
  for seed in 42 2024 3407; do
    checkpoint="${model_dir}/socialdebias_${dataset}_${language}_seed${seed}_bert_baseline.pt"
    history="${model_dir}/socialdebias_${dataset}_${language}_seed${seed}_bert_baseline_history.json"
    if [[ -s "${checkpoint}" && -s "${history}" ]]; then
      echo "[SKIP BERT] ${dataset}/seed${seed}"
      continue
    fi
    python -u scripts/train_bert_baseline.py \
      "${data_args[@]}" --seed "${seed}" --batch_size "${batch_size}" \
      --epochs 3 --max_length 512 --lr 2e-5 --weight_decay 0.01 \
      --label_smoothing 0.1 --save_dir "${model_dir}" --log_dir "${log_dir}" \
      2>&1 | tee "${log_dir}/bert_${dataset}_seed${seed}.log"
    [[ -s "${checkpoint}" && -s "${history}" ]] || { echo "[ERROR] BERT output missing" >&2; exit 1; }
  done
done

for dataset in politifact gossipcop; do
  for seed in 42 2024 3407; do
    checkpoint="${model_dir}/socialdebias_${dataset}_en_seed${seed}_bert_baseline.pt"
    output="${bert_adv_dir}/bert_adv_${dataset}_seed${seed}.json"
    if [[ -s "${output}" && "${output}" -nt "${checkpoint}" ]]; then
      echo "[SKIP BERT EVAL] ${dataset}/seed${seed}"
      continue
    fi
    python -u scripts/evaluate_bert_adv.py \
      --dataset "${dataset}" --seed "${seed}" --variants A,B,C,D \
      --ckpt_dir "${model_dir}" --output_dir "${bert_adv_dir}" \
      2>&1 | tee "${log_dir}/bert_adv_${dataset}_seed${seed}.log"
    [[ -s "${output}" ]] || { echo "[ERROR] BERT eval output missing" >&2; exit 1; }
  done
done

[[ "$(find "${lstm_dir}" -maxdepth 1 -type f -name 'lstm_*_result.json' | wc -l)" -eq 6 ]]
[[ "$(find "${model_dir}" -maxdepth 1 -type f -name '*_bert_baseline_history.json' | wc -l)" -eq 9 ]]
[[ "$(find "${bert_adv_dir}" -maxdepth 1 -type f -name 'bert_adv_*.json' | wc -l)" -eq 6 ]]
echo "P2_CORE_BASELINES_READY=True"
