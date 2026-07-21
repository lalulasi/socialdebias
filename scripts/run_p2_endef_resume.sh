#!/usr/bin/env bash
# Resume-safe ENDEF data preparation, 18 training jobs and English evaluation.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
batch_root=/autodl-fs/data/socialdebias/results/paper_v2_20260719
endef_root=/autodl-fs/data/ENDEF-SIGIR2022
endef_en=${endef_root}/ENDEF_en
endef_ch=${endef_root}/ENDEF_ch
venv_activate=${endef_root}/endefvenv/bin/activate
log_dir=${batch_root}/logs/endef
evaluation_dir=${batch_root}/endef_adv

cd "${project_root}"
[[ -f "${venv_activate}" ]] || {
  echo "[ERROR] ENDEF venv missing; first run: bash scripts/setup_endef_env.sh ${endef_root}" >&2
  exit 1
}
source "${venv_activate}"
mkdir -p "${log_dir}" "${evaluation_dir}"
python scripts/patch_endef_ch_dataloader.py --endef_ch_root "${endef_ch}"

data_ready=true
for root in "${endef_en}/data_politifact" "${endef_en}/data_gossipcop" "${endef_ch}/data_weibo21"; do
  for split in train val test; do [[ -s "${root}/${split}.json" ]] || data_ready=false; done
done
if [[ "${data_ready}" != true ]]; then
  python -u prepare_endef_data.py 2>&1 | tee "${log_dir}/prepare_endef_data.log"
fi

for root in "${endef_en}" "${endef_ch}"; do
  mkdir -p "${root}/backup_ckpt" "${root}/logs/json" "${root}/logs/param" "${root}/logs/event" "${root}/param_model"
done

for dataset in politifact gossipcop; do
  for model in bert bert_endef; do
    if [[ "${model}" == bert ]]; then checkpoint_name=parameter_bert.pkl; else checkpoint_name=parameter_bertendef.pkl; fi
    for seed in 42 2024 3407; do
      source_checkpoint="${endef_en}/param_model/${model}/${checkpoint_name}"
      backup="${endef_en}/backup_ckpt/${dataset}_${model}_seed${seed}.pkl"
      if [[ -s "${backup}" ]]; then echo "[SKIP ENDEF EN] ${dataset}/${model}/seed${seed}"; continue; fi
      cd "${endef_en}"
      python -u main.py --model_name "${model}" --root_path "./data_${dataset}/" \
        --seed "${seed}" --lr 7e-05 --batchsize 64 --max_len 170 --epoch 3 \
        --save_param_dir ./param_model 2>&1 | tee "${log_dir}/${dataset}_${model}_seed${seed}.log"
      [[ -s "${source_checkpoint}" ]] || { echo "[ERROR] ENDEF checkpoint missing" >&2; exit 1; }
      cp -p "${source_checkpoint}" "${backup}"
    done
  done
done

for model in bert bert_endef; do
  if [[ "${model}" == bert ]]; then checkpoint_name=parameter_bert.pkl; else checkpoint_name=parameter_bertendef.pkl; fi
  for seed in 42 2024 3407; do
    source_checkpoint="${endef_ch}/param_model/${model}/${checkpoint_name}"
    backup="${endef_ch}/backup_ckpt/weibo21_${model}_seed${seed}.pkl"
    if [[ -s "${backup}" ]]; then echo "[SKIP ENDEF CH] ${model}/seed${seed}"; continue; fi
    cd "${endef_ch}"
    python -u main.py --model_name "${model}" --root_path ./data_weibo21/ \
      --seed "${seed}" --lr 7e-05 --batchsize 64 --max_len 170 --epoch 3 \
      --save_param_dir ./param_model 2>&1 | tee "${log_dir}/weibo21_${model}_seed${seed}.log"
    [[ -s "${source_checkpoint}" ]] || { echo "[ERROR] ENDEF CH checkpoint missing" >&2; exit 1; }
    cp -p "${source_checkpoint}" "${backup}"
  done
done

[[ "$(find "${endef_en}/backup_ckpt" -maxdepth 1 -type f -name '*.pkl' | wc -l)" -eq 12 ]]
[[ "$(find "${endef_ch}/backup_ckpt" -maxdepth 1 -type f -name '*.pkl' | wc -l)" -eq 6 ]]
cd "${project_root}"
SOCIALDEBIAS_ROOT="${project_root}" ENDEF_EN_ROOT="${endef_en}" \
  python -u evaluate_endef_adversarial.py --batch --output_dir "${evaluation_dir}" \
  2>&1 | tee "${log_dir}/endef_adv.log"
[[ "$(find "${evaluation_dir}" -maxdepth 1 -type f -name 'endef_adv_*.json' | wc -l)" -eq 12 ]]
echo "P2_ENDEF_READY=True"
