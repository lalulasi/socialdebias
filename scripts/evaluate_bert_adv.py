"""
evaluate_bert_adv.py — BERT baseline 在 SheepDog adv_A/B/C/D 上的鲁棒性评估
与 evaluate_surface_adv.py 同口径（paired ASR + sample-level）。

用法:
    python scripts/evaluate_bert_adv.py \
        --dataset politifact --seed 42
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils.real_dataloader import load_dataset
from utils.device import get_device

# 复用训练脚本里的模型 + dataset 定义
from train_bert_baseline import BertBaselineModel, BertTextDataset


def load_adversarial_test(dataset, variant):
    path = f"./data/sheepdog/adversarial_test/{dataset}_test_adv_{variant}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [{"text": t, "label": int(l), "language": "en"}
            for t, l in zip(data["news"], data["labels"])]


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(labels.numpy()); all_probs.extend(probs)
    y_pred = np.array(all_preds, dtype=np.int64)
    y_true = np.array(all_labels, dtype=np.int64)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, all_probs)
    except ValueError:
        auc = float("nan")
    metrics = {"accuracy": acc, "f1": f1, "auc": auc}
    return metrics, y_pred, y_true


def compute_paired_asr(clean_pred, clean_label, adv_pred):
    """Paired ASR = #(clean 正确 ∩ adv 错误) / #(clean 正确)"""
    clean_correct = (clean_pred == clean_label)
    adv_wrong = (adv_pred != clean_label)
    attacked = clean_correct & adv_wrong
    denom = int(clean_correct.sum())
    if denom == 0:
        return float("nan"), 0, 0
    asr = float(attacked.sum()) / denom
    return asr, int(attacked.sum()), denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["politifact", "gossipcop"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variants", default="A,B,C,D")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", default="results/bert_adv")
    args = parser.parse_args()

    device = get_device()
    bert_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    ckpt_path = f"./results/models/socialdebias_{args.dataset}_en_seed{args.seed}_bert_baseline.pt"
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] ckpt 不存在: {ckpt_path}")
        sys.exit(1)

    print("=" * 80)
    print(f"BERT baseline 对抗鲁棒性评估: {args.dataset} × seed={args.seed}")
    print(f"  ckpt: {ckpt_path}")
    print("=" * 80)

    # 加载 ckpt
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BertBaselineModel(model_name=bert_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    val_f1 = ckpt.get("val_f1", float("nan"))
    print(f"加载完成 (val_f1={val_f1:.4f})")

    # 干净集（与 SD 同切分）
    _, _, clean_samples = load_dataset(dataset_name=args.dataset, seed=args.seed)
    clean_set = BertTextDataset(clean_samples, tokenizer, args.max_length)
    clean_loader = DataLoader(clean_set, batch_size=args.batch_size, shuffle=False)

    results = OrderedDict()
    sample_preds = {}
    clean_m, clean_y_pred, clean_y_true = evaluate_model(model, clean_loader, device)
    results["clean"] = clean_m
    sample_preds["clean"] = {"y_pred": clean_y_pred, "y_true": clean_y_true}
    print(f"\n干净集 ({len(clean_samples):>4}): acc={clean_m['accuracy']:.4f}  f1={clean_m['f1']:.4f}  auc={clean_m['auc']:.4f}")

    # 4 个对抗变体
    for v in args.variants.split(","):
        v = v.strip()
        adv_samples = load_adversarial_test(args.dataset, v)
        adv_set = BertTextDataset(adv_samples, tokenizer, args.max_length)
        adv_loader = DataLoader(adv_set, batch_size=args.batch_size, shuffle=False)
        adv_m, adv_y_pred, adv_y_true = evaluate_model(model, adv_loader, device)
        results[f"adv_{v}"] = adv_m
        sample_preds[f"adv_{v}"] = {"y_pred": adv_y_pred, "y_true": adv_y_true}
        print(f"对抗 {v}   ({len(adv_samples):>4}): acc={adv_m['accuracy']:.4f}  f1={adv_m['f1']:.4f}  auc={adv_m['auc']:.4f}")

    # paired ASR
    asr_per_variant = {}
    for v in args.variants.split(","):
        v = v.strip()
        adv_y_pred = sample_preds[f"adv_{v}"]["y_pred"]
        adv_y_true = sample_preds[f"adv_{v}"]["y_true"]
        if not np.array_equal(clean_y_true, adv_y_true):
            print(f"  ⚠ 警告: adv_{v} label 序列与 clean 不一致，ASR 不可靠")
        asr, attacked_n, base_n = compute_paired_asr(
            sample_preds["clean"]["y_pred"], clean_y_true, adv_y_pred
        )
        asr_per_variant[f"adv_{v}"] = {
            "asr": asr, "attacked": attacked_n, "clean_correct": base_n,
        }

    # 汇总
    adv_f1s = [results[f"adv_{v.strip()}"]["f1"] for v in args.variants.split(",")]
    avg_adv_f1 = sum(adv_f1s) / len(adv_f1s)
    f1_drop = clean_m["f1"] - avg_adv_f1
    avg_asr = float(np.mean([asr_per_variant[k]["asr"] for k in asr_per_variant]))

    results["summary"] = {
        "clean_f1": clean_m["f1"],
        "avg_adv_f1": avg_adv_f1,
        "f1_drop": f1_drop,
        "asr_per_variant": asr_per_variant,
        "avg_asr": avg_asr,
    }

    print(f"\nAvg Adv F1: {avg_adv_f1:.4f}")
    print(f"F1 Drop: {f1_drop*100:.2f}pp")
    print("ASR per variant: " +
          ", ".join([f"{v.strip()}={asr_per_variant[f'adv_{v.strip()}']['asr']*100:.2f}%"
                     for v in args.variants.split(",")]) +
          f"   avg={avg_asr*100:.2f}%")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bert_adv_{args.dataset}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": ckpt_path,
            "model": "bert_baseline",
            "seed": args.seed,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n结果保存: {out_path}")


if __name__ == "__main__":
    main()
