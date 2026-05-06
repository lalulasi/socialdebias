"""
评估对比学习 ckpt 在对抗集上的鲁棒性。
基于 evaluate_adversarial.py 改造，加载 contrastive 版本的 ckpt。

使用:
    python scripts/evaluate_contrastive_adv.py \\
        --dataset politifact --language en --seed 42 --lambda_contrast 0.3
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from modeling.social_debias import SocialDebiasModel
from utils.dataloader import FakeNewsDataset
from utils.real_dataloader import load_dataset
from utils.device import get_device


def load_adversarial_test(dataset: str, variant: str) -> list:
    """加载对抗测试集（变体 A/B/C/D）"""
    import pickle
    path = f"./data/sheepdog/adversarial_test/{dataset}_test_adv_{variant}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    samples = []
    for i, (text, label) in enumerate(zip(data["news"], data["labels"])):
        samples.append({
            "id": f"{dataset}_test_adv_{variant}_{i}",
            "text": text,
            "label": int(label),
            "language": "en",
        })
    return samples


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            outputs = model(input_ids, attention_mask)
            logits = outputs["fact_logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lambda_contrast", type=float, required=True,
                        help="用于查找 ckpt 的 lambda_contrast 值")
    parser.add_argument("--variants", default="A,B,C,D")
    parser.add_argument("--output_dir", default="results/contrastive_adv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    device = get_device()
    bert_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    # 构造 ckpt 路径
    suffix = f"lc{args.lambda_contrast}"
    ckpt_path = (f"./results/models/socialdebias_{args.dataset}_{args.language}"
                 f"_seed{args.seed}_{suffix}.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] ckpt 不存在: {ckpt_path}")
        sys.exit(1)
    
    print(f"=" * 80)
    print(f"对抗鲁棒性评估")
    print(f"  ckpt: {ckpt_path}")
    print(f"  设备: {device}")
    print("=" * 80)
    
    # 加载模型
    model = SocialDebiasModel(
        model_name=bert_name, num_classes=2,
        hidden_dim=384, dropout=0.1,
        grl_lambda=1.0, use_frozen_bert=True,
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"加载 ckpt (val_f1={ckpt.get('val_f1', 'N/A'):.4f})")

    # 加载干净测试集
    _, _, clean_samples = load_dataset(dataset_name=args.dataset, seed=42)
    clean_set = FakeNewsDataset(clean_samples, tokenizer, args.max_length)
    clean_loader = DataLoader(clean_set, batch_size=args.batch_size, shuffle=False)
    
    results = OrderedDict()
    clean_m = evaluate_model(model, clean_loader, device)
    results["clean"] = clean_m
    print(f"\n干净集: Acc={clean_m['accuracy']:.4f} F1={clean_m['f1']:.4f}")

    # 评估各对抗变体
    variants = args.variants.split(",")
    for v in variants:
        adv_samples = load_adversarial_test(args.dataset, v)
        adv_set = FakeNewsDataset(adv_samples, tokenizer, args.max_length)
        adv_loader = DataLoader(adv_set, batch_size=args.batch_size, shuffle=False)
        adv_m = evaluate_model(model, adv_loader, device)
        results[f"adv_{v}"] = adv_m
        print(f"对抗 {v}: Acc={adv_m['accuracy']:.4f} F1={adv_m['f1']:.4f}")

    # 计算平均
    adv_f1s = [results[f"adv_{v}"]["f1"] for v in variants]
    avg_adv_f1 = sum(adv_f1s) / len(adv_f1s)
    f1_drop = clean_m["f1"] - avg_adv_f1
    
    results["summary"] = {
        "clean_f1": clean_m["f1"],
        "avg_adv_f1": avg_adv_f1,
        "f1_drop": f1_drop,
        "retention_rate": avg_adv_f1 / clean_m["f1"] if clean_m["f1"] > 0 else 0,
    }
    
    print(f"\n=== 鲁棒性汇总 ===")
    print(f"  Clean F1: {clean_m['f1']:.4f}")
    print(f"  Avg Adv F1: {avg_adv_f1:.4f}")
    print(f"  F1 降幅: {f1_drop*100:.2f}pp")

    # 保存
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"contrastive_adv_{args.dataset}_seed{args.seed}_lc{args.lambda_contrast}.json"
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": ckpt_path,
            "lambda_contrast": args.lambda_contrast,
            "seed": args.seed,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n结果保存: {out_path}")


if __name__ == "__main__":
    main()