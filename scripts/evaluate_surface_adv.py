"""
评估带表层特征的 ckpt 在对抗集上的鲁棒性。

使用:
    python scripts/evaluate_surface_adv.py \\
        --dataset politifact --language en --seed 42 \\
        --save_suffix surface
        
    python scripts/evaluate_surface_adv.py \\
        --dataset politifact --language en --seed 42 \\
        --save_suffix surface_contrast
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

from modeling.social_debias import SocialDebiasModel
from utils.surface_features import SurfaceFeatureExtractor
from utils.real_dataloader import load_dataset
from utils.device import get_device


class SurfaceTestDataset(Dataset):
    """评估用：用训练时的 normalizer 标准化新数据。"""
    def __init__(self, samples, tokenizer, max_length, extractor, feat_mean, feat_std):
        from tqdm import tqdm
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

        feats = []
        for s in tqdm(samples, desc="提取表层特征"):
            feats.append(extractor.extract(s["text"]))
        feats = np.stack(feats, axis=0).astype(np.float32)
        self.surface_features = (feats - feat_mean) / (feat_std + 1e-8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        encoded = self.tokenizer(
            s["text"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(s["label"], dtype=torch.long),
            "surface_feat": torch.tensor(self.surface_features[idx], dtype=torch.float32),
        }


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
            surface = batch["surface_feat"].to(device)
            out = model(input_ids, attention_mask, surface_feat=surface)
            logits = out["fact_logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(labels.numpy()); all_probs.extend(probs)
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
    parser.add_argument("--save_suffix", required=True,
                        help="surface 或 surface_contrast")
    parser.add_argument("--variants", default="A,B,C,D")
    parser.add_argument("--output_dir", default="results/surface_adv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    device = get_device()
    bert_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    ckpt_path = (f"./results/models/socialdebias_{args.dataset}_{args.language}"
                 f"_seed{args.seed}_{args.save_suffix}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] ckpt 不存在: {ckpt_path}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"对抗鲁棒性评估")
    print(f"  ckpt: {ckpt_path}")
    print("=" * 80)

    # 加载 ckpt
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    surface_dim = ckpt["config"].get("surface_feat_dim", 0)
    feat_mean = np.array(ckpt["feat_mean"]) if ckpt.get("feat_mean") else None
    feat_std = np.array(ckpt["feat_std"]) if ckpt.get("feat_std") else None

    model = SocialDebiasModel(
        model_name=bert_name, num_classes=2,
        hidden_dim=384, dropout=0.1, grl_lambda=1.0,
        use_frozen_bert=True,
        surface_feat_dim=surface_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"加载 ckpt (val_f1={ckpt.get('val_f1', 'N/A'):.4f}, surface_dim={surface_dim})")

    extractor = SurfaceFeatureExtractor() if surface_dim > 0 else None

    # 干净集
    _, _, clean_samples = load_dataset(dataset_name=args.dataset, seed=args.seed)
    clean_set = SurfaceTestDataset(
        clean_samples, tokenizer, args.max_length,
        extractor, feat_mean, feat_std,
    )
    clean_loader = DataLoader(clean_set, batch_size=args.batch_size, shuffle=False)
    
    results = OrderedDict()
    clean_m = evaluate_model(model, clean_loader, device)
    results["clean"] = clean_m
    print(f"\n干净集: Acc={clean_m['accuracy']:.4f} F1={clean_m['f1']:.4f}")

    # 4 个对抗变体
    for v in args.variants.split(","):
        v = v.strip()
        adv_samples = load_adversarial_test(args.dataset, v)
        adv_set = SurfaceTestDataset(
            adv_samples, tokenizer, args.max_length,
            extractor, feat_mean, feat_std,
        )
        adv_loader = DataLoader(adv_set, batch_size=args.batch_size, shuffle=False)
        adv_m = evaluate_model(model, adv_loader, device)
        results[f"adv_{v}"] = adv_m
        print(f"对抗 {v}: Acc={adv_m['accuracy']:.4f} F1={adv_m['f1']:.4f}")

    # 汇总
    adv_f1s = [results[f"adv_{v}"]["f1"] for v in args.variants.split(",")]
    avg_adv_f1 = sum(adv_f1s) / len(adv_f1s)
    f1_drop = clean_m["f1"] - avg_adv_f1
    
    results["summary"] = {
        "clean_f1": clean_m["f1"],
        "avg_adv_f1": avg_adv_f1,
        "f1_drop": f1_drop,
    }
    
    print(f"\nAvg Adv F1: {avg_adv_f1:.4f}")
    print(f"F1 Drop: {f1_drop*100:.2f}pp")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"surface_adv_{args.dataset}_seed{args.seed}_{args.save_suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": ckpt_path,
            "save_suffix": args.save_suffix,
            "seed": args.seed,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n结果保存: {out_path}")


if __name__ == "__main__":
    main()