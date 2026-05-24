"""
train_bert_baseline.py — BERT-base 基线训练（与 SocialDebias 同主干 / 同切分 / 同超参）

设计与 SocialDebias 严格对齐，仅去除 surface 分支与所有损失项：
- 主干: bert-base-uncased，冻结除 encoder.layer.11 外的所有层（同 ENDEF / SD）
- 数据: utils.real_dataloader.load_dataset(val_ratio=0.15, seed=seed)（同 SD）
- 超参: lr=2e-5, batch_size=4, epoch=3（同 SD surface）
- 选模型: 按 val F1 best

ckpt 保存格式与 SD 兼容，但不含 surface_feat / feat_mean / feat_std。

用法:
    python scripts/train_bert_baseline.py \
        --dataset politifact --seed 42
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from utils.real_dataloader import load_dataset
from utils.device import get_device


class BertBaselineModel(nn.Module):
    """与 SD 主干完全一致：bert-base-uncased，仅微调 encoder.layer.11；
    分类头：dropout + Linear(768, 2)。"""
    def __init__(self, model_name="bert-base-uncased", num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name).requires_grad_(False)
        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask)
        # 用 [CLS] 的 last_hidden_state[:, 0, :]（与 SD 主流路径一致）
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        return self.classifier(x)


class BertTextDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        }


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_probs.extend(probs); all_preds.extend(preds); all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": acc, "f1": f1, "auc": auc,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["politifact", "gossipcop"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--save_dir", default="results/models")
    parser.add_argument("--log_dir", default="results/baseline_logs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    bert_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    print("=" * 80)
    print(f"BERT baseline 训练: {args.dataset} × seed={args.seed}")
    print(f"  lr={args.lr}  batch={args.batch_size}  epoch={args.epoch}  max_len={args.max_length}")
    print("=" * 80)

    # 数据加载（与 SD 同切分）
    train_samples, val_samples, test_samples = load_dataset(
        dataset_name=args.dataset, val_ratio=args.val_ratio, seed=args.seed,
    )
    print(f"数据规模: train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")

    train_set = BertTextDataset(train_samples, tokenizer, args.max_length)
    val_set = BertTextDataset(val_samples, tokenizer, args.max_length)
    test_set = BertTextDataset(test_samples, tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 模型
    model = BertBaselineModel(model_name=bert_name).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{args.save_dir}/socialdebias_{args.dataset}_en_seed{args.seed}_bert_baseline.pt"
    history = []
    best_val_f1 = -1.0
    best_test_metrics = None

    for ep in range(1, args.epoch + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"E{ep} train"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_m = evaluate(model, val_loader, device, criterion)
        test_m = evaluate(model, test_loader, device, criterion)
        dt = time.time() - t0
        print(f"[E{ep}] {dt:.0f}s | train_loss={total_loss/len(train_loader):.4f}")
        print(f"        val:  acc={val_m['accuracy']:.4f} f1={val_m['f1']:.4f} auc={val_m['auc']:.4f}")
        print(f"        test: acc={test_m['accuracy']:.4f} f1={test_m['f1']:.4f} auc={test_m['auc']:.4f}")

        is_best = val_m["f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_m["f1"]
            best_test_metrics = test_m
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "model_name": bert_name,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "epoch_best": ep,
                    "max_length": args.max_length,
                    "val_ratio": args.val_ratio,
                },
                "val_f1": best_val_f1,
                "test_metrics_at_best_val": test_m,
            }, save_path)
            print(f"        ⭐ 新最佳 val f1={best_val_f1:.4f}, ckpt 保存: {save_path}")

        history.append({
            "epoch": ep, "train_loss": total_loss / len(train_loader),
            "val": val_m, "test": test_m, "is_best": is_best,
        })

    # 保存 history
    hist_path = save_path.replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump({
            "args": vars(args),
            "best_val_f1": best_val_f1,
            "best_test_metrics": best_test_metrics,
            "history": history,
        }, f, indent=2)
    print(f"\nhistory 保存: {hist_path}")
    print(f"完成: best val f1={best_val_f1:.4f}  best test f1={best_test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
