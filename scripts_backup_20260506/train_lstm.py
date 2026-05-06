"""
LSTM 基线训练脚本（意见 19）。
用法:
    python scripts/train_lstm.py --dataset politifact --seed 42
    python scripts/train_lstm.py --dataset gossipcop --seed 42 --glove_path ./glove.840B.300d.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from modeling.lstm_classifier import BiLSTMClassifier
from utils.lstm_dataloader import (
    LSTMTextDataset, build_vocab, load_glove, PAD_IDX
)
from utils.device import get_device
from utils.real_dataloader import load_dataset


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            logits = model(input_ids, attention_mask)
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
    return {"acc": acc, "f1": f1, "auc": auc}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["politifact", "gossipcop", "lun"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--glove_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="results/lstm")
    args = parser.parse_args()

    # 种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"[LSTM] device={device}, seed={args.seed}, dataset={args.dataset}")

    # ==== 加载数据 ====
    train_samples, val_samples, test_samples = load_dataset(
        dataset_name=args.dataset,
        max_train_samples=None,
        seed=args.seed,
    )

    # ==== 构建词表（只用训练集文本） ====
    train_texts = [s["text"] for s in train_samples]
    vocab = build_vocab(train_texts, max_vocab_size=args.vocab_size)
    print(f"[LSTM] vocab_size={len(vocab)}")

    # ==== GloVe（可选） ====
    pretrained = None
    if args.glove_path:
        pretrained = load_glove(args.glove_path, vocab, args.embed_dim)

    # ==== DataLoader ====
    train_ds = LSTMTextDataset(train_samples, vocab, args.max_len)
    val_ds = LSTMTextDataset(val_samples, vocab, args.max_len)
    test_ds = LSTMTextDataset(test_samples, vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    # ==== 模型 ====
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=PAD_IDX,
        pretrained_embedding=pretrained,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    # 与 BERT 基线保持 label_smoothing=0.1 一致
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # ==== 训练 ====
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"lstm_{args.dataset}_seed{args.seed}.pt"

    best_val_f1 = -1.0
    best_val_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}})
        print(f"[E{epoch}] loss={train_loss:.4f} | val acc={val_m['acc']:.4f} f1={val_m['f1']:.4f} auc={val_m['auc']:.4f}")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_val_metrics = {"epoch": epoch, **val_m}
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "args": vars(args),
                "best_epoch": epoch,
            }, ckpt_path)

    # ==== 最终测试 ====
    print(f"\n[LSTM] best val F1={best_val_f1:.4f} @ epoch {best_val_metrics['epoch']}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, device)
    print(f"[LSTM] TEST acc={test_m['acc']:.4f} f1={test_m['f1']:.4f} auc={test_m['auc']:.4f}")

    # ==== 保存结果 ====
    result = {
        "dataset": args.dataset,
        "seed": args.seed,
        "args": vars(args),
        "best_val": best_val_metrics,
        "test": test_m,
        "history": history,
    }
    result_path = out_dir / f"lstm_{args.dataset}_seed{args.seed}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[LSTM] 结果已保存：{result_path}")


if __name__ == "__main__":
    main()