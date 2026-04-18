"""
训练脚本：在假数据上训练一个 BERT 基线分类器。

这是整个项目的"骨架"——后面所有训练任务都基于这个脚本扩展。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import time

from utils.device import get_device, get_recommended_batch_size
from utils.dummy_data import generate_dummy_dataset, load_dummy_dataset
from utils.dataloader import create_dataloaders
from modeling.bert_classifier import BertClassifier
from configs.base_config import get_config


def evaluate(model, loader, device, criterion):
    """在指定 dataloader 上评估模型，返回各项指标"""
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
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # 假新闻的概率
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="macro"),
    }

    # AUC 只在两个类别都出现时才能算
    if len(set(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["auc"] = float("nan")

    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device, log_every=5):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # 前向
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"],
                        help="使用英文(en)还是中文(zh)数据")
    parser.add_argument("--mode", type=str, default="dev_real",
                        choices=["dev_dummy", "dev_real", "prod"])
    parser.add_argument("--dataset", type=str, default="politifact",
                        choices=["politifact", "gossipcop", "lun"])
    args = parser.parse_args()

    print("=" * 70)
    print(f"SocialDebias - 基线 BERT 训练 [{args.language.upper()}]")
    print("=" * 70)

    # 1. 配置
    config = get_config(mode=args.mode, language=args.language, dataset=args.dataset)
    print(f"\n配置: {config}")

    # 2. 设备
    device = get_device()
    print(f"\n设备: {device}")
    # 3. Tokenizer
    print(f"\n加载 tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 4. 数据加载器（根据 config 自动选假数据 / 真实数据）
    print("\n加载数据...")
    from utils.dataloader import create_dataloaders_auto
    train_loader, val_loader, test_loader = create_dataloaders_auto(config, tokenizer)
    # 6. 模型
    print(f"\n构建模型: {config.model_name}")
    model = BertClassifier(
        model_name=config.model_name,
        num_classes=config.num_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.1f}M")

    # 7. 优化器和损失
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 8. 训练循环
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    best_val_f1 = 0.0
    best_epoch = 0
    save_dir = "./results/models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/baseline_{args.dataset}_{args.language}.pt"

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n[Epoch {epoch}/{config.num_epochs}]")
        start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            log_every=config.log_every_n_steps
        )

        val_metrics = evaluate(model, val_loader, device, criterion)

        elapsed = time.time() - start
        print(
            f"  耗时: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_metrics["f1"],
                "val_acc": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
                "config": {
                    "model_name": config.model_name,
                    "language": args.language,
                    "dataset": args.dataset,
                },
            }, save_path)
            print(f"  ⭐ 新最佳！已保存到 {save_path}")

    # 加载最佳 checkpoint
    print(f"\n加载最佳 checkpoint (Epoch {best_epoch}, val F1={best_val_f1:.4f})")
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    # 9. 测试集评估
    print("\n" + "=" * 70)
    print("测试集最终评估")
    print("=" * 70)
    test_metrics = evaluate(model, test_loader, device, criterion)
    print(
        f"Test Acc: {test_metrics['accuracy']:.4f} | "
        f"Test F1: {test_metrics['f1']:.4f} | "
        f"Test AUC: {test_metrics['auc']:.4f}"
    )

    print("\n✅ 训练流水线跑通！")


if __name__ == "__main__":
    main()