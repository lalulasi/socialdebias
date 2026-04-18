"""
训练 SocialDebias 双分支模型。
相比 train_baseline.py，主要变化：
1. 模型换成 SocialDebiasModel
2. 损失函数从单一交叉熵变成三项加权和
3. 评估时只用事实分支的预测（去偏后的结果）
4. 日志里显示三个损失分量各自的变化
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import time

from utils.device import get_device
from utils.dummy_data import generate_dummy_dataset, load_dummy_dataset
from utils.dataloader import create_dataloaders
from modeling.social_debias import SocialDebiasModel, compute_losses
from configs.base_config import get_config


def evaluate(model, loader, device):
    """评估：只用事实分支的预测（这是真正的去偏输出）"""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # 推理只用事实分支
            logits = model.predict(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="macro"),
    }
    if len(set(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["auc"] = float("nan")
    return metrics


def evaluate_bias_branch(model, loader, device):
    """
    诊断用：评估偏置分支的准确率。
    理想情况下，对抗训练收敛后，偏置分支应该接近随机猜测（acc ~ 0.5），
    因为共享表示已经不再包含偏置分支能利用的信息。
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = outputs["bias_logits"].argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


def train_one_epoch(model, loader, optimizer, device, loss_weights, log_every=5):
    """训练一个 epoch，记录每个损失分量"""
    model.train()
    # 冻结 BERT 要保持 eval 模式（BatchNorm 等会受影响）
    if model.use_frozen_bert:
        model.frozen_bert.eval()

    running = {"L_total": 0, "L_fact": 0, "L_bias": 0, "L_consist": 0}
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # 前向
        outputs = model(input_ids, attention_mask)
        total_loss, loss_dict = compute_losses(outputs, labels, weights=loss_weights)

        # 反向
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 累计各分量
        for k in running:
            running[k] += loss_dict[k].item()
        n_batches += 1

        if step % log_every == 0:
            pbar.set_postfix(
                fact=f"{loss_dict['L_fact'].item():.3f}",
                bias=f"{loss_dict['L_bias'].item():.3f}",
                consist=f"{loss_dict['L_consist'].item():.3f}",
            )

    return {k: v / n_batches for k, v in running.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--mode", type=str, default="dev_real",
                        choices=["dev_dummy", "dev_real", "prod"])
    parser.add_argument("--dataset", type=str, default="politifact",
                        choices=["politifact", "gossipcop", "lun"])
    parser.add_argument("--lambda_fact", type=float, default=1.0)
    parser.add_argument("--lambda_bias", type=float, default=1.0)
    parser.add_argument("--lambda_consist", type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 70)
    print(f"SocialDebias 双分支模型训练 [{args.language.upper()}]")
    print("=" * 70)

    config = get_config(mode=args.mode, language=args.language, dataset=args.dataset)
    device = get_device()

    print(f"配置: mode={args.mode}, language={args.language}")
    print(f"设备: {device}")
    print(f"损失权重: fact={args.lambda_fact}, bias={args.lambda_bias}, consist={args.lambda_consist}")

    # 数据
    '''
    dummy_path = f"./data/dummy/dummy_data_{args.language}.json"
    if not os.path.exists(dummy_path):
        generate_dummy_dataset(n_samples=config.max_samples, language=args.language)
    data = load_dummy_dataset(path=dummy_path)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_loader, val_loader, test_loader = create_dataloaders(
        data, tokenizer, batch_size=config.batch_size, max_length=config.max_length,
    )
    '''
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # 数据加载器（根据 config 自动选假数据 / 真实数据）
    from utils.dataloader import create_dataloaders_auto
    train_loader, val_loader, test_loader = create_dataloaders_auto(config, tokenizer)

    # 模型
    print(f"\n构建模型 (加载两份 BERT，约 800MB)...")
    model = SocialDebiasModel(
        model_name=config.model_name,
        num_classes=config.num_classes,
        use_frozen_bert=True,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {n_trainable / 1e6:.1f}M")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay
    )

    loss_weights = {
        "fact": args.lambda_fact,
        "bias": args.lambda_bias,
        "consist": args.lambda_consist,
    }

    # 训练
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    # 最佳模型追踪
    best_val_f1 = 0.0
    best_epoch = 0
    save_dir = "./results/models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/socialdebias_{args.dataset}_{args.language}.pt"

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n[Epoch {epoch}/{config.num_epochs}]")
        start = time.time()

        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, loss_weights,
            log_every=config.log_every_n_steps,
        )

        val_metrics = evaluate(model, val_loader, device)
        bias_acc = evaluate_bias_branch(model, val_loader, device)

        elapsed = time.time() - start
        print(
            f"  耗时: {elapsed:.1f}s\n"
            f"  训练损失: total={train_losses['L_total']:.4f} | "
            f"fact={train_losses['L_fact']:.4f} | "
            f"bias={train_losses['L_bias']:.4f} | "
            f"consist={train_losses['L_consist']:.4f}\n"
            f"  验证 [事实分支]: Acc={val_metrics['accuracy']:.4f} | "
            f"F1={val_metrics['f1']:.4f} | AUC={val_metrics['auc']:.4f}\n"
            f"  验证 [偏置分支]: Acc={bias_acc:.4f} (理想情况应接近 0.5，表明偏置被去除)"
        )
        # 如果这个 epoch 是新最佳，保存 checkpoint
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
                    "lambda_fact": args.lambda_fact,
                    "lambda_bias": args.lambda_bias,
                    "lambda_consist": args.lambda_consist,
                },
            }, save_path)
            print(f"  💾 新最佳！已保存到 {save_path}")

    # 测试
    print("\n" + "=" * 70)
    print("测试集最终评估")
    print("=" * 70)
    test_metrics = evaluate(model, test_loader, device)
    test_bias = evaluate_bias_branch(model, test_loader, device)
    print(f"Test [事实分支]: Acc={test_metrics['accuracy']:.4f} | "
          f"F1={test_metrics['f1']:.4f} | AUC={test_metrics['auc']:.4f}")
    print(f"Test [偏置分支]: Acc={test_bias:.4f}")
    # 测试前加载最佳 checkpoint（而不是用最后一个 epoch 的模型）
    print(f"\n加载最佳 checkpoint (来自 Epoch {best_epoch}, val F1={best_val_f1:.4f})")
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("\n✅ 双分支训练完成")


if __name__ == "__main__":
    main()