"""
消融实验（意见 20）：通过 λ 组合控制不同变体
    full        λ_bias=0.5, λ_consist=0.3
    no_grl      λ_bias=0.0, λ_consist=0.3   # 关掉偏置分支反传
    no_consist  λ_bias=0.5, λ_consist=0.0   # 关掉语义一致性
    no_both     λ_bias=0.0, λ_consist=0.0   # 两个都关（退化为 BERT 基线）

用法:
    python scripts/train_ablation.py --dataset politifact --variant full --seed 42
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from modeling.social_debias import SocialDebiasModel  # 按实际 class 名调整
from utils.dataloader import create_dataloaders_auto
from utils.device import get_device


# ==================== 损失计算 ====================

def compute_loss(outputs, labels, lambdas, criterion):
    """
    按 λ 组合计算总损失。
    outputs: model forward 返回的 dict
    lambdas: dict {lambda_fact, lambda_bias, lambda_consist}

    语义一致性约束：shared_repr 与 frozen_repr 的余弦相似度
    （两个都是 768 维 BERT [CLS]，维度天然对齐）
    """
    device = outputs["fact_logits"].device

    # L_fact：主任务损失（必有）
    loss_fact = criterion(outputs["fact_logits"], labels)
    total = lambdas["lambda_fact"] * loss_fact

    loss_bias = torch.tensor(0.0, device=device)
    loss_consist = torch.tensor(0.0, device=device)

    # L_bias：偏置分支损失（通过 GRL 自动反转梯度）
    if lambdas["lambda_bias"] > 0 and "bias_logits" in outputs:
        loss_bias = criterion(outputs["bias_logits"], labels)
        total = total + lambdas["lambda_bias"] * loss_bias

    # L_consist：语义一致性（要求开启 frozen_bert 才有 frozen_repr）
    if lambdas["lambda_consist"] > 0 and "frozen_repr" in outputs:
        shared = outputs["shared_repr"]
        frozen = outputs["frozen_repr"]
        cos = torch.nn.functional.cosine_similarity(shared, frozen, dim=-1)
        loss_consist = (1.0 - cos).mean()
        total = total + lambdas["lambda_consist"] * loss_consist

    return total, {
        "total": total.item(),
        "fact": loss_fact.item(),
        "bias": loss_bias.item() if torch.is_tensor(loss_bias) else 0.0,
        "consist": loss_consist.item() if torch.is_tensor(loss_consist) else 0.0,
    }


# ==================== 评估 ====================

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            out = model(input_ids, attention_mask)
            logits = out["fact_logits"]
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


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["politifact", "gossipcop", "lun"])
    parser.add_argument("--variant", required=True,
                        choices=["full", "no_grl", "no_consist", "no_both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bert_name", type=str, default="bert-base-uncased")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--output_dir", type=str, default="results/ablation")
    args = parser.parse_args()

    # λ 组合
    LAMBDA_MAP = {
        "full":       {"lambda_fact": 1.0, "lambda_bias": 0.5, "lambda_consist": 0.3},
        "no_grl":     {"lambda_fact": 1.0, "lambda_bias": 0.0, "lambda_consist": 0.3},
        "no_consist": {"lambda_fact": 1.0, "lambda_bias": 0.5, "lambda_consist": 0.0},
        "no_both":    {"lambda_fact": 1.0, "lambda_bias": 0.0, "lambda_consist": 0.0},
    }
    lambdas = LAMBDA_MAP[args.variant]
    print(f"[Ablation] dataset={args.dataset} variant={args.variant} seed={args.seed}")
    print(f"           λ = {lambdas}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    # ==== 构造 config（SimpleNamespace 模拟真 config 对象） ====
    config = SimpleNamespace(
        use_dummy_data=False,
        dataset_name=args.dataset,
        max_train_samples=None,
        batch_size=args.batch_size,
        max_length=args.max_length,
        language=args.language,
    )

    # ==== Tokenizer + DataLoader ====
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    train_loader, val_loader, test_loader = create_dataloaders_auto(
        config, tokenizer, seed=args.seed
    )
    print(f"[Ablation] train batches={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # ==== 模型 ====
    # 注意：即使 lambda_consist=0，仍然保持 use_frozen_bert=True，
    # 这样结构一致，只是不用 frozen_repr 算损失而已
    # （如果你的 SocialDebiasModel 构造函数签名不同，按实际改）
    model = SocialDebiasModel(
        model_name=args.bert_name,  # 参数名改成 model_name
        num_classes=2,
        hidden_dim=384,
        dropout=0.1,
        grl_lambda=1.0,
        use_frozen_bert=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # ==== 训练 ====
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ablation_{args.dataset}_{args.variant}_seed{args.seed}.pt"

    best_val_f1 = -1.0
    best_test_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss, info = compute_loss(outputs, labels, lambdas, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(info)

        avg_total = np.mean([x["total"] for x in batch_losses])
        avg_fact = np.mean([x["fact"] for x in batch_losses])
        avg_bias = np.mean([x["bias"] for x in batch_losses])
        avg_consist = np.mean([x["consist"] for x in batch_losses])

        val_m = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "loss_total": avg_total, "loss_fact": avg_fact,
            "loss_bias": avg_bias, "loss_consist": avg_consist,
            **{f"val_{k}": v for k, v in val_m.items()},
        })
        print(f"[E{epoch}] total={avg_total:.4f} fact={avg_fact:.4f} "
              f"bias={avg_bias:.4f} consist={avg_consist:.4f}")
        print(f"       val acc={val_m['acc']:.4f} f1={val_m['f1']:.4f} auc={val_m['auc']:.4f}")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            test_m = evaluate(model, test_loader, device)
            best_test_metrics = {"epoch": epoch, **test_m}
            torch.save(model.state_dict(), ckpt_path)
            print(f"       [best!] test acc={test_m['acc']:.4f} f1={test_m['f1']:.4f}")

    # ==== 结果保存 ====
    result = {
        "dataset": args.dataset,
        "variant": args.variant,
        "lambdas": lambdas,
        "seed": args.seed,
        "args": vars(args),
        "best_val_f1": best_val_f1,
        "test": best_test_metrics,
        "history": history,
    }
    result_path = out_dir / f"ablation_{args.dataset}_{args.variant}_seed{args.seed}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[Ablation] 结果已保存：{result_path}")


if __name__ == "__main__":
    main()