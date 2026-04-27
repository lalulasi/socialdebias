"""
SocialDebias + 17 维表层特征训练脚本（意见 1 + 意见 8 合并实现）

使用:
    python scripts/train_socialdebias_surface.py \
      --dataset politifact --language en --seed 42 \
      --lambda_fact 1.0 --lambda_bias 0.5 --lambda_consist 0.3

    # 加 InfoNCE：
    python scripts/train_socialdebias_surface.py \
      --dataset politifact --language en --seed 42 \
      --use_contrastive \
      --lambda_contrast 0.3 \
      --orig_pkl data/sheepdog/news_articles/politifact_train.pkl \
      --adv_pkl data/qwen_adv/politifact_train_adv_filtered.pkl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from modeling.social_debias import SocialDebiasModel
from modeling.infonce import info_nce_loss
from utils.surface_features import SurfaceFeatureExtractor
from utils.dataloader import SurfaceAugmentedDataset
from utils.real_dataloader import load_dataset
from utils.device import get_device


def set_seed(seed):
    random.seed(seed);
    np.random.seed(seed)
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            surface = batch.get("surface_feat")
            if surface is not None:
                surface = surface.to(device)
            out = model(input_ids, attention_mask, surface_feat=surface)
            logits = out["fact_logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds);
            all_labels.extend(labels.numpy());
            all_probs.extend(probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="politifact",
                        choices=["politifact", "gossipcop", "lun"])
    parser.add_argument("--language", default="en")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lambda_fact", type=float, default=1.0)
    parser.add_argument("--lambda_bias", type=float, default=0.5)
    parser.add_argument("--lambda_consist", type=float, default=0.3)

    # 表层特征
    parser.add_argument("--surface_feat_dim", type=int, default=17,
                        help="表层特征维度（0 表示禁用）")

    # 对比学习（可选）
    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--lambda_contrast", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--orig_pkl", default=None)
    parser.add_argument("--adv_pkl", default=None)

    parser.add_argument("--save_dir", default="./results/models")
    parser.add_argument("--save_suffix", default="surface")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[Surface] device={device}, seed={args.seed}, surface_dim={args.surface_feat_dim}")

    bert_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    # ==== 加载数据 ====
    train_data, val_data, test_data = load_dataset(
        dataset_name=args.dataset, seed=args.seed,
    )

    # 表层特征提取器
    extractor = None
    if args.surface_feat_dim > 0:
        print("[Surface] 加载 SurfaceFeatureExtractor...")
        extractor = SurfaceFeatureExtractor()

    # 训练集（fit normalizer）
    train_set = SurfaceAugmentedDataset(
        train_data, tokenizer, args.max_length,
        surface_extractor=extractor,
    )
    # val/test 复用训练集的 normalizer
    normalizer = (train_set.feat_mean, train_set.feat_std) if extractor else None
    val_set = SurfaceAugmentedDataset(
        val_data, tokenizer, args.max_length,
        surface_extractor=extractor, normalizer=normalizer,
    )
    test_set = SurfaceAugmentedDataset(
        test_data, tokenizer, args.max_length,
        surface_extractor=extractor, normalizer=normalizer,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 对比学习数据（可选）
    contrast_loader = None
    if args.use_contrastive:
        if not args.adv_pkl or not args.orig_pkl:
            raise ValueError("--use_contrastive 需要同时指定 --orig_pkl 和 --adv_pkl")
        from utils.contrastive_dataloader import ContrastiveFakeNewsDataset
        # 注意：对比学习不带表层特征（保持简单，避免组合爆炸）
        contrast_set = ContrastiveFakeNewsDataset.from_pkl(
            original_pkl_path=args.orig_pkl,
            adversarial_pkl_path=args.adv_pkl,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        contrast_loader = DataLoader(
            contrast_set, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        print(f"[Surface] 对比学习启用，contrast batches={len(contrast_loader)}")

    # ==== 模型 ====
    model = SocialDebiasModel(
        model_name=bert_name, num_classes=2,
        hidden_dim=384, dropout=0.1, grl_lambda=1.0,
        use_frozen_bert=True,
        surface_feat_dim=args.surface_feat_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ==== 训练 ====
    save_path = (f"{args.save_dir}/socialdebias_{args.dataset}_{args.language}"
                 f"_seed{args.seed}_{args.save_suffix}.pt")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_test = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        ll = {"total": [], "fact": [], "bias": [], "consist": [], "contrast": []}

        # 主训练循环
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            surface = batch.get("surface_feat")
            if surface is not None:
                surface = surface.to(device)

            outputs = model(input_ids, attention_mask, surface_feat=surface)

            loss_fact = criterion(outputs["fact_logits"], labels)
            loss_bias = torch.tensor(0.0, device=device)
            loss_consist = torch.tensor(0.0, device=device)

            if args.lambda_bias > 0:
                loss_bias = criterion(outputs["bias_logits"], labels)
            if args.lambda_consist > 0 and "frozen_repr" in outputs:
                cos = torch.nn.functional.cosine_similarity(
                    outputs["shared_repr"], outputs["frozen_repr"], dim=-1
                )
                loss_consist = (1.0 - cos).mean()

            loss = (args.lambda_fact * loss_fact
                    + args.lambda_bias * loss_bias
                    + args.lambda_consist * loss_consist)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ll["total"].append(loss.item())
            ll["fact"].append(loss_fact.item())
            ll["bias"].append(loss_bias.item() if torch.is_tensor(loss_bias) else 0.0)
            ll["consist"].append(loss_consist.item() if torch.is_tensor(loss_consist) else 0.0)

        # 对比学习单独一个 sub-epoch（如果启用）
        if contrast_loader is not None and args.lambda_contrast > 0:
            for batch in contrast_loader:
                optimizer.zero_grad()
                orig_ids = batch["orig_input_ids"].to(device)
                orig_mask = batch["orig_attention_mask"].to(device)
                adv_ids = batch["adv_input_ids"].to(device)
                adv_mask = batch["adv_attention_mask"].to(device)
                # 对比学习不传 surface_feat（输入已不同，不做表层）
                out_orig = model(orig_ids, orig_mask, surface_feat=None)
                out_adv = model(adv_ids, adv_mask, surface_feat=None)
                loss_contrast = info_nce_loss(
                    out_orig["fact_repr"], out_adv["fact_repr"],
                    temperature=args.temperature,
                )
                loss = args.lambda_contrast * loss_contrast
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ll["contrast"].append(loss_contrast.item())

        epoch_time = time.time() - t0
        means = {k: (np.mean(v) if v else 0.0) for k, v in ll.items()}
        val_m = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch, "epoch_time": epoch_time,
            **{f"loss_{k}": v for k, v in means.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        })

        print(f"[E{epoch}] {epoch_time:.0f}s | total={means['total']:.4f} "
              f"fact={means['fact']:.4f} bias={means['bias']:.4f} "
              f"consist={means['consist']:.4f} contrast={means['contrast']:.4f}")
        print(f"        val acc={val_m['acc']:.4f} f1={val_m['f1']:.4f} auc={val_m['auc']:.4f}")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            test_m = evaluate(model, test_loader, device)
            best_test = {"epoch": epoch, **test_m}
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_m["f1"],
                "val_acc": val_m["acc"],
                "val_auc": val_m["auc"],
                "feat_mean": train_set.feat_mean.tolist() if extractor else None,
                "feat_std": train_set.feat_std.tolist() if extractor else None,
                "config": {
                    "model_name": bert_name,
                    "language": args.language, "dataset": args.dataset,
                    "lambda_fact": args.lambda_fact,
                    "lambda_bias": args.lambda_bias,
                    "lambda_consist": args.lambda_consist,
                    "lambda_contrast": args.lambda_contrast if args.use_contrastive else 0.0,
                    "surface_feat_dim": args.surface_feat_dim,
                },
            }, save_path)
            print(f"        ⭐ 新最佳 test acc={test_m['acc']:.4f} f1={test_m['f1']:.4f}")

    # 保存训练历史
    history_path = save_path.replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "args": vars(args),
            "best_val_f1": best_val_f1,
            "best_test": best_test,
            "history": history,
        }, f, indent=2, default=str)
    print(f"\n[Surface] 完成。ckpt: {save_path}")
    print(f"          best val F1: {best_val_f1:.4f}, best test F1: {best_test['f1'] if best_test else 'N/A'}")


if __name__ == "__main__":
    main()