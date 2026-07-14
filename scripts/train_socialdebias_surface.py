"""
SocialDebias training with surface features, optional InfoNCE, and NLI soft labels.

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
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from modeling.social_debias import SocialDebiasModel
from modeling.infonce import info_nce_loss, info_nce_loss_weighted
from utils.surface_features import SurfaceFeatureExtractor
from utils.dataloader import SurfaceAugmentedDataset
from utils.real_dataloader import load_dataset
from utils.device import get_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    parser.add_argument("--use_weibo21", action="store_true",
                        help="使用 Weibo21 中文数据集")
    parser.add_argument("--use_liar", action="store_true",
                        help="使用 LIAR 数据集（speaker 特征作为社交代理）")
    parser.add_argument("--surface_feat_dim", type=int, default=8,
                        help="表层特征维度（0 表示禁用）")

    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--lambda_contrast", type=float, default=0.3)
    parser.add_argument("--use_soft_labels", action="store_true",
                        help="使用 NLI p_entail 作为 InfoNCE 权重")
    parser.add_argument("--alpha_floor", type=float, default=0.5,
                        help="p_entail 权重下限")
    parser.add_argument("--lambda_fact_soft", type=float, default=0.0,
                        help="改写样本分类软标签损失权重；0 表示关闭")
    parser.add_argument("--soft_label_floor", type=float, default=0.5,
                        help="分类软标签的 α 下限：α=max(soft_label_floor, p_entail)")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--orig_pkl", default=None)
    parser.add_argument("--adv_pkl", default=None)

    parser.add_argument("--save_dir", default="./results/models")
    parser.add_argument("--save_suffix", default="surface")
    parser.add_argument("--adaptive_lambda", action="store_true",
                        help="启用基于训练集规模与 BERT baseline F1 的自适应 λ 缩放")
    parser.add_argument("--adaptive_size_thresh", type=int, default=1000,
                        help="train_size 阈值，≥ 此值视为数据充足（触发缩放）")
    parser.add_argument("--adaptive_f1_thresh", type=float, default=0.85,
                        help="BERT baseline val F1 阈值，≥ 此值视为基线已强（触发缩放）")
    parser.add_argument("--adaptive_scale", type=float, default=0.1,
                        help="触发后 λ_bias 与 λ_consist 的乘数（默认 0.1，即缩小到 1/10）")
    
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[Surface] device={device}, seed={args.seed}, surface_dim={args.surface_feat_dim}")

    if args.use_weibo21:
        bert_name = "bert-base-chinese"
    else:
        bert_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    if args.use_weibo21:
        from utils.weibo21_dataloader import load_weibo21_dataset
        train_data, val_data, test_data = load_weibo21_dataset()
    elif args.use_liar:
        from utils.liar_dataloader import load_liar_dataset
        train_data, val_data, test_data = load_liar_dataset()
    else:
        train_data, val_data, test_data = load_dataset(
            dataset_name=args.dataset, seed=args.seed,
        )

    extractor = None
    if args.surface_feat_dim > 0:
        print("[Surface] 加载 SurfaceFeatureExtractor...")
        extractor = SurfaceFeatureExtractor(dim=args.surface_feat_dim)

    train_set = SurfaceAugmentedDataset(
        train_data, tokenizer, args.max_length,
        surface_extractor=extractor,
    )
    normalizer = (train_set.feat_mean, train_set.feat_std) if hasattr(train_set, "feat_mean") and train_set.feat_mean is not None else None
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

    contrast_loader = None
    if args.use_contrastive:
        if not args.adv_pkl or not args.orig_pkl:
            raise ValueError("--use_contrastive 需要同时指定 --orig_pkl 和 --adv_pkl")
        from utils.contrastive_dataloader import ContrastiveFakeNewsDataset
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

    model = SocialDebiasModel(
        model_name=bert_name, num_classes=2,
        hidden_dim=384, dropout=0.1, grl_lambda=1.0,
        use_frozen_bert=True,
        surface_feat_dim=args.surface_feat_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    lang = "zh" if args.use_weibo21 else args.language
    dataset_tag = "weibo21" if args.use_weibo21 else (
        "liar" if args.use_liar else args.dataset)
    save_path = (f"{args.save_dir}/socialdebias_{dataset_tag}_{lang}"
                 f"_seed{args.seed}_{args.save_suffix}.pt")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_test = None
    history = []
    args.adaptive_triggered = False
    args.adaptive_trigger_reasons = []
    args.adaptive_orig_lambda_bias = args.lambda_bias
    args.adaptive_orig_lambda_consist = args.lambda_consist
    args.adaptive_baseline_val_f1 = None

    if args.adaptive_lambda:
        train_size = len(train_set)
        triggered_by_size = train_size >= args.adaptive_size_thresh
        if triggered_by_size:
            args.adaptive_trigger_reasons.append(f"train_size={train_size}>={args.adaptive_size_thresh}")

        triggered_by_f1 = False
        candidate_paths = [
            f"{args.save_dir}/baseline_{dataset_tag}_{lang}_seed{args.seed}_history.json",
            f"{args.save_dir}/socialdebias_{dataset_tag}_{lang}_seed{args.seed}_bert_history.json",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        bert_hist = json.load(f)
                    val_f1 = bert_hist.get("best_test", {}).get("f1") or bert_hist.get("best_val_f1")
                    if val_f1 is not None:
                        args.adaptive_baseline_val_f1 = float(val_f1)
                        if args.adaptive_baseline_val_f1 >= args.adaptive_f1_thresh:
                            triggered_by_f1 = True
                            args.adaptive_trigger_reasons.append(
                                f"baseline_val_f1={args.adaptive_baseline_val_f1:.4f}>={args.adaptive_f1_thresh}")
                        break
                except Exception as e:
                    print(f"[Adaptive] 读取 BERT history 失败 ({p}): {e}")

        args.adaptive_triggered = triggered_by_size or triggered_by_f1

        if args.adaptive_triggered:
            args.lambda_bias *= args.adaptive_scale
            args.lambda_consist *= args.adaptive_scale
            print(f"[Adaptive] 触发缩放 ({'; '.join(args.adaptive_trigger_reasons)})")
            print(f"  lambda_bias:    {args.adaptive_orig_lambda_bias} -> {args.lambda_bias}")
            print(f"  lambda_consist: {args.adaptive_orig_lambda_consist} -> {args.lambda_consist}")
        else:
            print(f"[Adaptive] 未触发：train_size={train_size}, "
                  f"baseline_val_f1={args.adaptive_baseline_val_f1}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        ll = {"total": [], "fact": [], "bias": [], "consist": [], "contrast": [], "fact_soft": []}

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

        if contrast_loader is not None and args.lambda_contrast > 0:
            for batch in contrast_loader:
                optimizer.zero_grad()
                orig_ids = batch["orig_input_ids"].to(device)
                orig_mask = batch["orig_attention_mask"].to(device)
                adv_ids = batch["adv_input_ids"].to(device)
                adv_mask = batch["adv_attention_mask"].to(device)
                out_orig = model(orig_ids, orig_mask, surface_feat=None)
                out_adv = model(adv_ids, adv_mask, surface_feat=None)
                if args.use_soft_labels and "p_entail" in batch:
                    weights = batch["p_entail"].to(device)
                    weights = torch.clamp(weights, min=args.alpha_floor)
                    loss_contrast = info_nce_loss_weighted(
                        out_orig["fact_repr"], out_adv["fact_repr"],
                        weights=weights,
                        temperature=args.temperature,
                    )
                else:
                    loss_contrast = info_nce_loss(
                        out_orig["fact_repr"], out_adv["fact_repr"],
                        temperature=args.temperature,
                    )
                loss = args.lambda_contrast * loss_contrast

                if args.lambda_fact_soft > 0 and "p_entail" in batch:
                    labels_adv = batch["label"].to(device)
                    alpha = torch.clamp(batch["p_entail"].to(device),
                                        min=args.soft_label_floor)
                    n_classes = out_adv["fact_logits"].size(1)
                    y_hard = F.one_hot(labels_adv, num_classes=n_classes).float()
                    y_soft = alpha.unsqueeze(1) * y_hard + \
                             (1 - alpha.unsqueeze(1)) * (1.0 / n_classes)
                    log_probs = F.log_softmax(out_adv["fact_logits"], dim=1)
                    loss_fact_soft = -(y_soft * log_probs).sum(dim=1).mean()
                    loss = loss + args.lambda_fact_soft * loss_fact_soft
                    ll["fact_soft"].append(loss_fact_soft.item())

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
              f"fact={means['fact']:.4f} fact_soft={means['fact_soft']:.4f} "
              f"bias={means['bias']:.4f} consist={means['consist']:.4f} "
              f"contrast={means['contrast']:.4f}")
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
            print(f"        best test acc={test_m['acc']:.4f} f1={test_m['f1']:.4f}")

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
