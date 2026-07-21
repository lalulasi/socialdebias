"""Train SocialDebias ablation variants with the reproduction-guide CLI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from modeling.social_debias import SocialDebiasModel
from modeling.infonce import info_nce_loss, info_nce_loss_weighted
from utils.dataloader import SurfaceAugmentedDataset
from utils.device import get_device
from utils.real_dataloader import load_dataset
from utils.surface_features import SurfaceFeatureExtractor


VARIANT_LAMBDAS = {
    "full": (1.0, 0.5, 0.3),
    "no_grl": (1.0, 0.0, 0.3),
    "no_consist": (1.0, 0.5, 0.0),
    "no_both": (1.0, 0.0, 0.0),
}
PAIRED_FORWARD_VERSION = "skip_unused_orig_v1"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    preds, probs, labels_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            surface = batch.get("surface_feat")
            if surface is not None:
                surface = surface.to(device)

            out = model(input_ids, attention_mask, surface_feat=surface)
            logits = out["fact_logits"]
            preds.extend(logits.argmax(dim=-1).cpu().numpy())
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    try:
        auc = roc_auc_score(labels_all, probs)
    except ValueError:
        auc = float("nan")
    return {
        "acc": accuracy_score(labels_all, preds),
        "f1": f1_score(labels_all, preds, average="macro"),
        "auc": auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["politifact", "gossipcop"])
    parser.add_argument("--language", default="en")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lambda_fact", type=float, default=None)
    parser.add_argument("--lambda_bias", type=float, default=None)
    parser.add_argument("--lambda_consist", type=float, default=None)
    parser.add_argument("--surface_feat_dim", type=int, default=8)
    parser.add_argument("--surface_lexicon_path", default=None)
    parser.add_argument("--surface_stopwords_path", default=None)
    parser.add_argument("--surface_feature_version", default="nrc_emolex_v1",
                        choices=["nrc_emolex_v1", "legacy_seed_v0"])
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--lambda_contrast", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--use_soft_labels", action="store_true")
    parser.add_argument("--alpha_floor", type=float, default=0.5)
    parser.add_argument("--lambda_fact_soft", type=float, default=0.0)
    parser.add_argument("--soft_label_floor", type=float, default=0.5)
    parser.add_argument("--orig_pkl", default=None)
    parser.add_argument("--adv_pkl", default=None)
    parser.add_argument("--save_dir", default="results/models")
    parser.add_argument("--save_suffix", default=None)
    parser.add_argument("--variant", choices=list(VARIANT_LAMBDAS), default=None,
                        help="旧入口兼容；未显式传 lambda 时用该变体的默认权重")
    args = parser.parse_args()

    if args.variant and args.lambda_fact is None:
        args.lambda_fact, args.lambda_bias, args.lambda_consist = VARIANT_LAMBDAS[args.variant]
    args.lambda_fact = 1.0 if args.lambda_fact is None else args.lambda_fact
    args.lambda_bias = 0.5 if args.lambda_bias is None else args.lambda_bias
    args.lambda_consist = 0.3 if args.lambda_consist is None else args.lambda_consist
    if args.save_suffix is None:
        args.save_suffix = f"abl_{args.variant or 'custom'}"
    # Persist implementation provenance in both history and checkpoint config.
    args.paired_forward_version = PAIRED_FORWARD_VERSION

    set_seed(args.seed)
    device = get_device()
    bert_name = "bert-base-uncased" if args.language == "en" else "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    train_data, val_data, test_data = load_dataset(args.dataset, seed=args.seed)
    extractor = SurfaceFeatureExtractor(
        dim=args.surface_feat_dim,
        lexicon_path=args.surface_lexicon_path,
        language=args.language,
        feature_version=args.surface_feature_version,
        stopwords_path=args.surface_stopwords_path,
    ) if args.surface_feat_dim > 0 else None

    train_set = SurfaceAugmentedDataset(
        train_data, tokenizer, args.max_length, surface_extractor=extractor
    )
    normalizer = (train_set.feat_mean, train_set.feat_std) if extractor else None
    val_set = SurfaceAugmentedDataset(
        val_data, tokenizer, args.max_length, surface_extractor=extractor, normalizer=normalizer
    )
    test_set = SurfaceAugmentedDataset(
        test_data, tokenizer, args.max_length, surface_extractor=extractor, normalizer=normalizer
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    paired_loader = None
    needs_paired_data = (
        (args.use_contrastive and args.lambda_contrast > 0)
        or args.lambda_fact_soft > 0
    )
    if needs_paired_data:
        if not args.orig_pkl or not args.adv_pkl:
            raise ValueError(
                "InfoNCE 或 NLI 分类软标签需要 --orig_pkl 与 --adv_pkl"
            )
        from utils.contrastive_dataloader import ContrastiveFakeNewsDataset
        paired_set = ContrastiveFakeNewsDataset.from_pkl(
            args.orig_pkl, args.adv_pkl, tokenizer, args.max_length
        )
        if (args.use_soft_labels or args.lambda_fact_soft > 0) and not paired_set.has_p_entail:
            raise ValueError("NLI 软标签已启用，但 --adv_pkl 不包含 p_entail")
        paired_loader = DataLoader(
            paired_set, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

    model = SocialDebiasModel(
        model_name=bert_name,
        num_classes=2,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        dropout=0.1,
        grl_lambda=1.0,
        use_frozen_bert=True,
        surface_feat_dim=args.surface_feat_dim,
    ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.save_dir) / (
        f"socialdebias_{args.dataset}_{args.language}_seed{args.seed}_{args.save_suffix}.pt"
    )

    best_val_f1 = -1.0
    best_test = None
    history = []
    print(f"[Ablation] dataset={args.dataset} seed={args.seed} suffix={args.save_suffix}")
    print(f"[Ablation] lambdas fact={args.lambda_fact} bias={args.lambda_bias} consist={args.lambda_consist}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = {
            "total": [], "fact": [], "bias": [], "consist": [],
            "contrast": [], "fact_soft": [],
        }

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            surface = batch.get("surface_feat")
            if surface is not None:
                surface = surface.to(device)

            out = model(input_ids, attention_mask, surface_feat=surface)
            loss_fact = criterion(out["fact_logits"], labels)
            loss_bias = criterion(out["bias_logits"], labels) if args.lambda_bias > 0 else torch.tensor(0.0, device=device)
            if args.lambda_consist > 0 and "frozen_repr" in out:
                cos = torch.nn.functional.cosine_similarity(out["shared_repr"], out["frozen_repr"], dim=-1)
                loss_consist = (1.0 - cos).mean()
            else:
                loss_consist = torch.tensor(0.0, device=device)

            loss = (
                args.lambda_fact * loss_fact
                + args.lambda_bias * loss_bias
                + args.lambda_consist * loss_consist
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses["total"].append(loss.item())
            losses["fact"].append(loss_fact.item())
            losses["bias"].append(loss_bias.item())
            losses["consist"].append(loss_consist.item())

        if paired_loader is not None:
            for batch in paired_loader:
                optimizer.zero_grad()
                adv_ids = batch["adv_input_ids"].to(device)
                adv_mask = batch["adv_attention_mask"].to(device)
                needs_orig_forward = args.use_contrastive and args.lambda_contrast > 0
                if needs_orig_forward:
                    orig_ids = batch["orig_input_ids"].to(device)
                    orig_mask = batch["orig_attention_mask"].to(device)
                    out_orig = model(orig_ids, orig_mask)
                else:
                    # Classification-only NLI loss never reads the original
                    # representation.  Avoid retaining an unused full BERT
                    # graph while forwarding the adversarial view.
                    out_orig = None
                out_adv = model(adv_ids, adv_mask)
                pair_loss = torch.tensor(0.0, device=device)

                if needs_orig_forward:
                    if args.use_soft_labels:
                        weights = torch.clamp(
                            batch["p_entail"].to(device), min=args.alpha_floor
                        )
                        loss_contrast = info_nce_loss_weighted(
                            out_orig["fact_repr"], out_adv["fact_repr"], weights,
                            temperature=args.temperature,
                        )
                    else:
                        loss_contrast = info_nce_loss(
                            out_orig["fact_repr"], out_adv["fact_repr"],
                            temperature=args.temperature,
                        )
                    pair_loss = pair_loss + args.lambda_contrast * loss_contrast
                    losses["contrast"].append(loss_contrast.item())

                if args.lambda_fact_soft > 0:
                    labels_adv = batch["label"].to(device)
                    alpha = torch.clamp(
                        batch["p_entail"].to(device), min=args.soft_label_floor
                    )
                    n_classes = out_adv["fact_logits"].size(1)
                    y_hard = F.one_hot(labels_adv, num_classes=n_classes).float()
                    y_soft = (
                        alpha.unsqueeze(1) * y_hard
                        + (1 - alpha.unsqueeze(1)) / n_classes
                    )
                    loss_fact_soft = -(
                        y_soft * F.log_softmax(out_adv["fact_logits"], dim=1)
                    ).sum(dim=1).mean()
                    pair_loss = pair_loss + args.lambda_fact_soft * loss_fact_soft
                    losses["fact_soft"].append(loss_fact_soft.item())

                pair_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses["total"].append(pair_loss.item())

        val_m = evaluate(model, val_loader, device)
        means = {k: float(np.mean(v)) if v else 0.0 for k, v in losses.items()}
        history.append({
            "epoch": epoch,
            "epoch_time": time.time() - t0,
            **{f"loss_{k}": v for k, v in means.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        })
        print(f"[E{epoch}] total={means['total']:.4f} fact={means['fact']:.4f} "
              f"bias={means['bias']:.4f} consist={means['consist']:.4f} "
              f"contrast={means['contrast']:.4f} fact_soft={means['fact_soft']:.4f}")
        print(f"       val acc={val_m['acc']:.4f} f1={val_m['f1']:.4f} auc={val_m['auc']:.4f}")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_test = {"epoch": epoch, **evaluate(model, test_loader, device)}
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1,
                "feat_mean": train_set.feat_mean.tolist() if train_set.feat_mean is not None else None,
                "feat_std": train_set.feat_std.tolist() if train_set.feat_std is not None else None,
                "config": vars(args),
            }, ckpt_path)
            print(f"       best test acc={best_test['acc']:.4f} f1={best_test['f1']:.4f}")

    hist_path = ckpt_path.with_name(ckpt_path.stem + "_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "best_val_f1": best_val_f1,
            "best_test": best_test,
            "history": history,
        }, f, indent=2, default=str)
    print(f"\n[Ablation] ckpt: {ckpt_path}")
    print(f"[Ablation] history: {hist_path}")


if __name__ == "__main__":
    main()
