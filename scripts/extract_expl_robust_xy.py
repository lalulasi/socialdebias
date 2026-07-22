"""逐样本计算归因一致性与对抗改写前后的预测变化。"""
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.social_debias import SocialDebiasModel, infer_bottleneck_dim
from modeling.attributor import BertAttributor
from utils.explanation_metrics import compute_all_metrics
from utils.surface_features import SurfaceFeatureExtractor
from utils.device import get_device


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict):
        texts = d.get("news") or d.get("rewritten") or d.get("content")
        labels = d.get("labels") or d.get("label")
    else:
        texts = d["news"].tolist() if "news" in d.columns else d["content"].tolist()
        labels = d["labels"].tolist() if "labels" in d.columns else d["label"].tolist()
    return list(texts), [int(x) for x in labels]


@torch.no_grad()
def predict_p_true(model, tokenizer, extractor, feat_mean, feat_std,
                   text, label, device, max_length=512):
    enc = tokenizer(text, max_length=max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    surface = None
    if extractor is not None:
        raw = extractor.extract(text)
        std = (raw - feat_mean) / (feat_std + 1e-8)
        surface = torch.tensor(std, dtype=torch.float32).unsqueeze(0).to(device)
    out = model(input_ids, attention_mask, surface_feat=surface)
    probs = torch.softmax(out["fact_logits"], dim=-1)[0]
    return float(probs[label].item()), int(out["fact_logits"].argmax(dim=-1).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_pkl", default="data/sheepdog/news_articles/politifact_test.pkl")
    ap.add_argument("--adv_tpl",  default="data/sheepdog/adversarial_test/politifact_test_adv_{v}.pkl",
                    help="改写 pkl 路径模板，{v} 替换为变体名")
    ap.add_argument("--variants", default="A,B,C,D")
    ap.add_argument("--ckpt",     default="results/models/socialdebias_politifact_en_seed42_surface.pt")
    ap.add_argument("--output",   default="results/expl_robust_xy_politifact_ALL.csv")
    ap.add_argument(
        "--target_class", type=int, choices=(0, 1), default=None,
        help="固定归因类别；默认按每条样本的真实标签归因，与 P6 口径一致",
    )
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument(
        "--ig_internal_batch_size", type=int, default=4,
        help="Captum IG 内部分批大小；显存不足时降为 2 或 1",
    )
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--bert_name", default="bert-base-uncased")
    args = ap.parse_args()

    device = get_device()
    print(f"设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    state_dict = ckpt["model_state_dict"]
    surface_dim = config.get("surface_feat_dim", 0)
    # 事实分类与 IG 都不读取 surface_feat；无需在相关性评估时加载冻结语义锚点，
    # 否则会额外占用一整份 BERT 显存。
    feat_mean = None
    feat_std = None
    model = SocialDebiasModel(
        model_name=args.bert_name, num_classes=2,
        hidden_dim=config.get("hidden_dim", 384),
        bottleneck_dim=infer_bottleneck_dim(state_dict, config), dropout=0.1,
        grl_lambda=1.0, use_frozen_bert=False, surface_feat_dim=surface_dim,
    ).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith("frozen_bert.")
    ]
    if incompatible.missing_keys or unexpected:
        raise ValueError(
            "checkpoint 架构不匹配: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    model.eval()
    print(f"加载 ckpt: surface_dim={surface_dim}")
    # fact_logits 不读取 surface_feat；表层特征只通过训练期偏置梯度生效。
    extractor = None
    if args.ig_internal_batch_size <= 0:
        raise ValueError("--ig_internal_batch_size must be positive")
    attributor = BertAttributor(
        model,
        tokenizer,
        device,
        n_steps=args.n_steps,
        internal_batch_size=args.ig_internal_batch_size,
    )

    # ---- 原文侧只算一次 ----
    orig_texts, orig_labels = load_pkl(args.orig_pkl)
    n = len(orig_texts)
    print(f"原文样本数 n={n}，预计算原文归因与置信度 …")
    orig_attr, orig_p, orig_pred = {}, {}, {}
    for i in range(n):
        attr_target = orig_labels[i] if args.target_class is None else args.target_class
        try:
            a = attributor.attribute(
                orig_texts[i],
                target_class=attr_target,
                max_length=args.max_length,
            )
            orig_attr[i] = (a["tokens"], a["scores"])
        except Exception as e:
            print(f"  [原文归因失败 idx={i}] {e}"); orig_attr[i] = None
        p, pr = predict_p_true(model, tokenizer, extractor, feat_mean, feat_std,
                               orig_texts[i], orig_labels[i], device, args.max_length)
        orig_p[i], orig_pred[i] = p, pr
        if (i + 1) % 20 == 0: print(f"    原文 {i+1}/{n}")

    # ---- 逐变体 ----
    variants = [v.strip() for v in args.variants.split(",")]
    rows = []
    for v in variants:
        adv_path = args.adv_tpl.format(v=v)
        adv_texts, adv_labels = load_pkl(adv_path)
        m = min(n, len(adv_texts))
        if orig_labels[:m] != adv_labels[:m]:
            raise ValueError(f"原文与变体 {v} 的标签顺序不一致")
        print(f"\n=== 变体 {v}（{adv_path}，{m} 样本）===")
        for i in range(m):
            label = orig_labels[i]
            attr_target = label if args.target_class is None else args.target_class
            # X: 一致性
            if orig_attr[i] is not None:
                try:
                    a_adv = attributor.attribute(
                        adv_texts[i],
                        target_class=attr_target,
                        max_length=args.max_length,
                    )
                    mm = compute_all_metrics(orig_attr[i][0], orig_attr[i][1],
                                             a_adv["tokens"], a_adv["scores"], k=args.top_k)
                except Exception as e:
                    print(f"  [变体{v}归因失败 idx={i}] {e}")
                    mm = {"top_k_overlap": np.nan, "spearman": np.nan, "js_divergence": np.nan}
            else:
                mm = {"top_k_overlap": np.nan, "spearman": np.nan, "js_divergence": np.nan}
            # Y: Δp
            p_a, pred_a = predict_p_true(model, tokenizer, extractor, feat_mean, feat_std,
                                         adv_texts[i], label, device, args.max_length)
            rows.append({
                "variant": v, "idx": i, "label": label,
                "attribution_target": attr_target,
                "sd_top_k_overlap": mm["top_k_overlap"],
                "sd_spearman": mm["spearman"], "sd_js_div": mm["js_divergence"],
                "p_true_orig": orig_p[i], "p_true_adv": p_a,
                "delta_p": orig_p[i] - p_a,
                "pred_orig": orig_pred[i], "pred_adv": pred_a,
                "correct_orig": int(orig_pred[i] == label),
                "correct_adv": int(pred_a == label),
            })
            if (i + 1) % 20 == 0: print(f"    {v} {i+1}/{m}")
            if len(rows) % 30 == 0 or (v == variants[-1] and i == m - 1):
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)

    # ---- 汇总 ----
    def stat(rs, k):
        v = [r[k] for r in rs if not (isinstance(r[k], float) and np.isnan(r[k]))]
        return (np.mean(v), np.std(v)) if v else (float("nan"), float("nan"))

    print("\n" + "=" * 70)
    print(f"{'变体':<8}{'n':<5}{'top_k':<10}{'spearman':<10}{'Δp均值':<10}{'攻破/干净对':<12}{'ASR'}")
    for v in variants + ["ALL"]:
        rs = rows if v == "ALL" else [r for r in rows if r["variant"] == v]
        tk, _ = stat(rs, "sd_top_k_overlap"); sp, _ = stat(rs, "sd_spearman")
        dp, _ = stat(rs, "delta_p")
        clean_ok = sum(1 for r in rs if r["correct_orig"] == 1)
        flip = sum(1 for r in rs if r["correct_orig"] == 1 and r["correct_adv"] == 0)
        asr = 100 * flip / max(clean_ok, 1)
        print(f"{v:<8}{len(rs):<5}{tk:<10.4f}{sp:<10.4f}{dp:<10.4f}{f'{flip}/{clean_ok}':<12}{asr:.1f}%")
    print("=" * 70)
    print(f"\n逐样本×变体结果已保存: {args.output}")
    print("请将 adv_C 同 seed 的结果与 P6 JSON 对照，作为归因口径自检。")


if __name__ == "__main__":
    main()
