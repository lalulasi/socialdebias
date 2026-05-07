"""
解释一致性批量评估（论文 5.4 节）

输入: PolitiFact 测试集 (90 条) + 1 个对抗变体 (默认 adv_C, 90 条)
模型:
  1. 未微调 bert-base-uncased (零基线参考)
  2. SocialDebias surface ckpt (3 seed 平均，可选单 seed)
逻辑: 每条样本对 (原文, 对抗文) 各跑一次 IG → 计算三指标 → 全集汇总
输出: results/explanation_consistency_{variant}.csv
      + 终端打印 mean ± std 对比表

工作量预估 (4090): 90 条 × 2 模型 × 2 版本 = 360 次 IG, ~30-60 min
"""
import argparse
import csv
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from modeling.attributor import BertAttributor
from modeling.social_debias import SocialDebiasModel
from utils.explanation_metrics import compute_all_metrics

warnings.filterwarnings("ignore", category=UserWarning)


# ============== 模型加载 ==============
class BertOnlyWrapper(torch.nn.Module):
    """未微调 BERT 包装类，给 BertAttributor 用。
    内部带个随机初始化分类头（只为让 IG 有目标输出，不参与归因结果有意义性）。
    """
    def __init__(self, bert_name="bert-base-uncased", num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        # 用确定性 seed 初始化分类头，保证可重复
        torch.manual_seed(42)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


class SocialDebiasWrapper(torch.nn.Module):
    """SocialDebiasModel 包装：让 forward 输出 fact_logits，给 BertAttributor 用。"""
    def __init__(self, sd_model):
        super().__init__()
        self.sd = sd_model
        # BertAttributor 内部会查找 .bert 属性做归因，所以暴露主干
        self.bert = sd_model.bert

    def forward(self, input_ids, attention_mask, surface_feat=None):
        out = self.sd(input_ids, attention_mask, surface_feat=surface_feat)
        return out["fact_logits"]


def load_models(sd_ckpt_path, device, surface_feat_dim=8):
    print(f"[加载] 未微调 BERT (bert-base-uncased)")
    bert_model = BertOnlyWrapper("bert-base-uncased").to(device).eval()

    print(f"[加载] SocialDebias from {sd_ckpt_path}")
    sd_inner = SocialDebiasModel(
        model_name="bert-base-uncased",
        num_classes=2,
        surface_feat_dim=surface_feat_dim,
    ).to(device)
    state = torch.load(sd_ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    sd_inner.load_state_dict(state, strict=False)
    sd_inner.eval()
    sd_model = SocialDebiasWrapper(sd_inner).to(device).eval()
    return bert_model, sd_model


# ============== 数据加载 ==============
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


# ============== 主流程 ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_pkl",
                        default="data/sheepdog/news_articles/politifact_test.pkl")
    parser.add_argument("--adv_pkl",
                        default="data/sheepdog/adversarial_test/politifact_test_adv_C.pkl")
    parser.add_argument("--sd_ckpt",
                        default="results/models/socialdebias_politifact_en_seed42_surface.pt")
    parser.add_argument("--variant_name", default="adv_C")
    parser.add_argument("--output", default="results/explanation_consistency_adv_C.csv")
    parser.add_argument("--n_steps", type=int, default=50, help="IG 积分步数")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 = 全量；调试可设小数字")
    parser.add_argument("--target_class", type=int, default=1, help="对哪个类做归因")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
    orig_texts, orig_labels = load_pkl(args.orig_pkl)
    adv_texts, adv_labels = load_pkl(args.adv_pkl)
    if len(orig_texts) != len(adv_texts):
        raise RuntimeError(f"原文 {len(orig_texts)} ≠ 对抗 {len(adv_texts)}")

    if args.max_samples > 0:
        orig_texts = orig_texts[:args.max_samples]
        adv_texts = adv_texts[:args.max_samples]
        orig_labels = orig_labels[:args.max_samples]
    n = len(orig_texts)
    print(f"样本数: {n} 对")

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model, sd_model = load_models(args.sd_ckpt, device)

    bert_attr = BertAttributor(bert_model, tokenizer, device, n_steps=args.n_steps)
    sd_attr = BertAttributor(sd_model, tokenizer, device, n_steps=args.n_steps)

    # 主循环
    rows = []
    for i, (orig, adv, label) in enumerate(zip(orig_texts, adv_texts, orig_labels)):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"\n[{i+1}/{n}]  label={label}")

        # BERT (未微调)
        try:
            bert_orig = bert_attr.attribute(orig[:3000], target_class=args.target_class)
            bert_adv = bert_attr.attribute(adv[:3000], target_class=args.target_class)
            bert_m = compute_all_metrics(
                bert_orig["tokens"], bert_orig["scores"],
                bert_adv["tokens"], bert_adv["scores"],
                k=args.top_k,
            )
        except Exception as e:
            print(f"  [BERT err] {e}")
            bert_m = {"top_k_overlap": np.nan, "spearman": np.nan, "js_divergence": np.nan}

        # SocialDebias
        try:
            sd_orig = sd_attr.attribute(orig[:3000], target_class=args.target_class)
            sd_adv = sd_attr.attribute(adv[:3000], target_class=args.target_class)
            sd_m = compute_all_metrics(
                sd_orig["tokens"], sd_orig["scores"],
                sd_adv["tokens"], sd_adv["scores"],
                k=args.top_k,
            )
        except Exception as e:
            print(f"  [SD err] {e}")
            sd_m = {"top_k_overlap": np.nan, "spearman": np.nan, "js_divergence": np.nan}

        rows.append({
            "idx": i, "label": label,
            "bert_top_k_overlap": bert_m["top_k_overlap"],
            "bert_spearman": bert_m["spearman"],
            "bert_js_div": bert_m["js_divergence"],
            "sd_top_k_overlap": sd_m["top_k_overlap"],
            "sd_spearman": sd_m["spearman"],
            "sd_js_div": sd_m["js_divergence"],
        })

        # 增量保存（防中断）
        if (i + 1) % 10 == 0 or (i + 1) == n:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    # 汇总
    def stat(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    print("\n" + "=" * 80)
    print(f"PolitiFact × {args.variant_name} 解释一致性对比 (n={n}, k={args.top_k})")
    print("=" * 80)
    print(f"{'指标':<22}{'BERT(未微调)':<24}{'SocialDebias':<24}")
    for metric in [("top_k_overlap", "Top-K 重合度", True),
                   ("spearman", "Spearman", True),
                   ("js_div", "JS 散度", False)]:
        key, label, higher_better = metric
        b_m, b_s = stat(f"bert_{key}")
        s_m, s_s = stat(f"sd_{key}")
        arrow = "↑" if (higher_better and s_m > b_m) or (not higher_better and s_m < b_m) else "↓"
        print(f"{label:<22}{b_m:.4f} ± {b_s:.4f}    {s_m:.4f} ± {s_s:.4f}  {arrow}")
    print("=" * 80)
    print(f"\n结果 CSV: {args.output}")


if __name__ == "__main__":
    main()
