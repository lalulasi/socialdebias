#!/usr/bin/env python
"""
scripts/analyze_human_eval.py

分析人工标注结果，输出以下指标：
1. 人类判别准确率（原文 / 对抗文 / 攻击成功率）
2. 置信度分布
3. 人类关键词解释一致性（原文 vs 对抗 Jaccard）
4. （可选）模型 IG 归因 vs 人类标注 Jaccard 对齐

用法：
    # 只计算人工标注指标
    python scripts/analyze_human_eval.py \
        --input results/human_eval/politifact_pre_annotated_task.xlsx \
        --output_dir results/human_eval/

    # 同时计算模型 IG 与人工关键词的对齐度
    python scripts/analyze_human_eval.py \
        --input results/human_eval/politifact_pre_annotated_task.xlsx \
        --ig_json results/explanation/politifact_surface_seed42.json \
        --output_dir results/human_eval/
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# 关键词分隔符：英文逗号 / 中文逗号 / 管道符 / 分号
KEYWORD_SEP = re.compile(r"[,，|;；]")


def parse_keywords(s):
    """解析人类关键词字符串，返回 lower-case 去重集合。"""
    if pd.isna(s) or not str(s).strip():
        return set()
    parts = KEYWORD_SEP.split(str(s))
    return {p.strip().lower() for p in parts if p.strip()}


def normalize_judgment(s):
    """规范化人类判断。Real/Fake/Uncertain → real/fake/uncertain。"""
    if pd.isna(s):
        return "missing"
    s = str(s).strip().lower()
    if s in ("real", "true", "0"):
        return "real"
    if s in ("fake", "false", "1"):
        return "fake"
    if s in ("uncertain", "unsure", "unknown"):
        return "uncertain"
    return s


def score_judgment(judgment, label, uncertain_score=0.5):
    """计算单次判断的得分。
    
    判对 → 1.0
    判错 → 0.0
    Uncertain → uncertain_score（默认 0.5，可改为 0.0 或 1.0）
    """
    if judgment == "uncertain":
        return uncertain_score
    return 1.0 if judgment == label else 0.0


def jaccard(a, b):
    """两个集合的 Jaccard 相似度。任一为空返回 nan。"""
    if not a or not b:
        return np.nan
    return len(a & b) / len(a | b)


def load_human_eval(path, key_path=None):
    if str(path).endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if key_path:
        if str(key_path).endswith((".xlsx", ".xlsm")):
            key = pd.read_excel(key_path)
        else:
            key = pd.read_csv(key_path)
        if "blind_id" not in df.columns or "blind_id" not in key.columns:
            raise ValueError("Blind task and answer key must both contain blind_id")
        df = df.merge(key, on="blind_id", how="left", validate="one_to_one")
    required = {"id", "label", "text_type", "human_keywords", "human_judgment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def reshape_to_pairs(df, uncertain_score=0.5):
    """长表（一条 sample 两行）→ 宽表（每 sample 一行）。"""
    df = df.copy()
    df["human_judgment_norm"] = df["human_judgment"].apply(normalize_judgment)
    df["keywords_set"] = df["human_keywords"].apply(parse_keywords)
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    orig = df[df["text_type"] == "original"].set_index("id")
    adv = df[df["text_type"] == "adv_C"].set_index("id")

    common_ids = sorted(set(orig.index) & set(adv.index))
    rows = []
    for sid in common_ids:
        o = orig.loc[sid]
        a = adv.loc[sid]
        label = str(o["label"]).strip().lower()
        rows.append(
            {
                "id": sid,
                "label": label,
                "orig_judgment": o["human_judgment_norm"],
                "orig_score": score_judgment(
                    o["human_judgment_norm"], label, uncertain_score
                ),
                "orig_confidence": o.get("confidence", np.nan),
                "orig_keywords": o["keywords_set"],
                "adv_judgment": a["human_judgment_norm"],
                "adv_score": score_judgment(
                    a["human_judgment_norm"], label, uncertain_score
                ),
                "adv_confidence": a.get("confidence", np.nan),
                "adv_keywords": a["keywords_set"],
            }
        )
    pairs = pd.DataFrame(rows)
    # ASR：原文得满分（1.0）但对抗未得满分（<1.0）
    pairs["attack_succeeded"] = (
        (pairs["orig_score"] >= 1.0) & (pairs["adv_score"] < 1.0)
    ).astype(int)
    pairs["jaccard_orig_adv"] = pairs.apply(
        lambda r: jaccard(r["orig_keywords"], r["adv_keywords"]), axis=1
    )
    return pairs


def compute_human_metrics(pairs, uncertain_score=0.5):
    """聚合人类标注指标。"""
    n = len(pairs)
    n_real = int((pairs["label"] == "real").sum())
    n_fake = int((pairs["label"] == "fake").sum())
    n_uncertain_orig = int((pairs["orig_judgment"] == "uncertain").sum())
    n_uncertain_adv = int((pairs["adv_judgment"] == "uncertain").sum())

    orig_acc = pairs["orig_score"].mean()
    adv_acc = pairs["adv_score"].mean()

    fully_correct_orig = pairs[pairs["orig_score"] >= 1.0]
    asr_human = (
        pairs["attack_succeeded"].sum() / len(fully_correct_orig)
        if len(fully_correct_orig) > 0
        else np.nan
    )

    orig_acc_real = pairs[pairs["label"] == "real"]["orig_score"].mean()
    orig_acc_fake = pairs[pairs["label"] == "fake"]["orig_score"].mean()
    adv_acc_real = pairs[pairs["label"] == "real"]["adv_score"].mean()
    adv_acc_fake = pairs[pairs["label"] == "fake"]["adv_score"].mean()

    jacc_mean = pairs["jaccard_orig_adv"].mean()
    jacc_std = pairs["jaccard_orig_adv"].std()
    jacc_valid = pairs["jaccard_orig_adv"].notna().sum()

    return {
        "n_samples": n,
        "n_real": n_real,
        "n_fake": n_fake,
        "n_uncertain_orig": n_uncertain_orig,
        "n_uncertain_adv": n_uncertain_adv,
        "uncertain_score": uncertain_score,
        "human_acc_clean": orig_acc,
        "human_acc_clean_real": orig_acc_real,
        "human_acc_clean_fake": orig_acc_fake,
        "human_acc_adv": adv_acc,
        "human_acc_adv_real": adv_acc_real,
        "human_acc_adv_fake": adv_acc_fake,
        "human_acc_drop": orig_acc - adv_acc,
        "human_asr": asr_human,
        "jaccard_orig_adv_mean": jacc_mean,
        "jaccard_orig_adv_std": jacc_std,
        "jaccard_orig_adv_valid_n": jacc_valid,
        "orig_confidence_mean": pairs["orig_confidence"].mean(),
        "adv_confidence_mean": pairs["adv_confidence"].mean(),
    }


def load_ig_topk(path, topk=10):
    """加载 IG 归因 JSON，提取每个 sample 的原文 / 对抗 Top-K 词。
    
    期望格式（run_explanation_metrics.py 输出）：
        {
          "samples": [
            {"id": "pf_test_000", "orig_topk_tokens": [...], "adv_topk_tokens": [...]},
            ...
          ]
        }
    若实际格式不同，根据 utils/explanation_metrics.py 调整解析。
    """
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    elif isinstance(data, dict) and "rows" in data:
        samples = data["rows"]
    elif isinstance(data, list):
        samples = data
    else:
        print(f"Warning: unknown IG JSON structure, skipping model alignment")
        return None

    out = {}
    for s in samples:
        sid = s.get("id") or s.get("sample_id")
        if not sid:
            continue
        orig = s.get("orig_topk_tokens") or s.get("orig_top_tokens") or []
        adv = s.get("adv_topk_tokens") or s.get("adv_top_tokens") or []
        out[sid] = {
            "orig": {t.lower().strip("##") for t in orig[:topk] if t},
            "adv": {t.lower().strip("##") for t in adv[:topk] if t},
        }
    return out


def compute_model_alignment(pairs, ig_data):
    """模型 IG Top-K 词 vs 人类关键词 Jaccard 对齐。"""
    rows = []
    for _, r in pairs.iterrows():
        sid = r["id"]
        if sid not in ig_data:
            continue
        model_orig = ig_data[sid]["orig"]
        model_adv = ig_data[sid]["adv"]
        rows.append(
            {
                "id": sid,
                "jaccard_human_model_orig": jaccard(r["orig_keywords"], model_orig),
                "jaccard_human_model_adv": jaccard(r["adv_keywords"], model_adv),
            }
        )
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    metrics = {
        "n_aligned": len(df),
        "jaccard_human_model_orig_mean": df["jaccard_human_model_orig"].mean(),
        "jaccard_human_model_orig_std": df["jaccard_human_model_orig"].std(),
        "jaccard_human_model_adv_mean": df["jaccard_human_model_adv"].mean(),
        "jaccard_human_model_adv_std": df["jaccard_human_model_adv"].std(),
    }
    return df, metrics


def format_pct(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.{digits}f}%"


def format_float(x, digits=4):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"


def print_report(metrics, alignment_metrics=None):
    print("=" * 60)
    print("人工评估分析报告（PolitiFact × adv_C）")
    print("=" * 60)
    print()
    print("【数据完整性】")
    print(f"  样本对数：           {metrics['n_samples']}")
    print(
        f"  标签分布：           real={metrics['n_real']} / "
        f"fake={metrics['n_fake']}"
    )
    print(
        f"  Uncertain 数量：     原文 {metrics['n_uncertain_orig']} / "
        f"对抗 {metrics['n_uncertain_adv']}"
    )
    print(
        f"  Uncertain 计分方式： {metrics['uncertain_score']} 分（0=不对，0.5=半对，1=判对）"
    )
    print()
    print("【人类判别准确率】")
    print(f"  原文 acc：           {format_pct(metrics['human_acc_clean'])}")
    print(
        f"    real 子集：        {format_pct(metrics['human_acc_clean_real'])}"
    )
    print(
        f"    fake 子集：        {format_pct(metrics['human_acc_clean_fake'])}"
    )
    print(f"  对抗 acc：           {format_pct(metrics['human_acc_adv'])}")
    print(
        f"    real 子集：        {format_pct(metrics['human_acc_adv_real'])}"
    )
    print(
        f"    fake 子集：        {format_pct(metrics['human_acc_adv_fake'])}"
    )
    print(
        f"  Acc Drop：           {format_pct(metrics['human_acc_drop'])} (pp)"
    )
    print(f"  ASR（攻击成功率）：  {format_pct(metrics['human_asr'])}")
    print()
    print("【置信度】")
    print(
        f"  原文均值：           {format_float(metrics['orig_confidence_mean'], 2)} / 5"
    )
    print(
        f"  对抗均值：           {format_float(metrics['adv_confidence_mean'], 2)} / 5"
    )
    print()
    print("【人类关键词解释一致性（原文 vs 对抗 Jaccard）】")
    print(
        f"  mean ± std：         {format_float(metrics['jaccard_orig_adv_mean'])} ± "
        f"{format_float(metrics['jaccard_orig_adv_std'])}"
    )
    print(
        f"  有效样本数：         {int(metrics['jaccard_orig_adv_valid_n'])} / "
        f"{metrics['n_samples']}"
    )
    print()
    if alignment_metrics:
        print("【模型 IG 归因 vs 人类关键词对齐（Jaccard）】")
        print(f"  对齐样本数：         {alignment_metrics['n_aligned']}")
        print(
            f"  原文上 Jaccard：     "
            f"{format_float(alignment_metrics['jaccard_human_model_orig_mean'])} ± "
            f"{format_float(alignment_metrics['jaccard_human_model_orig_std'])}"
        )
        print(
            f"  对抗上 Jaccard：     "
            f"{format_float(alignment_metrics['jaccard_human_model_adv_mean'])} ± "
            f"{format_float(alignment_metrics['jaccard_human_model_adv_std'])}"
        )
        print()


def save_outputs(pairs, metrics, output_dir, alignment_df=None, alignment_metrics=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_out = pairs.copy()
    pairs_out["orig_keywords"] = pairs_out["orig_keywords"].apply(
        lambda s: ",".join(sorted(s)) if s else ""
    )
    pairs_out["adv_keywords"] = pairs_out["adv_keywords"].apply(
        lambda s: ",".join(sorted(s)) if s else ""
    )
    per_sample_path = output_dir / "human_eval_per_sample.csv"
    pairs_out.to_csv(per_sample_path, index=False, encoding="utf-8-sig")
    print(f"逐样本明细 → {per_sample_path}")

    metrics_path = output_dir / "human_eval_metrics.json"
    out_dict = {"human_metrics": metrics}
    if alignment_metrics:
        out_dict["model_alignment"] = alignment_metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2, default=float)
    print(f"汇总指标 → {metrics_path}")

    if alignment_df is not None:
        align_path = output_dir / "human_eval_model_alignment.csv"
        alignment_df.to_csv(align_path, index=False, encoding="utf-8-sig")
        print(f"模型对齐明细 → {align_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="人工标注 xlsx 或 csv 文件",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="answer-key CSV for a task generated with --blind",
    )
    parser.add_argument(
        "--ig_json",
        type=str,
        default=None,
        help="（可选）IG 归因 JSON 文件，用于模型对齐",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="IG Top-K 词数（默认 10，对应论文 5.4 节）",
    )
    parser.add_argument(
        "--uncertain_score",
        type=float,
        default=0.5,
        help="Uncertain 判断的得分：0=不对，0.5=半对（默认），1=判对",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/human_eval/",
    )
    args = parser.parse_args()

    df = load_human_eval(args.input, args.key)
    print(f"加载：{args.input}，共 {len(df)} 行")

    pairs = reshape_to_pairs(df, uncertain_score=args.uncertain_score)
    print(f"重塑为 {len(pairs)} 个样本对")

    metrics = compute_human_metrics(pairs, uncertain_score=args.uncertain_score)

    alignment_df, alignment_metrics = None, None
    if args.ig_json:
        ig_data = load_ig_topk(args.ig_json, args.topk)
        if ig_data:
            alignment_df, alignment_metrics = compute_model_alignment(pairs, ig_data)

    print_report(metrics, alignment_metrics)
    save_outputs(pairs, metrics, args.output_dir, alignment_df, alignment_metrics)


if __name__ == "__main__":
    main()
