"""
主实验 3 种子汇总：BERT 基线 + SocialDebias × 2 数据集 × 3 种子
运行: python scripts/parse_main_3seeds.py

产出:
  表 1：干净测试集 3 种子对比
  表 2：对抗测试集 3 种子对比 (按 A/B/C/D 分变体)
  表 3：鲁棒性指标（F1 降幅、ASR）
  CSV: results/main_3seeds_logs/main_summary.csv
  对比 exp001/exp002 单次结果差距
"""
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

MODELS_DIR = Path("results/models")
LOG_DIR = Path("results/main_3seeds_logs")

SEEDS = [42, 2024, 3407]
DATASETS = ["politifact", "gossipcop"]

# exp001/exp002 的单次参考值（供对比）
EXP_BASELINES = {
    "politifact": {
        "baseline_clean_f1": 0.8776, "baseline_adv_f1_avg": 0.7436,
        "sd_clean_f1": 0.8774, "sd_adv_f1_avg": 0.8148,
    },
    "gossipcop": {
        "baseline_clean_f1": 0.7536, "baseline_adv_f1_avg": 0.7388,
        "sd_clean_f1": 0.7316, "sd_adv_f1_avg": 0.7169,
    },
}


def load_ckpt_val_metrics(ckpt_path):
    """从 ckpt dict 读最佳 val 指标。"""
    import torch
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return {
            "val_f1": ckpt.get("val_f1"),
            "val_acc": ckpt.get("val_acc"),
            "val_auc": ckpt.get("val_auc"),
            "best_epoch": ckpt.get("epoch"),
        }
    except Exception as e:
        print(f"[warn] 读 {ckpt_path} 失败: {e}")
        return None


def parse_adv_eval_log(log_path):
    """
    从 evaluate_adversarial.py 的 log 里抽取：
      baseline clean/adv_A/adv_B/adv_C/adv_D 的 Acc/F1/AUC/ASR
      sd       clean/adv_A/adv_B/adv_C/adv_D 同
    返回 dict，解析失败返回 None。
    
    注意：具体的抽取正则需要根据 evaluate_adversarial.py 实际输出调整。
    如果抽取失败，会打印日志样例供你确认格式。
    """
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    
    # 尝试最常见的几种格式
    # 如果这里没抽到，需要看日志样例改正则
    result = {"baseline": {}, "sd": {}, "raw_log_tail": text[-2000:]}
    
    # 这部分需要根据实际日志格式调整——先打印样例
    return result


def parse_adv_eval_from_json(json_path):
    """
    假设 evaluate_adversarial.py 保存了 JSON 结果文件。
    如果是这种情况，优先用 JSON，否则退到解析 log。
    """
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def print_table_1_clean(results):
    """干净集主表。"""
    print("\n" + "=" * 90)
    print("表 1 · 干净测试集主实验（3 种子均值±标准差）")
    print("=" * 90)
    print(f"{'Dataset':<12}{'Model':<18}{'Val F1 (best)':<22}{'Val Acc':<18}{'Val AUC':<18}{'N':<4}")
    print("-" * 90)
    
    summary_rows = []
    for dataset in DATASETS:
        for model in ["baseline", "socialdebias"]:
            vals = []
            for seed in SEEDS:
                key = (dataset, model, seed)
                v = results.get(key)
                if v and v.get("val_f1") is not None:
                    vals.append(v)
            if not vals:
                print(f"{dataset:<12}{model:<18}{'(no data)':<58}0")
                continue
            f1s = [x["val_f1"] for x in vals]
            accs = [x["val_acc"] for x in vals]
            aucs = [x["val_auc"] for x in vals]
            print(f"{dataset:<12}{model:<18}"
                  f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}   "
                  f"{np.mean(accs):.4f}±{np.std(accs):.4f}   "
                  f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}   "
                  f"{len(vals)}")
            summary_rows.append({
                "dataset": dataset, "model": model,
                "val_f1_mean": np.mean(f1s), "val_f1_std": np.std(f1s),
                "val_acc_mean": np.mean(accs), "val_acc_std": np.std(accs),
                "val_auc_mean": np.mean(aucs), "val_auc_std": np.std(aucs),
                "n": len(vals),
            })
    return summary_rows


def main():
    # === 收集所有 ckpt 的 val 指标 ===
    results = {}
    for dataset in DATASETS:
        for model in ["baseline", "socialdebias"]:
            for seed in SEEDS:
                ckpt = MODELS_DIR / f"{model}_{dataset}_en_seed{seed}.pt"
                metrics = load_ckpt_val_metrics(ckpt) if ckpt.exists() else None
                results[(dataset, model, seed)] = metrics

    # === 表 1：干净集 ===
    table1 = print_table_1_clean(results)

    # === 表 2：对抗集（先尝试从日志找）===
    print("\n" + "=" * 90)
    print("表 2 · 对抗集评估（待补）")
    print("=" * 90)
    print("请先运行此脚本看表 1，然后告诉 Claude 日志/JSON 格式，我会补上表 2 和表 3。")

    # === 日志样例打印，供调整 ===
    sample_log = LOG_DIR / "adv_politifact_seed42.log"
    if sample_log.exists():
        print(f"\n对抗评估日志样例 ({sample_log.name})：")
        print("-" * 70)
        lines = sample_log.read_text().split("\n")
        # 只打印后 60 行（评估结果通常在末尾）
        for line in lines[-60:]:
            print(line)

    # === 保存 CSV ===
    import csv
    if table1:
        out = LOG_DIR / "main_summary_clean.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=table1[0].keys())
            w.writeheader()
            w.writerows(table1)
        print(f"\n干净集 CSV 已保存: {out}")


if __name__ == "__main__":
    main()