"""汇总 BERT 基线和 SocialDebias 的三种子对抗评测结果。"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean_std(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def load_result(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def add_run(groups, dataset, model, seed, result):
    metrics = result.get("results", {})
    for split, values in metrics.items():
        if split == "summary" or not isinstance(values, dict) or "f1" not in values:
            continue
        groups[(dataset, model, split)].append({
            "seed": seed,
            "accuracy": values.get("accuracy"),
            "f1": values.get("f1"),
            "auc": values.get("auc"),
            "asr": result.get("results", {}).get("summary", {}).get(
                "asr_per_variant", {}
            ).get(split, {}).get("asr"),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["politifact", "gossipcop"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407])
    parser.add_argument("--surface_suffix", default="surface")
    parser.add_argument("--bert_dir", type=Path, default=Path("results/bert_adv"))
    parser.add_argument("--surface_dir", type=Path, default=Path("results/surface_adv"))
    parser.add_argument("--output", type=Path, default=Path("results/main_3seeds_logs/main_summary.csv"))
    args = parser.parse_args()

    groups = defaultdict(list)
    missing = []
    for dataset in args.datasets:
        for seed in args.seeds:
            bert_path = args.bert_dir / f"bert_adv_{dataset}_seed{seed}.json"
            surface_path = args.surface_dir / (
                f"surface_adv_{dataset}_seed{seed}_{args.surface_suffix}.json"
            )
            if bert_path.exists():
                add_run(groups, dataset, "bert", seed, load_result(bert_path))
            else:
                missing.append(bert_path)
            if surface_path.exists():
                add_run(groups, dataset, "socialdebias", seed, load_result(surface_path))
            else:
                missing.append(surface_path)

    if not groups:
        raise FileNotFoundError("没有找到可汇总的评测 JSON 文件")

    rows = []
    for (dataset, model, split), runs in sorted(groups.items()):
        accuracy_mean, accuracy_std = mean_std([run["accuracy"] for run in runs])
        f1_mean, f1_std = mean_std([run["f1"] for run in runs])
        auc_mean, auc_std = mean_std([run["auc"] for run in runs])
        asr_mean, asr_std = mean_std([run["asr"] for run in runs])
        rows.append({
            "dataset": dataset,
            "model": model,
            "split": split,
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "asr_mean": asr_mean,
            "asr_std": asr_std,
            "n": len(runs),
        })

    print(f"{'数据集':<12}{'模型':<16}{'测试集':<10}{'F1':<20}{'ASR':<20}{'N':<4}")
    print("-" * 82)
    for row in rows:
        f1 = "N/A" if row["f1_mean"] is None else f"{row['f1_mean']:.4f}±{row['f1_std']:.4f}"
        asr = "N/A" if row["asr_mean"] is None else f"{row['asr_mean'] * 100:.2f}±{row['asr_std'] * 100:.2f}%"
        print(f"{row['dataset']:<12}{row['model']:<16}{row['split']:<10}{f1:<20}{asr:<20}{row['n']:<4}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n汇总结果已保存：{args.output}")

    if missing:
        print("\n缺少以下评测结果：")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
