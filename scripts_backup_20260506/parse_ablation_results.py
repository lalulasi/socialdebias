"""
汇总消融实验结果（4 变体 × 3 种子 × N 数据集）。
运行: python scripts/parse_ablation_results.py

输出三张表：
1. 主汇总表（均值±标准差，按变体排序）
2. 明细表（每个种子的结果）
3. 相对于 full 的 F1/Acc/AUC 差值（方便在论文里写"去掉 X 之后 F1 下降 y pp"）
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

RESULT_DIR = Path("results/ablation")

VAR_ORDER = ["full", "no_grl", "no_consist", "no_both"]
VAR_LABEL = {
    "full": "SocialDebias (full)",
    "no_grl": "w/o GRL",
    "no_consist": "w/o Consist",
    "no_both": "w/o Both (= BERT)",
}


def fmt_ms(mean, std):
    return f"{mean:.4f}±{std:.4f}"


def main():
    # (dataset, variant) -> list of json dicts
    groups = defaultdict(list)
    for jf in sorted(RESULT_DIR.glob("ablation_*.json")):
        with open(jf) as f:
            d = json.load(f)
        groups[(d["dataset"], d["variant"])].append(d)

    if not groups:
        print(f"[warn] 在 {RESULT_DIR} 里没找到 ablation_*.json 文件")
        return

    datasets = sorted(set(k[0] for k in groups.keys()))

    # ============ 表 1：主汇总表 ============
    print("\n" + "=" * 112)
    print("表 1 · 消融实验主汇总（均值±标准差）")
    print("=" * 112)
    print(f"{'Dataset':<12}{'Variant':<22}{'λ_bias':<8}{'λ_cons':<8}"
          f"{'Test Acc':<18}{'Test F1':<18}{'Test AUC':<18}{'N':<4}")
    print("-" * 112)

    summary_rows = []
    for dataset in datasets:
        for variant in VAR_ORDER:
            runs = groups.get((dataset, variant), [])
            if not runs:
                continue
            accs = [r["test"]["acc"] for r in runs]
            f1s = [r["test"]["f1"] for r in runs]
            aucs = [r["test"]["auc"] for r in runs]
            lam = runs[0]["lambdas"]

            label = VAR_LABEL.get(variant, variant)
            print(f"{dataset:<12}{label:<22}"
                  f"{lam['lambda_bias']:<8.2f}{lam['lambda_consist']:<8.2f}"
                  f"{fmt_ms(np.mean(accs), np.std(accs)):<18}"
                  f"{fmt_ms(np.mean(f1s), np.std(f1s)):<18}"
                  f"{fmt_ms(np.mean(aucs), np.std(aucs)):<18}"
                  f"{len(runs):<4}")

            summary_rows.append({
                "dataset": dataset, "variant": variant,
                "lambda_bias": lam["lambda_bias"],
                "lambda_consist": lam["lambda_consist"],
                "test_acc_mean": np.mean(accs), "test_acc_std": np.std(accs),
                "test_f1_mean": np.mean(f1s), "test_f1_std": np.std(f1s),
                "test_auc_mean": np.mean(aucs), "test_auc_std": np.std(aucs),
                "n": len(runs),
            })

    # ============ 表 2：相对 full 的 Δ（F1 变化） ============
    print("\n" + "=" * 80)
    print("表 2 · 相对 full 变体的 Δ（负值=去掉该组件后性能下降，即该组件的贡献）")
    print("=" * 80)
    print(f"{'Dataset':<12}{'Variant':<22}{'ΔAcc':<14}{'ΔF1':<14}{'ΔAUC':<14}")
    print("-" * 80)

    delta_rows = []
    for dataset in datasets:
        full_row = next((r for r in summary_rows
                         if r["dataset"] == dataset and r["variant"] == "full"), None)
        if not full_row:
            continue

        for variant in VAR_ORDER:
            if variant == "full":
                continue
            row = next((r for r in summary_rows
                        if r["dataset"] == dataset and r["variant"] == variant), None)
            if not row:
                continue

            d_acc = row["test_acc_mean"] - full_row["test_acc_mean"]
            d_f1 = row["test_f1_mean"] - full_row["test_f1_mean"]
            d_auc = row["test_auc_mean"] - full_row["test_auc_mean"]
            label = VAR_LABEL.get(variant, variant)
            print(f"{dataset:<12}{label:<22}"
                  f"{d_acc:+.4f}       {d_f1:+.4f}       {d_auc:+.4f}")

            delta_rows.append({
                "dataset": dataset, "variant": variant,
                "delta_acc": d_acc, "delta_f1": d_f1, "delta_auc": d_auc,
            })

    # ============ 表 3：所有单次明细 ============
    print("\n" + "=" * 98)
    print("表 3 · 所有单次实验明细（每个种子一行）")
    print("=" * 98)
    print(f"{'Dataset':<12}{'Variant':<22}{'Seed':<8}{'Best_E':<8}"
          f"{'Test Acc':<12}{'Test F1':<12}{'Test AUC':<12}")
    print("-" * 98)

    detail_rows = []
    for dataset in datasets:
        for variant in VAR_ORDER:
            runs = groups.get((dataset, variant), [])
            for r in sorted(runs, key=lambda x: x["seed"]):
                label = VAR_LABEL.get(variant, variant)
                test = r["test"]
                print(f"{dataset:<12}{label:<22}{r['seed']:<8}{test['epoch']:<8}"
                      f"{test['acc']:<12.4f}{test['f1']:<12.4f}{test['auc']:<12.4f}")
                detail_rows.append({
                    "dataset": dataset, "variant": variant, "seed": r["seed"],
                    "best_epoch": test["epoch"],
                    "test_acc": test["acc"], "test_f1": test["f1"], "test_auc": test["auc"],
                    "best_val_f1": r["best_val_f1"],
                })

    # ============ 保存 CSV ============
    import csv

    def save_csv(rows, filename):
        if not rows:
            return
        path = RESULT_DIR / filename
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  已保存：{path}")

    print("\n" + "=" * 80)
    print("CSV 文件")
    print("=" * 80)
    save_csv(summary_rows, "ablation_summary.csv")
    save_csv(delta_rows, "ablation_delta.csv")
    save_csv(detail_rows, "ablation_detail.csv")

    # ============ 一段文字建议（供论文写作直接抄） ============
    print("\n" + "=" * 80)
    print("论文写作参考段（5.5.1 节可直接改写）")
    print("=" * 80)
    for dataset in datasets:
        full_row = next((r for r in summary_rows
                         if r["dataset"] == dataset and r["variant"] == "full"), None)
        if not full_row:
            continue
        print(f"\n[{dataset}]")
        print(f"  完整 SocialDebias 达到 F1={full_row['test_f1_mean']:.4f}±{full_row['test_f1_std']:.4f}。")
        for row in delta_rows:
            if row["dataset"] != dataset:
                continue
            variant_name = VAR_LABEL.get(row["variant"], row["variant"])
            if row["delta_f1"] < 0:
                print(f"  去除 {variant_name}：F1 下降 {abs(row['delta_f1'])*100:.2f}pp，"
                      f"证明该组件贡献 {abs(row['delta_f1'])*100:.2f}pp")
            else:
                print(f"  去除 {variant_name}：F1 反而上升 {row['delta_f1']*100:.2f}pp，"
                      f"需要讨论（可能是小样本波动或该组件在该数据集上不关键）")


if __name__ == "__main__":
    main()