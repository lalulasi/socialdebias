"""
汇总 LSTM 基线 6 个实验结果（3 种子 × 2 数据集），算均值±标准差。
运行: python scripts/parse_lstm_results.py
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

RESULT_DIR = Path("results/lstm")


def main():
    # key = dataset, value = list of (seed, test_metrics)
    groups = defaultdict(list)
    for jf in sorted(RESULT_DIR.glob("lstm_*_result.json")):
        with open(jf) as f:
            data = json.load(f)
        groups[data["dataset"]].append((data["seed"], data["test"], data["best_val"]))

    print(f"\n{'Dataset':<12}{'Acc mean±std':<22}{'F1 mean±std':<22}{'AUC mean±std':<22}{'N':<4}")
    print("=" * 82)

    # 同时输出 CSV
    csv_rows = []
    for dataset, runs in sorted(groups.items()):
        accs = [r[1]["acc"] for r in runs]
        f1s = [r[1]["f1"] for r in runs]
        aucs = [r[1]["auc"] for r in runs]

        acc_s = f"{np.mean(accs):.4f}±{np.std(accs):.4f}"
        f1_s = f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}"
        auc_s = f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}"
        print(f"{dataset:<12}{acc_s:<22}{f1_s:<22}{auc_s:<22}{len(runs):<4}")

        # 明细
        for seed, test, best_val in runs:
            csv_rows.append({
                "dataset": dataset, "seed": seed,
                "best_val_epoch": best_val["epoch"],
                "test_acc": test["acc"], "test_f1": test["f1"], "test_auc": test["auc"],
                "val_acc": best_val["acc"], "val_f1": best_val["f1"],
            })

    print(f"\n明细：")
    print(f"{'Dataset':<12}{'Seed':<8}{'Best_E':<8}{'Test_Acc':<12}{'Test_F1':<12}{'Test_AUC':<12}")
    print("=" * 64)
    for r in csv_rows:
        print(f"{r['dataset']:<12}{r['seed']:<8}{r['best_val_epoch']:<8}"
              f"{r['test_acc']:<12.4f}{r['test_f1']:<12.4f}{r['test_auc']:<12.4f}")

    # 保存 CSV
    import csv
    out_csv = RESULT_DIR / "lstm_summary.csv"
    if csv_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n明细 CSV 已保存：{out_csv}")


if __name__ == "__main__":
    main()