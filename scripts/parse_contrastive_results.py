"""
汇总对比学习实验结果（3 种子 × 3 个 λ_contrast 值 = 9 次）。
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

RESULT_DIR = Path("results/contrastive_adv_gc")


def main():
    # (lambda_contrast,) -> list of result dicts
    groups = defaultdict(list)
    for jf in sorted(RESULT_DIR.glob("contrastive_adv_*.json")):
        with open(jf) as f:
            d = json.load(f)
        groups[(d["lambda_contrast"],)].append(d)
    
    if not groups:
        print(f"[warn] 没找到结果: {RESULT_DIR}")
        return

    print("\n" + "=" * 100)
    print("对比学习消融：λ_contrast 对鲁棒性的影响")
    print("=" * 100)
    print(f"{'λ_contrast':<14}{'Clean F1':<22}{'Avg Adv F1':<22}{'F1 Drop':<22}{'N':<4}")
    print("-" * 100)

    rows = []
    for (lc,), runs in sorted(groups.items()):
        clean_f1s = [r["results"]["clean"]["f1"] for r in runs]
        avg_adv_f1s = [r["results"]["summary"]["avg_adv_f1"] for r in runs]
        drops = [r["results"]["summary"]["f1_drop"] for r in runs]
        
        clean_s = f"{np.mean(clean_f1s):.4f}±{np.std(clean_f1s):.4f}"
        adv_s = f"{np.mean(avg_adv_f1s):.4f}±{np.std(avg_adv_f1s):.4f}"
        drop_s = f"{np.mean(drops):.4f}±{np.std(drops):.4f}"
        
        print(f"{lc:<14.2f}{clean_s:<22}{adv_s:<22}{drop_s:<22}{len(runs):<4}")
        rows.append({
            "lambda_contrast": lc,
            "clean_f1_mean": np.mean(clean_f1s), "clean_f1_std": np.std(clean_f1s),
            "avg_adv_f1_mean": np.mean(avg_adv_f1s), "avg_adv_f1_std": np.std(avg_adv_f1s),
            "f1_drop_mean": np.mean(drops), "f1_drop_std": np.std(drops),
            "n": len(runs),
        })

    import csv
    if rows:
        out_csv = RESULT_DIR / "contrastive_summary.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\n汇总 CSV: {out_csv}")

if __name__ == "__main__":
    main()
