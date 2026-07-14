"""
汇总表层特征实验及其 InfoNCE 变体的评测结果。
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

RESULT_DIR = Path("results/surface_adv")


def main():
    # (dataset, save_suffix) -> list of result dicts
    groups = defaultdict(list)
    for jf in sorted(RESULT_DIR.glob("surface_adv_*.json")):
        with open(jf) as f:
            d = json.load(f)
        # 从检查点路径中提取数据集与实验后缀
        ckpt = d.get("ckpt", "")
        parts = Path(ckpt).stem.split("_")  
        dataset = parts[1]
        suffix = "_".join(parts[4:])
        groups[(dataset, suffix)].append(d)

    if not groups:
        print(f"[warn] 没找到结果")
        return

    print("\n" + "=" * 110)
    print("表层特征实验结果（三种子均值和标准差）")
    print("=" * 110)
    print(f"{'Dataset':<12}{'配置':<30}{'Clean F1':<22}{'Avg Adv F1':<22}{'F1 Drop':<22}{'N':<4}")
    print("-" * 110)

    rows = []
    suffix_order = ["surface", "surface_contrast"]
    suffix_label = {
        "surface": "SD + 8维表层",
        "surface_contrast": "SD + 8维表层 + InfoNCE",
    }
    
    for ds in ["politifact", "gossipcop"]:
        for suffix in suffix_order:
            runs = groups.get((ds, suffix), [])
            if not runs:
                continue
            clean_f1s = [r["results"]["clean"]["f1"] for r in runs]
            avg_adv_f1s = [r["results"]["summary"]["avg_adv_f1"] for r in runs]
            drops = [r["results"]["summary"]["f1_drop"] for r in runs]
            
            label = suffix_label.get(suffix, suffix)
            print(f"{ds:<12}{label:<30}"
                  f"{np.mean(clean_f1s):.4f}±{np.std(clean_f1s):.4f}    "
                  f"{np.mean(avg_adv_f1s):.4f}±{np.std(avg_adv_f1s):.4f}    "
                  f"{np.mean(drops):.4f}±{np.std(drops):.4f}    "
                  f"{len(runs)}")
            
            rows.append({
                "dataset": ds, "config": label,
                "clean_f1_mean": np.mean(clean_f1s), "clean_f1_std": np.std(clean_f1s),
                "avg_adv_f1_mean": np.mean(avg_adv_f1s),
                "avg_adv_f1_std": np.std(avg_adv_f1s),
                "f1_drop_mean": np.mean(drops),
                "f1_drop_std": np.std(drops),
                "n": len(runs),
            })

    import csv
    if rows:
        out_csv = RESULT_DIR / "surface_summary.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\n汇总: {out_csv}")


if __name__ == "__main__":
    main()
