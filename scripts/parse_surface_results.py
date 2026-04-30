"""
汇总 SD + 表层特征实验结果（含 InfoNCE 变体）。
对比基准：SD（无表层特征）= 原 train_socialdebias 主实验
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
        # 文件名格式: surface_adv_{dataset}_seed{N}_{suffix}.json
        # 但 d 里也有 dataset / suffix 信息
        # 从 ckpt 路径解析
        ckpt = d.get("ckpt", "")
        # ckpt: ./results/models/socialdebias_politifact_en_seed42_surface.pt
        parts = Path(ckpt).stem.split("_")  
        # ['socialdebias', 'politifact', 'en', 'seed42', 'surface']
        dataset = parts[1]
        suffix = "_".join(parts[4:])  # 处理 surface_contrast
        groups[(dataset, suffix)].append(d)

    if not groups:
        print(f"[warn] 没找到结果")
        return

    print("\n" + "=" * 110)
    print("★ 论文 5.5.4 节：表层特征消融对照表（3 种子均值±std）")
    print("=" * 110)
    print(f"{'Dataset':<12}{'配置':<30}{'Clean F1':<22}{'Avg Adv F1':<22}{'F1 Drop':<22}{'N':<4}")
    print("-" * 110)

    rows = []
    suffix_order = ["surface", "surface_contrast"]
    suffix_label = {
        "surface": "SD + 17维表层",
        "surface_contrast": "SD + 17维表层 + InfoNCE",
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

    # 对比基准
    print("\n=== 对比基准（来自之前实验）===")
    print("PolitiFact SD baseline:        Clean F1=0.8699±0.0263, F1 Drop=8.80pp±1.32")
    print("PolitiFact SD+InfoNCE λ=0.3:   Clean F1=0.8852±0.0210, F1 Drop=6.50pp±0.94")
    print("GossipCop SD baseline:         Clean F1=0.7974±0.0015, F1 Drop=2.45pp±1.64")
    
    # CSV
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