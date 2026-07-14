"""
汇总四种消融设置在三个随机种子下的对抗评测结果。

运行: python scripts/parse_ablation_adv.py
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

import sys; RESULT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ablation_adv")
VAR_ORDER = ["full", "no_grl", "no_consist", "no_both"]
VAR_LABEL = {
    "full": "SocialDebias (full)",
    "no_grl": "w/o GRL",
    "no_consist": "w/o Consist",
    "no_both": "w/o Both (≈BERT)",
}


def fmt_ms(vals):
    if not vals:
        return "N/A"
    return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"


def main():
    groups = defaultdict(list)
    for jf in sorted(RESULT_DIR.glob("ablation_adv_*.json")):
        with open(jf) as f:
            d = json.load(f)
        variant = d["variant"]
        for prefix in ("abl_", "b1_"):
            if variant.startswith(prefix):
                variant = variant.removeprefix(prefix)
                break
        key = (d["dataset"], variant)
        groups[key].append(d)

    if not groups:
        print(f"[warn] 在 {RESULT_DIR} 没找到 ablation_adv_*.json")
        return

    datasets = sorted(set(k[0] for k in groups.keys()))

    # 各消融设置的完整指标
    print("\n" + "=" * 120)
    print("消融实验结果（对抗鲁棒性）")
    print("=" * 120)
    print(f"{'Dataset':<12}{'Variant':<22}"
          f"{'Clean F1':<18}{'Adv A F1':<16}{'Adv B F1':<16}{'Adv C F1':<16}{'Adv D F1':<16}"
          f"{'Avg Adv F1':<18}{'F1 Drop':<14}")
    print("-" * 120)

    rows = []
    for dataset in datasets:
        for variant in VAR_ORDER:
            runs = groups.get((dataset, variant), [])
            if not runs:
                continue

            clean_f1s = [r["results"]["clean"]["f1"] for r in runs]
            adv_a_f1s = [r["results"].get("adv_A", {}).get("f1") for r in runs]
            adv_b_f1s = [r["results"].get("adv_B", {}).get("f1") for r in runs]
            adv_c_f1s = [r["results"].get("adv_C", {}).get("f1") for r in runs]
            adv_d_f1s = [r["results"].get("adv_D", {}).get("f1") for r in runs]
            avg_adv_f1s = [r["results"].get("summary", {}).get("avg_adv_f1") for r in runs]
            drops = [r["results"].get("summary", {}).get("f1_drop") for r in runs]

            # 清理 None
            adv_a_f1s = [x for x in adv_a_f1s if x is not None]
            adv_b_f1s = [x for x in adv_b_f1s if x is not None]
            adv_c_f1s = [x for x in adv_c_f1s if x is not None]
            adv_d_f1s = [x for x in adv_d_f1s if x is not None]
            avg_adv_f1s = [x for x in avg_adv_f1s if x is not None]
            drops = [x for x in drops if x is not None]

            label = VAR_LABEL.get(variant, variant)
            print(f"{dataset:<12}{label:<22}"
                  f"{fmt_ms(clean_f1s):<18}"
                  f"{fmt_ms(adv_a_f1s):<16}"
                  f"{fmt_ms(adv_b_f1s):<16}"
                  f"{fmt_ms(adv_c_f1s):<16}"
                  f"{fmt_ms(adv_d_f1s):<16}"
                  f"{fmt_ms(avg_adv_f1s):<18}"
                  f"{fmt_ms(drops):<14}")

            rows.append({
                "dataset": dataset, "variant": variant,
                "clean_f1_mean": np.mean(clean_f1s), "clean_f1_std": np.std(clean_f1s),
                "adv_a_f1_mean": np.mean(adv_a_f1s) if adv_a_f1s else None,
                "adv_b_f1_mean": np.mean(adv_b_f1s) if adv_b_f1s else None,
                "adv_c_f1_mean": np.mean(adv_c_f1s) if adv_c_f1s else None,
                "adv_d_f1_mean": np.mean(adv_d_f1s) if adv_d_f1s else None,
                "avg_adv_f1_mean": np.mean(avg_adv_f1s) if avg_adv_f1s else None,
                "f1_drop_mean": np.mean(drops) if drops else None,
                "f1_drop_std": np.std(drops) if drops else None,
                "n": len(runs),
            })

    # 与完整模型比较 F1 降幅
    print("\n" + "=" * 80)
    print("移除组件后的 F1 降幅变化")
    print("=" * 80)
    print(f"{'Dataset':<12}{'Variant':<22}{'F1 Drop':<20}{'Δ Drop vs full':<18}")
    print("-" * 80)

    for dataset in datasets:
        full_row = next((r for r in rows if r["dataset"] == dataset and r["variant"] == "full"), None)
        if not full_row or full_row["f1_drop_mean"] is None:
            continue

        for variant in VAR_ORDER:
            row = next((r for r in rows if r["dataset"] == dataset and r["variant"] == variant), None)
            if not row or row["f1_drop_mean"] is None:
                continue
            drop = row["f1_drop_mean"]
            delta = drop - full_row["f1_drop_mean"]
            label = VAR_LABEL.get(variant, variant)
            flag = "（完整模型）" if variant == "full" else f" (+{delta*100:.2f}pp)" if delta > 0 else f" ({delta*100:.2f}pp)"
            print(f"{dataset:<12}{label:<22}{drop:.4f} ({drop*100:.2f}pp){flag}")

    import csv
    if rows:
        out_csv = RESULT_DIR / "ablation_adv_summary.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\n汇总 CSV 已保存：{out_csv}")

if __name__ == "__main__":
    main()
