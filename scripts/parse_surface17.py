"""汇总 PolitiFact 上 8 维与 17 维表层特征的对抗评测结果。"""
import json, csv, os
import numpy as np

RESULT_DIR = "results/surface_adv"
DATASET = "politifact"
SEEDS = [42, 2024, 3407]
# 显示名称与训练时使用的检查点后缀
CONFIGS = [
    ("8 维（仅情绪, surface）",        "surface"),
    ("17 维（情绪+句法+词汇, surface17）", "surface17"),
]

def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], (int, float)):
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None

rows = []
for name, suffix in CONFIGS:
    cleans, drops = [], []
    for seed in SEEDS:
        p = os.path.join(RESULT_DIR, f"surface_adv_{DATASET}_seed{seed}_{suffix}.json")
        if not os.path.exists(p):
            print(f"  [缺失] {p}")
            continue
        d = json.load(open(p, encoding="utf-8"))
        c, dr = find_key(d, "clean_f1"), find_key(d, "f1_drop")
        if c is None or dr is None:
            print(f"  [字段缺失] {p}")
            continue
        cleans.append(c); drops.append(dr)
    if not drops:
        print(f"  [跳过] {name}: 无有效结果")
        continue
    rows.append({
        "配置": name, "suffix": suffix, "n_seeds": len(drops),
        "Clean_F1": f"{np.mean(cleans):.4f} ± {np.std(cleans):.4f}",
        "F1_Drop_pp": f"{np.mean(drops)*100:.2f} ± {np.std(drops)*100:.2f}",
    })

os.makedirs("results", exist_ok=True)
out = "results/surface17_vs_8_politifact.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["配置", "suffix", "n_seeds", "Clean_F1", "F1_Drop_pp"])
    w.writeheader(); w.writerows(rows)

print(f"\n已保存: {out}\n")
for r in rows:
    print(f"  {r['配置']:28s} Clean F1={r['Clean_F1']}   F1 Drop={r['F1_Drop_pp']}")
if len(rows) == 2:
    d8 = float(rows[0]["F1_Drop_pp"].split("±")[0])
    d17 = float(rows[1]["F1_Drop_pp"].split("±")[0])
    v = "17 维更差（降幅更大）→ 支持'少即是多'" if d17 > d8 + 0.05 else \
        ("17 维更好（降幅更小）→ 与原论点相反，需重写" if d17 < d8 - 0.05 else "两者基本持平→'少即是多'应弱化为'未见增益'")
    print(f"\n  → 8维={d8}pp vs 17维={d17}pp：{v}")
