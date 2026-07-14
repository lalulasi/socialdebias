"""
解释一致性 → 对抗鲁棒性 相关分析 + 绘图
================================================================
输入：extract_expl_robust_xy.py 产出的逐样本×变体 CSV
      （expl_robust_xy_politifact_ALL.csv，360 行）
输出：
  1) 终端打印全部统计量（合并 / 聚类稳健自助 / 按 origin 聚合 / 攻破vs存活）
  2) results/expl_robust_stats.csv   —— 统计量汇总表（供论文表格）
  3) results/fig_expl_robust.png/.pdf —— 两栏图：(a) 散点+回归  (b) 攻破vs存活箱线

一致性度量默认用 sd_spearman（信号最强、且对齐论文表 5-8 的 Spearman）。
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    def col(rs, k):
        return np.array([float(r[k]) for r in rs])
    return rows, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/expl_robust_xy_politifact_ALL.csv")
    ap.add_argument("--xmetric", default="sd_spearman",
                    help="一致性度量列（sd_spearman / sd_top_k_overlap）")
    ap.add_argument("--fig", default="results/fig_expl_robust")
    ap.add_argument("--stats_csv", default="results/expl_robust_stats.csv")
    ap.add_argument("--n_boot", type=int, default=5000)
    args = ap.parse_args()

    rows, col = load(args.csv)
    X = col(rows, args.xmetric); Y = col(rows, "delta_p")
    mask = ~(np.isnan(X) | np.isnan(Y))
    X, Y = X[mask], Y[mask]
    rows_ok = [r for r, ok in zip(rows, mask) if ok]

    # 1) 合并
    rho_all, p_all = stats.spearmanr(X, Y)

    # 2) 聚类稳健：按 origin(idx) 自助
    by_o = {}
    for r in rows_ok:
        by_o.setdefault(r["idx"], []).append(r)
    origins = list(by_o.keys())
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(args.n_boot):
        samp = rng.choice(origins, size=len(origins), replace=True)
        rs = [r for o in samp for r in by_o[o]]
        xx = np.array([float(r[args.xmetric]) for r in rs])
        yy = np.array([float(r["delta_p"]) for r in rs])
        rr, _ = stats.spearmanr(xx, yy)
        boot.append(rr)
    boot = np.array(boot)
    ci = np.percentile(boot, [2.5, 97.5])
    p_boot = 2 * min((boot >= 0).mean(), (boot <= 0).mean())

    # 3) 按 origin 聚合（独立点）
    ax_ = np.array([np.mean([float(r[args.xmetric]) for r in rs]) for rs in by_o.values()])
    ay_ = np.array([np.mean([float(r["delta_p"]) for r in rs]) for rs in by_o.values()])
    rho_agg, p_agg = stats.spearmanr(ax_, ay_)

    # 4) 攻破 vs 存活
    flip = np.array([float(r[args.xmetric]) for r in rows_ok
                     if r["correct_orig"] == "1" and r["correct_adv"] == "0"])
    keep = np.array([float(r[args.xmetric]) for r in rows_ok
                     if r["correct_orig"] == "1" and r["correct_adv"] == "1"])
    u, p_mwu = stats.mannwhitneyu(flip, keep, alternative="less")

    # ---- 打印 + 存表 ----
    summary = [
        ("合并（全部样本×变体）", f"n={len(X)}", f"rho={rho_all:+.3f}", f"p={p_all:.2e}"),
        ("聚类稳健（按origin自助）", f"boot={args.n_boot}",
         f"95%CI=[{ci[0]:+.3f},{ci[1]:+.3f}]", f"p={p_boot:.4f}"),
        ("按origin聚合（独立点）", f"n={len(ax_)}", f"rho={rho_agg:+.3f}", f"p={p_agg:.4f}"),
        ("攻破vs存活（MWU单尾）", f"flip={len(flip)},keep={len(keep)}",
         f"均值{flip.mean():.3f}vs{keep.mean():.3f}", f"p={p_mwu:.4f}"),
    ]
    print("\n" + "=" * 74)
    print(f"一致性度量 = {args.xmetric}   （越高越一致；预期与 Δp 负相关）")
    print("=" * 74)
    for a, b, c, d in summary:
        print(f"  {a:<22}{b:<18}{c:<24}{d}")
    print("=" * 74)
    Path(args.stats_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["分析", "样本", "统计量", "显著性"])
        w.writerows(summary)
    print(f"统计汇总 -> {args.stats_csv}")

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.scatter(X, Y, s=18, alpha=0.45, color="#3b6ea5", edgecolors="none")
    b1, a1 = np.polyfit(X, Y, 1)
    xs = np.linspace(X.min(), X.max(), 50)
    ax.plot(xs, b1 * xs + a1, color="#c0392b", lw=2)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Explanation consistency (Spearman of IG attributions)")
    ax.set_ylabel("Confidence drop (delta p)")
    ax.set_title(f"(a) Per-sample (n={len(X)})   rho={rho_all:.3f}, p<1e-4")

    ax = axes[1]
    bp = ax.boxplot([keep, flip], tick_labels=[f"Survived (n={len(keep)})",
                    f"Attacked (n={len(flip)})"], patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#27ae60", "#c0392b"]):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    rng2 = np.random.default_rng(1)
    for i, d in enumerate([keep, flip]):
        ax.scatter(rng2.normal(i + 1, 0.06, len(d)), d, s=12, alpha=0.4, color="gray")
    ax.set_ylabel("Explanation consistency (Spearman)")
    ax.set_title(f"(b) Attacked vs Survived   MWU p={p_mwu:.4f}")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{args.fig}.{ext}", dpi=200, bbox_inches="tight")
    print(f"图 -> {args.fig}.png / .pdf")


if __name__ == "__main__":
    main()
