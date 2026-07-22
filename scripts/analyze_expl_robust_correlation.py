"""Build paper Table 5-10 and Figure 5.4 from explanation/robustness pairs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


REQUIRED_COLUMNS = {
    "variant",
    "idx",
    "sd_spearman",
    "delta_p",
    "correct_orig",
    "correct_adv",
}


def finite_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan"), float("nan"), int(keep.sum())
    rho, p_value = spearmanr(x[keep], y[keep])
    return float(rho), float(p_value), int(keep.sum())


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.stem)
    if not match:
        raise ValueError(f"文件名缺少 seed 编号: {path}")
    return int(match.group(1))


def load_seed(path: Path, expected_rows: int) -> tuple[int, pd.DataFrame]:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(f"{path} 应有 {expected_rows} 行，实际 {len(frame)} 行")
    counts = frame.groupby("variant").size().to_dict()
    if counts != {"A": 90, "B": 90, "C": 90, "D": 90}:
        raise ValueError(f"{path} 变体覆盖错误: {counts}")
    frame = frame.copy()
    for column in ("idx", "sd_spearman", "delta_p", "correct_orig", "correct_adv"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return seed_from_path(path), frame


def cluster_bootstrap(frame: pd.DataFrame, iterations: int, rng) -> dict:
    cluster_ids = np.asarray(sorted(frame["idx"].dropna().unique()))
    values = []
    grouped = {idx: group for idx, group in frame.groupby("idx", sort=False)}
    for _ in range(iterations):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        boot = pd.concat([grouped[idx] for idx in sampled], ignore_index=True)
        rho, _, _ = finite_spearman(boot["sd_spearman"], boot["delta_p"])
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        return {"iterations": 0, "ci_low": None, "ci_high": None, "sign_p": None}
    arr = np.asarray(values)
    # This empirical sign probability accompanies the cluster-bootstrap CI. It
    # is not a permutation-test p-value, so the JSON keeps the explicit name.
    sign_p = min(1.0, 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0))))
    return {
        "iterations": len(values),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "sign_p": sign_p,
    }


def analyze_seed(seed: int, frame: pd.DataFrame, bootstrap_iters: int, rng) -> dict:
    pooled_rho, pooled_p, pooled_n = finite_spearman(
        frame["sd_spearman"], frame["delta_p"]
    )
    by_original = (
        frame.groupby("idx", as_index=False)[["sd_spearman", "delta_p"]]
        .mean(numeric_only=True)
    )
    aggregate_rho, aggregate_p, aggregate_n = finite_spearman(
        by_original["sd_spearman"], by_original["delta_p"]
    )

    eligible = frame[frame["correct_orig"] == 1].copy()
    broken = eligible[eligible["correct_adv"] == 0]["sd_spearman"].dropna().to_numpy()
    survived = eligible[eligible["correct_adv"] == 1]["sd_spearman"].dropna().to_numpy()
    if len(broken) and len(survived):
        _, mwu_p = mannwhitneyu(broken, survived, alternative="less")
    else:
        mwu_p = float("nan")

    return {
        "seed": seed,
        "pooled": {"n": pooled_n, "rho": pooled_rho, "p_value": pooled_p},
        "cluster_bootstrap": {
            "n_clusters": int(frame["idx"].nunique()),
            **cluster_bootstrap(frame, bootstrap_iters, rng),
        },
        "by_original": {
            "n": aggregate_n,
            "rho": aggregate_rho,
            "p_value": aggregate_p,
        },
        "broken_vs_survived": {
            "broken_n": int(len(broken)),
            "survived_n": int(len(survived)),
            "broken_mean": float(np.mean(broken)) if len(broken) else None,
            "survived_mean": float(np.mean(survived)) if len(survived) else None,
            "mwu_alternative": "broken < survived",
            "mwu_p_value": float(mwu_p),
        },
    }


def make_figure(frame: pd.DataFrame, stats: dict, output_dir: Path, seed: int, dpi: int):
    valid = frame[np.isfinite(frame["sd_spearman"]) & np.isfinite(frame["delta_p"])]
    eligible = frame[frame["correct_orig"] == 1]
    broken = eligible[eligible["correct_adv"] == 0]["sd_spearman"].dropna().to_numpy()
    survived = eligible[eligible["correct_adv"] == 1]["sd_spearman"].dropna().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(
        valid["sd_spearman"], valid["delta_p"], s=16, alpha=0.45, edgecolors="none"
    )
    if len(valid) >= 3 and valid["sd_spearman"].nunique() > 1:
        slope, intercept = np.polyfit(valid["sd_spearman"], valid["delta_p"], 1)
        xs = np.linspace(valid["sd_spearman"].min(), valid["sd_spearman"].max(), 200)
        axes[0].plot(xs, slope * xs + intercept, color="#c43c39", linewidth=2)
    axes[0].set_xlabel("IG Spearman consistency")
    axes[0].set_ylabel("True-class confidence drop (delta p)")
    axes[0].set_title(
        f"(a) Per-pair correlation, seed {seed}\n"
        f"rho={stats['pooled']['rho']:.3f}, p={stats['pooled']['p_value']:.3g}"
    )
    axes[0].grid(alpha=0.2)

    groups = []
    labels = []
    if len(survived):
        groups.append(survived)
        labels.append(f"Survived\n(n={len(survived)})")
    if len(broken):
        groups.append(broken)
        labels.append(f"Broken\n(n={len(broken)})")
    if groups:
        axes[1].boxplot(groups, tick_labels=labels, showmeans=True)
    axes[1].set_ylabel("IG Spearman consistency")
    p_value = stats["broken_vs_survived"]["mwu_p_value"]
    axes[1].set_title(f"(b) Robustness groups\none-sided MWU p={p_value:.3g}")
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"figure5_4_explanation_robustness_seed{seed}.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_primary_table(stats: dict, output: Path):
    bootstrap = stats["cluster_bootstrap"]
    groups = stats["broken_vs_survived"]
    rows = [
        {
            "analysis": "pooled_pairs",
            "sample_size": stats["pooled"]["n"],
            "statistic": f"rho={stats['pooled']['rho']:.6f}",
            "significance": f"p={stats['pooled']['p_value']:.8g}",
        },
        {
            "analysis": "cluster_bootstrap_by_original",
            "sample_size": f"{bootstrap['n_clusters']} clusters",
            "statistic": f"95% CI [{bootstrap['ci_low']:.6f}, {bootstrap['ci_high']:.6f}]",
            "significance": f"empirical sign p={bootstrap['sign_p']:.8g}",
        },
        {
            "analysis": "aggregated_by_original",
            "sample_size": stats["by_original"]["n"],
            "statistic": f"rho={stats['by_original']['rho']:.6f}",
            "significance": f"p={stats['by_original']['p_value']:.8g}",
        },
        {
            "analysis": "broken_vs_survived_mwu",
            "sample_size": f"{groups['broken_n']} vs {groups['survived_n']}",
            "statistic": (
                f"mean {groups['broken_mean']:.6f} vs {groups['survived_mean']:.6f}"
            ),
            "significance": f"one-sided p={groups['mwu_p_value']:.8g}",
        },
    ]
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--primary_seed", type=int, default=42)
    parser.add_argument("--expected_rows", type=int, default=360)
    parser.add_argument("--bootstrap_iters", type=int, default=5000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--figure_dpi", type=int, default=300)
    args = parser.parse_args()

    if args.bootstrap_iters <= 0:
        raise ValueError("--bootstrap_iters must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    frames = {}
    analyses = []
    for path in args.inputs:
        seed, frame = load_seed(path, args.expected_rows)
        if seed in frames:
            raise ValueError(f"重复 seed: {seed}")
        frames[seed] = frame
        analyses.append(analyze_seed(seed, frame, args.bootstrap_iters, rng))
    analyses.sort(key=lambda item: item["seed"])

    if args.primary_seed not in frames:
        raise ValueError(f"缺少 primary seed {args.primary_seed}")
    primary_stats = next(item for item in analyses if item["seed"] == args.primary_seed)

    pooled_rhos = np.asarray([item["pooled"]["rho"] for item in analyses], dtype=float)
    aggregate_rhos = np.asarray([item["by_original"]["rho"] for item in analyses], dtype=float)
    summary = {
        "primary_seed": args.primary_seed,
        "method": {
            "correlation": "Spearman",
            "cluster_bootstrap_iterations": args.bootstrap_iters,
            "cluster_key": "original sample idx",
            "mwu_alternative": "broken consistency < survived consistency",
        },
        "per_seed": analyses,
        "three_seed": {
            "n_seeds": len(analyses),
            "pooled_rho_mean": float(np.mean(pooled_rhos)),
            "pooled_rho_std": float(np.std(pooled_rhos, ddof=0)),
            "by_original_rho_mean": float(np.mean(aggregate_rhos)),
            "by_original_rho_std": float(np.std(aggregate_rhos, ddof=0)),
        },
    }

    with open(
        args.output_dir / "explanation_robustness_summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    flat_rows = []
    for item in analyses:
        flat_rows.append({
            "seed": item["seed"],
            "pooled_n": item["pooled"]["n"],
            "pooled_rho": item["pooled"]["rho"],
            "pooled_p": item["pooled"]["p_value"],
            "cluster_ci_low": item["cluster_bootstrap"]["ci_low"],
            "cluster_ci_high": item["cluster_bootstrap"]["ci_high"],
            "cluster_sign_p": item["cluster_bootstrap"]["sign_p"],
            "by_original_n": item["by_original"]["n"],
            "by_original_rho": item["by_original"]["rho"],
            "by_original_p": item["by_original"]["p_value"],
            "broken_n": item["broken_vs_survived"]["broken_n"],
            "survived_n": item["broken_vs_survived"]["survived_n"],
            "broken_mean": item["broken_vs_survived"]["broken_mean"],
            "survived_mean": item["broken_vs_survived"]["survived_mean"],
            "mwu_p": item["broken_vs_survived"]["mwu_p_value"],
        })
    pd.DataFrame(flat_rows).to_csv(
        args.output_dir / "explanation_robustness_3seed_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_primary_table(
        primary_stats,
        args.output_dir / f"table5_10_seed{args.primary_seed}.csv",
    )
    make_figure(
        frames[args.primary_seed],
        primary_stats,
        args.output_dir,
        args.primary_seed,
        args.figure_dpi,
    )
    print(f"TABLE5_10_READY={args.output_dir / f'table5_10_seed{args.primary_seed}.csv'}")
    print(
        "FIGURE5_4_READY="
        f"{args.output_dir / f'figure5_4_explanation_robustness_seed{args.primary_seed}.png'}"
    )
    print("P6B_CORRELATION_ANALYSIS_READY=True")


if __name__ == "__main__":
    main()
