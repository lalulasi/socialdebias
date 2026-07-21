"""Aggregate SocialDebias/ablation history JSON files into reproducible CSVs.

The training scripts store one ``*_history.json`` per dataset, seed and
configuration.  This helper writes both run-level details and population
mean/std summaries (ddof=0), and can fail on incomplete three-seed groups.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "best_val_f1",
    "test_acc",
    "test_f1",
    "test_auc",
    "test_bias_acc",
)


def dataset_name(args):
    if args.get("use_weibo21"):
        return "weibo21"
    if args.get("use_liar"):
        return "liar"
    return args.get("dataset", "unknown")


def language_name(args):
    if args.get("use_weibo21"):
        return "zh"
    return args.get("language", "en")


def load_row(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    run_args = payload.get("args", {})
    best_test = payload.get("best_test") or payload.get("best_test_metrics") or {}
    suffix = run_args.get("save_suffix")
    if not suffix:
        # BERT histories do not have save_suffix, but keeping this fallback makes
        # accidental mixed-directory scans explicit rather than ambiguous.
        suffix = "bert_baseline" if "bert_baseline" in path.stem else "unknown"

    return {
        "dataset": dataset_name(run_args),
        "language": language_name(run_args),
        "suffix": suffix,
        "seed": run_args.get("seed"),
        "best_val_f1": payload.get("best_val_f1"),
        "best_epoch": best_test.get("epoch"),
        "test_acc": best_test.get("acc", best_test.get("accuracy")),
        "test_f1": best_test.get("f1"),
        "test_auc": best_test.get("auc"),
        "test_bias_acc": best_test.get("bias_acc"),
        "adaptive_enabled": bool(run_args.get("adaptive_lambda", False)),
        "adaptive_triggered": bool(run_args.get("adaptive_triggered", False)),
        "adaptive_trigger_reasons": "; ".join(
            str(item) for item in run_args.get("adaptive_trigger_reasons", [])
        ),
        "adaptive_baseline_val_f1": run_args.get("adaptive_baseline_val_f1"),
        "adaptive_train_size": run_args.get("adaptive_train_size"),
        "lambda_bias_before": run_args.get("adaptive_orig_lambda_bias"),
        "lambda_bias_after": run_args.get("lambda_bias"),
        "lambda_consist_before": run_args.get("adaptive_orig_lambda_consist"),
        "lambda_consist_after": run_args.get("lambda_consist"),
        "surface_feat_dim": run_args.get("surface_feat_dim"),
        "label_smoothing": run_args.get("label_smoothing"),
        "lambda_contrast": (
            run_args.get("lambda_contrast") if run_args.get("use_contrastive") else 0.0
        ),
        "lambda_fact_soft": run_args.get("lambda_fact_soft", 0.0),
        "alpha_floor": run_args.get("alpha_floor"),
        "soft_label_floor": run_args.get("soft_label_floor"),
        "source_file": str(path),
    }


def mean_std(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return float(np.mean(clean)), float(np.std(clean, ddof=0))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--suffixes", nargs="+", required=True)
    parser.add_argument("--expected_datasets", nargs="+", default=None)
    parser.add_argument("--expected_seeds", nargs="+", type=int, default=[42, 2024, 3407])
    parser.add_argument("--output", type=Path, required=True, help="mean/std summary CSV")
    parser.add_argument("--details", type=Path, default=None, help="run-level CSV")
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    wanted_suffixes = set(args.suffixes)
    wanted_seeds = set(args.expected_seeds)
    rows = []
    errors = []
    available_suffixes = Counter()
    candidate_paths = sorted(args.model_dir.glob("socialdebias_*_history.json"))
    for path in candidate_paths:
        try:
            row = load_row(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        available_suffixes[str(row["suffix"])] += 1
        if row["suffix"] in wanted_suffixes and row["seed"] in wanted_seeds:
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No matching histories in {args.model_dir.resolve()}; "
            f"suffixes={sorted(wanted_suffixes)}. The directory exists but "
            "contains no readable history with the requested save_suffix. "
            f"Scanned {len(candidate_paths)} candidate histories; "
            f"available suffix counts={dict(sorted(available_suffixes.items()))}; "
            f"unreadable={len(errors)}. Locate or complete the requested "
            "training outputs before aggregation."
        )

    rows.sort(key=lambda row: (row["dataset"], row["suffix"], int(row["seed"])))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["language"], row["suffix"])].append(row)

    missing_groups = []
    summary_rows = []
    for (dataset, language, suffix), runs in sorted(groups.items()):
        actual_seeds = {int(run["seed"]) for run in runs}
        missing = sorted(wanted_seeds - actual_seeds)
        duplicates = len(runs) != len(actual_seeds)
        if missing or duplicates:
            missing_groups.append(
                f"{dataset}/{suffix}: missing={missing}, duplicate_seed={duplicates}"
            )
        summary = {
            "dataset": dataset,
            "language": language,
            "suffix": suffix,
            "seeds": ",".join(str(seed) for seed in sorted(actual_seeds)),
            "n": len(runs),
        }
        for metric in METRICS:
            mean, std = mean_std([run.get(metric) for run in runs])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summary_rows.append(summary)

    if args.expected_datasets:
        present = {(row["dataset"], row["suffix"]) for row in rows}
        for dataset in args.expected_datasets:
            for suffix in args.suffixes:
                if (dataset, suffix) not in present:
                    missing_groups.append(f"{dataset}/{suffix}: group absent")

    details_path = args.details or args.output.with_name(
        args.output.stem + "_details" + args.output.suffix
    )
    write_csv(details_path, rows)
    write_csv(args.output, summary_rows)
    print(f"Run details: {details_path}")
    print(f"Mean/std summary (ddof=0): {args.output}")
    for row in summary_rows:
        print(
            f"{row['dataset']:<12} {row['suffix']:<24} n={row['n']} "
            f"test_f1={row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f}"
        )

    if errors:
        print("Unreadable histories:")
        for error in errors:
            print(f"  {error}")
    if missing_groups:
        print("Incomplete groups:")
        for item in missing_groups:
            print(f"  {item}")
        if args.require_complete:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
