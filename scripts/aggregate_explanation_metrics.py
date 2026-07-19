"""Aggregate three-seed explanation summaries with population std (ddof=0)."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--pattern", default="politifact_surface_all_seed*.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected_n", type=int, default=3)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob(args.pattern))
    if len(files) != args.expected_n:
        raise SystemExit(
            f"Expected {args.expected_n} files matching {args.pattern}, found {len(files)}"
        )

    detail = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        summary = payload.get("summary")
        if not summary:
            raise ValueError(f"Incomplete explanation result (no summary): {path}")
        seed = payload.get("args", {}).get("ckpt", path.stem)
        for model, metrics in summary.items():
            row = {"source_file": str(path), "seed_or_ckpt": seed, "model": model}
            for metric in ("top_k_overlap", "spearman", "js_divergence"):
                row[metric] = metrics[metric]["mean"]
                row[f"within_seed_{metric}_std"] = metrics[metric]["std"]
            detail.append(row)

    summary_rows = []
    for model in sorted({row["model"] for row in detail}):
        runs = [row for row in detail if row["model"] == model]
        out = {"model": model, "n_seeds": len(runs)}
        for metric in ("top_k_overlap", "spearman", "js_divergence"):
            values = [float(row[metric]) for row in runs]
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Explanation 3-seed summary: {args.output}")


if __name__ == "__main__":
    main()
