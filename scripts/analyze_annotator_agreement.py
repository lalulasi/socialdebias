"""Compute agreement between two independently completed blind task files."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from analyze_human_eval import normalize_judgment


def load(path):
    if str(path).endswith((".xlsx", ".xlsm")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator1", required=True)
    parser.add_argument("--annotator2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    first = load(args.annotator1)[["blind_id", "human_judgment"]].copy()
    second = load(args.annotator2)[["blind_id", "human_judgment"]].copy()
    first["judgment_1"] = first.pop("human_judgment").map(normalize_judgment)
    second["judgment_2"] = second.pop("human_judgment").map(normalize_judgment)
    merged = first.merge(second, on="blind_id", validate="one_to_one")
    valid = merged[
        merged["judgment_1"].isin(["real", "fake", "uncertain"])
        & merged["judgment_2"].isin(["real", "fake", "uncertain"])
    ]
    if valid.empty:
        raise ValueError("No comparable real/fake/uncertain judgments")

    result = {
        "n": int(len(valid)),
        "raw_agreement": float((valid["judgment_1"] == valid["judgment_2"]).mean()),
        "cohen_kappa": float(cohen_kappa_score(valid["judgment_1"], valid["judgment_2"])),
        "disagreements": int((valid["judgment_1"] != valid["judgment_2"]).sum()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    disagreement_path = output.with_name(output.stem + "_disagreements.csv")
    valid[valid["judgment_1"] != valid["judgment_2"]].to_csv(
        disagreement_path, index=False, encoding="utf-8-sig"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Disagreements: {disagreement_path}")


if __name__ == "__main__":
    main()
