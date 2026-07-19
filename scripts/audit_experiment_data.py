"""Audit experiment assets without printing text samples.

The report focuses on paths, schemas, counts, alignment, style coverage and
NLI/NRC readiness. It intentionally does not expose article contents.
"""
import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EMOTIONS = {
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
}


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def aligned_lengths(mapping, keys):
    lengths = {key: len(mapping[key]) for key in keys if key in mapping}
    return lengths, len(set(lengths.values())) <= 1


def inspect_base(path, dataframe=False):
    result = {"path": str(path), "exists": path.exists(), "kind": "base"}
    if not path.exists():
        return result
    data = load_pickle(path)
    if dataframe:
        result.update({
            "rows": len(data),
            "columns": [str(column) for column in data.columns],
            "schema_ok": {"content", "label"}.issubset(data.columns),
        })
    elif isinstance(data, dict):
        lengths, aligned = aligned_lengths(data, ["news", "labels"])
        result.update({
            "rows": lengths.get("news", 0), "keys": sorted(data),
            "lengths": lengths, "aligned": aligned,
            "schema_ok": {"news", "labels"}.issubset(data),
        })
    else:
        result.update({"schema_ok": False, "type": type(data).__name__})
    return result


def inspect_adversarial(
    path,
    require_p_entail=False,
    entity_threshold=None,
    semantic_threshold=None,
):
    result = {"path": str(path), "exists": path.exists(), "kind": "adversarial"}
    if not path.exists():
        return result
    data = load_pickle(path)
    if not isinstance(data, dict):
        result.update({"schema_ok": False, "type": type(data).__name__})
        return result
    required = {"news", "labels", "style", "orig_idx"}
    if require_p_entail:
        required.update({"p_entail", "p_neutral", "p_contradict", "nli_label"})
    if entity_threshold is not None:
        required.add("entity_recall_score")
    if semantic_threshold is not None:
        required.add("semantic_score")
    lengths, aligned = aligned_lengths(data, sorted(required | ({"status"} & set(data))))
    n = lengths.get("news", 0)
    statuses = Counter(data.get("status", []))
    success_positions = [
        index for index in range(n)
        if not data.get("status") or data["status"][index] == "success"
    ]
    by_origin = defaultdict(set)
    for index in success_positions:
        by_origin[int(data["orig_idx"][index])].add(str(data["style"][index]))
    style_counts = Counter(str(data["style"][index]) for index in success_positions)
    result.update({
        "rows": n,
        "keys": sorted(data),
        "lengths": lengths,
        "aligned": aligned,
        "schema_ok": required.issubset(data) and aligned and n > 0,
        "status_counts": dict(sorted(statuses.items())),
        "success_rows": len(success_positions),
        "success_unique_orig": len(by_origin),
        "success_style_counts": dict(sorted(style_counts.items())),
        "success_versions_per_orig": dict(sorted(Counter(map(len, by_origin.values())).items())),
    })
    if entity_threshold is not None and "entity_recall_score" in data:
        values = [float(value) for value in data["entity_recall_score"]]
        threshold_ok = bool(values) and all(
            math.isfinite(value) and value >= entity_threshold for value in values
        )
        result["entity_recall"] = {
            "threshold": entity_threshold,
            "min": min(values) if values else None,
            "threshold_ok": threshold_ok,
        }
        result["schema_ok"] = result["schema_ok"] and threshold_ok
    if semantic_threshold is not None and "semantic_score" in data:
        values = [float(value) for value in data["semantic_score"]]
        threshold_ok = bool(values) and all(
            math.isfinite(value) and value >= semantic_threshold for value in values
        )
        result["semantic"] = {
            "threshold": semantic_threshold,
            "min": min(values) if values else None,
            "threshold_ok": threshold_ok,
        }
        result["schema_ok"] = result["schema_ok"] and threshold_ok
    if require_p_entail and "p_entail" in data:
        values = [float(value) for value in data["p_entail"]]
        probability_keys = ("p_entail", "p_neutral", "p_contradict")
        probability_rows = list(zip(*(data.get(key, []) for key in probability_keys)))
        probabilities_ok = bool(values) and all(
            all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in row)
            and abs(sum(float(value) for value in row) - 1.0) <= 1e-4
            for row in probability_rows
        )
        contradiction_absent = (
            "nli_label" in data
            and all(str(label).lower() != "contradiction" for label in data["nli_label"])
            and data.get("nli_filter", {}).get("exclude") == "contradiction"
        )
        result["p_entail"] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "in_unit_interval": bool(values) and all(
                math.isfinite(value) and 0.0 <= value <= 1.0 for value in values
            ),
            "probabilities_sum_to_one": probabilities_ok,
            "contradiction_absent": contradiction_absent,
        }
        result["schema_ok"] = (
            result["schema_ok"] and probabilities_ok and contradiction_absent
        )
    return result


def inspect_nrc(path, language):
    result = {"path": str(path), "exists": path.exists(), "kind": "nrc", "language": language}
    if not path.exists():
        return result
    from utils.surface_features import SurfaceFeatureExtractor

    lexicon = SurfaceFeatureExtractor._load_lexicon(path)
    counts = Counter(emotion for labels in lexicon.values() for emotion in labels)
    result.update({
        "words": len(lexicon),
        "emotion_counts": dict(sorted(counts.items())),
        "all_eight_emotions": EMOTIONS.issubset(counts),
        "schema_ok": bool(lexicon) and EMOTIONS.issubset(counts),
    })
    return result


def inspect_paired_tests(dataset):
    clean_path = ROOT / f"data/sheepdog/news_articles/{dataset}_test.pkl"
    result = {
        "dataset": dataset, "path": f"paired:{dataset}",
        "clean": str(clean_path), "kind": "paired_test",
        "exists": clean_path.exists(),
    }
    if not clean_path.exists():
        result.update({"schema_ok": False, "error": "clean missing"})
        return result
    clean = load_pickle(clean_path)
    clean_labels = list(clean["labels"])
    variants = {}
    all_ok = True
    for variant in "ABCD":
        path = ROOT / f"data/sheepdog/adversarial_test/{dataset}_test_adv_{variant}.pkl"
        item = {"path": str(path), "exists": path.exists()}
        if path.exists():
            data = load_pickle(path)
            item.update({
                "rows": len(data.get("news", [])),
                "labels_match_clean": list(data.get("labels", [])) == clean_labels,
            })
        item_ok = item.get("exists", False) and item.get("labels_match_clean", False)
        all_ok = all_ok and item_ok
        variants[variant] = item
    result.update({"clean_rows": len(clean_labels), "variants": variants, "schema_ok": all_ok})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_output", default=None)
    parser.add_argument("--strict_training", action="store_true",
                        help="NRC 与三个 paper_v2 p_entail 缺失时返回非零")
    parser.add_argument("--strict_paper_v2", action="store_true",
                        help="三个 filtered 与三个 p_entail 输出无效时返回非零")
    args = parser.parse_args()

    assets = []
    assets.extend([
        inspect_base(ROOT / "data/sheepdog/news_articles/politifact_train.pkl"),
        inspect_base(ROOT / "data/sheepdog/news_articles/gossipcop_train.pkl"),
        inspect_base(ROOT / "data/weibo21_repo/data/train.pkl", dataframe=True),
    ])
    for name in (
        "politifact_train_adv.pkl",
        "gossipcop_train_adv.pkl",
        "weibo21_train_adv_deepseek.pkl",
    ):
        assets.append(inspect_adversarial(ROOT / "data/qwen_adv" / name))
    for dataset, entity_threshold in (
        ("politifact", 0.7),
        ("gossipcop", 0.7),
        ("weibo21", 0.6),
    ):
        assets.append(inspect_adversarial(
            ROOT / f"data/qwen_adv/{dataset}_train_adv_filtered_paper_v2.pkl",
            entity_threshold=entity_threshold,
            semantic_threshold=0.65,
        ))
    for dataset in ("politifact", "gossipcop", "weibo21"):
        entity_threshold = 0.6 if dataset == "weibo21" else 0.7
        assets.append(inspect_adversarial(
            ROOT / f"data/qwen_adv/{dataset}_p_entail_paper_v2.pkl",
            require_p_entail=True,
            entity_threshold=entity_threshold,
            semantic_threshold=0.65,
        ))
    assets.extend([
        inspect_nrc(
            ROOT / "data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", "en"
        ),
        inspect_nrc(ROOT / "data/lexicons/NRC-Emotion-Lexicon-ZH.tsv", "zh"),
        inspect_paired_tests("politifact"),
        inspect_paired_tests("gossipcop"),
    ])

    report = {"assets": assets}
    training_critical = [
        item for item in assets
        if (item["kind"] == "nrc" or "_p_entail_paper_v2.pkl" in item.get("path", ""))
    ]
    report["training_ready"] = all(
        item.get("exists", False) and item.get("schema_ok", False)
        for item in training_critical
    )
    paper_v2_outputs = [
        item for item in assets
        if "_filtered_paper_v2.pkl" in item.get("path", "")
        or "_p_entail_paper_v2.pkl" in item.get("path", "")
    ]
    report["paper_v2_data_ready"] = (
        len(paper_v2_outputs) == 6
        and all(
            item.get("exists", False) and item.get("schema_ok", False)
            for item in paper_v2_outputs
        )
    )

    for item in assets:
        state = "OK" if item.get("exists") and item.get("schema_ok") else "MISSING/INVALID"
        detail = ""
        if item.get("kind") == "adversarial" and item.get("exists"):
            detail = (f" rows={item.get('rows')} success={item.get('success_rows')}"
                      f" orig={item.get('success_unique_orig')}")
        elif item.get("kind") == "nrc" and item.get("exists"):
            detail = f" words={item.get('words')} eight={item.get('all_eight_emotions')}"
        print(f"[{state}] {item.get('path', item.get('dataset'))}{detail}")
    print(f"\npaper_v2_data_ready={report['paper_v2_data_ready']}")
    print(f"training_ready={report['training_ready']}")

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report={output}")

    if args.strict_training and not report["training_ready"]:
        raise SystemExit(1)
    if args.strict_paper_v2 and not report["paper_v2_data_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
