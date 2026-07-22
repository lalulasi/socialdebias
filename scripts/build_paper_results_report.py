#!/usr/bin/env python3
"""Build the final paper-v2 acceptance and legacy-comparison report.

The report is intentionally read-only with respect to experiment artifacts. It
collects P1-P11 outputs, checks required counts, aggregates single-seed JSONs
when needed, and renders Markdown organized by thesis chapter/table number.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SEEDS = {42, 2024, 3407}


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean_std(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return statistics.fmean(clean), statistics.pstdev(clean)


def fmt(value, digits=4):
    return "—" if value is None else f"{float(value):.{digits}f}"


def fmt_ms(mean, std, digits=4):
    if mean is None:
        return "—"
    if std is None:
        return fmt(mean, digits)
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def fmt_pct(value, digits=2):
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def fmt_pp(value, std=None):
    if value is None:
        return "—"
    if std is None:
        return f"{float(value) * 100:.2f}pp"
    return f"{float(value) * 100:.2f} ± {float(std) * 100:.2f}pp"


def fmt_delta(new, old, percentage_points=False):
    if new is None or old is None:
        return "—"
    delta = float(new) - float(old)
    if percentage_points:
        return f"{delta * 100:+.2f}pp"
    return f"{delta:+.4f}"


def cell(value):
    text = str(value).replace("|", "\\|").replace("\n", "<br>")
    return text


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(cell(item) for item in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(cell(item) for item in row) + " |" for row in rows
    )
    return lines


def dataset_from_args(args):
    if args.get("use_weibo21"):
        return "weibo21"
    return str(args.get("dataset", "unknown"))


def collect_histories(model_dir: Path):
    runs = []
    unreadable = []
    for path in sorted(model_dir.glob("socialdebias_*_history.json")):
        try:
            payload = read_json(path)
            run_args = payload.get("args", {})
            suffix = run_args.get("save_suffix")
            if not suffix and "bert_baseline" in path.name:
                suffix = "bert_baseline"
            best = payload.get("best_test") or payload.get("best_test_metrics") or {}
            runs.append({
                "dataset": dataset_from_args(run_args),
                "suffix": suffix or "unknown",
                "seed": int(run_args.get("seed")),
                "test_f1": number(best.get("f1")),
                "test_acc": number(best.get("acc", best.get("accuracy"))),
                "test_auc": number(best.get("auc")),
                "best_val_f1": number(payload.get("best_val_f1")),
                "path": str(path),
            })
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            unreadable.append(f"{path}: {exc}")
    return runs, unreadable


def summarize_histories(runs, suffixes=None):
    groups = defaultdict(list)
    for run in runs:
        if suffixes is None or run["suffix"] in suffixes:
            groups[(run["dataset"], run["suffix"])].append(run)
    rows = []
    for (dataset, suffix), items in sorted(groups.items()):
        f1_mean, f1_std = mean_std([item["test_f1"] for item in items])
        acc_mean, acc_std = mean_std([item["test_acc"] for item in items])
        rows.append({
            "dataset": dataset,
            "suffix": suffix,
            "n": len(items),
            "seeds": sorted(item["seed"] for item in items),
            "test_f1_mean": f1_mean,
            "test_f1_std": f1_std,
            "test_acc_mean": acc_mean,
            "test_acc_std": acc_std,
        })
    return rows


def collect_adv_group(paths, forced_model=None):
    groups = defaultdict(list)
    unreadable = []
    for path in sorted(paths):
        try:
            payload = read_json(path)
            dataset = str(payload.get("dataset") or path.name.split("_")[2])
            model = forced_model or str(payload.get("model", "unknown"))
            if model == "bert_baseline":
                model = "BERT"
            elif model == "bert":
                model = "ENDEF (bert)"
            elif model == "bert_endef":
                model = "ENDEF (bert_endef)"
            elif model in ("socialdebias", "SocialDebias"):
                model = "SocialDebias"
            seed = payload.get("seed")
            results = payload.get("results", {})
            summary = results.get("summary", {})
            clean_f1 = number(summary.get("clean_f1"))
            if clean_f1 is None:
                clean_f1 = number(results.get("clean", {}).get("f1"))
            groups[(dataset, model)].append({
                "seed": int(seed) if seed is not None else None,
                "clean_f1": clean_f1,
                "avg_adv_f1": number(summary.get("avg_adv_f1")),
                "f1_drop": number(summary.get("f1_drop")),
                "avg_asr": number(summary.get("avg_asr")),
                "path": str(path),
            })
        except (OSError, ValueError, TypeError, json.JSONDecodeError, IndexError) as exc:
            unreadable.append(f"{path}: {exc}")
    rows = []
    for (dataset, model), items in sorted(groups.items()):
        row = {"dataset": dataset, "method": model, "n": len(items)}
        for metric in ("clean_f1", "avg_adv_f1", "f1_drop", "avg_asr"):
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std(
                [item[metric] for item in items]
            )
        row["seeds"] = sorted(
            item["seed"] for item in items if item["seed"] is not None
        )
        rows.append(row)
    return rows, unreadable


def summarize_lstm(path: Path):
    groups = defaultdict(list)
    for row in read_csv(path):
        groups[row.get("dataset")].append(number(row.get("test_f1")))
    return {
        dataset: (*mean_std(values), len([value for value in values if value is not None]))
        for dataset, values in groups.items()
    }


def summarize_llm(path: Path):
    groups = defaultdict(dict)
    for row in read_csv(path):
        groups[row.get("dataset")][row.get("variant")] = row
    result = {}
    for dataset, variants in groups.items():
        clean = variants.get("clean", {})
        adv_rows = [variants.get(f"adv_{variant}", {}) for variant in "ABCD"]
        adv_f1 = [number(row.get("f1_macro")) for row in adv_rows]
        adv_f1 = [value for value in adv_f1 if value is not None]
        parse_fail = [number(row.get("parse_fail_rate")) for row in variants.values()]
        asrs = [number(row.get("asr")) for row in adv_rows]
        asrs = [value for value in asrs if value is not None]
        clean_f1 = number(clean.get("f1_macro"))
        avg_adv_f1 = statistics.fmean(adv_f1) if adv_f1 else None
        result[dataset] = {
            "clean_f1": clean_f1,
            "avg_adv_f1": avg_adv_f1,
            "f1_drop": clean_f1 - avg_adv_f1
            if clean_f1 is not None and avg_adv_f1 is not None else None,
            "avg_asr": statistics.fmean(asrs) if asrs else None,
            "max_parse_fail_rate": max(
                (value for value in parse_fail if value is not None), default=None
            ),
        }
    return result


def old_index(legacy):
    return {
        (row["dataset"], row["method"]): row
        for row in legacy.get("main_results", [])
    }


def current_main_rows(results_root, history_summary, legacy):
    rows = []
    lstm = summarize_lstm(results_root / "lstm_summary.csv")
    bert, _ = collect_adv_group((results_root / "bert_adv").glob("bert_adv_*.json"))
    surface, _ = collect_adv_group(
        (results_root / "surface_adv").glob("surface_adv_*_surface_all.json"),
        forced_model="SocialDebias",
    )
    endef, _ = collect_adv_group((results_root / "endef_adv").glob("endef_adv_*.json"))
    llm = summarize_llm(results_root / "llm_baseline" / "summary.csv")

    for dataset, values in lstm.items():
        rows.append({
            "dataset": dataset, "method": "BiLSTM", "n": values[2],
            "clean_f1_mean": values[0], "clean_f1_std": values[1],
        })
    rows.extend(bert)
    rows.extend(endef)
    rows.extend(surface)
    for dataset, values in llm.items():
        rows.append({
            "dataset": dataset, "method": "DeepSeek", "n": 1,
            "clean_f1_mean": values["clean_f1"], "clean_f1_std": None,
            "avg_adv_f1_mean": values["avg_adv_f1"], "avg_adv_f1_std": None,
            "f1_drop_mean": values["f1_drop"], "f1_drop_std": None,
            "avg_asr_mean": values["avg_asr"], "avg_asr_std": None,
            "parse_fail_rate": values["max_parse_fail_rate"],
        })

    present = {(row["dataset"], row["method"]) for row in rows}
    for item in history_summary:
        if item["suffix"] not in ("bert_baseline", "surface_all"):
            continue
        method = "BERT" if item["suffix"] == "bert_baseline" else "SocialDebias"
        key = (item["dataset"], method)
        if key not in present:
            rows.append({
                "dataset": item["dataset"], "method": method, "n": item["n"],
                "clean_f1_mean": item["test_f1_mean"],
                "clean_f1_std": item["test_f1_std"],
            })

    old = old_index(legacy)
    output = []
    for row in sorted(rows, key=lambda item: (item["dataset"], item["method"])):
        reference = old.get((row["dataset"], row["method"]), {})
        same_config = True
        note = ""
        if row["method"] == "SocialDebias" and row["dataset"] == "politifact":
            same_config = reference.get("config") == "surface_all"
            note = "旧表为 surface，新表为统一 surface_all；仅方向参考"
        elif reference:
            note = "旧实现与 paper-v2 代码口径不同"
        output.append({
            **row,
            "old_clean_f1": reference.get("clean_f1"),
            "old_avg_adv_f1": reference.get("avg_adv_f1"),
            "old_f1_drop": reference.get("f1_drop"),
            "old_avg_asr": reference.get("avg_asr"),
            "same_config": same_config,
            "note": note,
        })
    return output


def add_check(checks, stage, item, actual, expected, path, required=True):
    ok = actual == expected
    checks.append({
        "stage": stage, "item": item, "actual": actual, "expected": expected,
        "ok": ok, "required": required, "path": str(path),
    })


def add_file_check(checks, stage, item, path, required=True):
    checks.append({
        "stage": stage, "item": item, "actual": "present" if path.is_file() and path.stat().st_size else "missing",
        "expected": "present", "ok": path.is_file() and path.stat().st_size > 0,
        "required": required, "path": str(path),
    })


def add_boolean_check(checks, stage, item, ok, actual, expected, path, required=True):
    checks.append({
        "stage": stage, "item": item, "actual": actual, "expected": expected,
        "ok": bool(ok), "required": required, "path": str(path),
    })


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_socialdebias_adv_release(project_root: Path, results_root: Path):
    """Validate the published filtered dataset as an alternative to raw API files."""
    filtered_dir = project_root / "data/socialdebias_adv/filtered"
    filter_report = filtered_dir / "filter_report.csv"
    manifest_dir = results_root / "manifests"
    commit_file = manifest_dir / "socialdebias_adv_dataset_commit.txt"
    hash_file = manifest_dir / "socialdebias_adv_dataset_sha256.txt"

    datasets = ("politifact", "gossipcop", "weibo21")
    tones = ("neutral", "objective", "sensational", "emotionally_triggering")
    sources = ("qwen", "deepseek")
    expected_names = {
        f"{dataset}_test_adv_{tone}_{source}.pkl"
        for dataset in datasets
        for tone in tones
        for source in sources
    }

    filtered_files = {
        path.name: path for path in filtered_dir.glob("*_test_adv_*.pkl")
        if path.is_file() and path.stat().st_size > 0
    }
    names_ok = set(filtered_files) == expected_names

    report_rows = read_csv(filter_report)
    report_names = {row.get("file", "") for row in report_rows}
    report_ok = len(report_rows) == 24 and report_names == expected_names

    commit_ok = False
    if commit_file.is_file():
        commit_text = commit_file.read_text(encoding="utf-8").strip()
        commit_ok = re.fullmatch(r"[0-9a-fA-F]{40,64}", commit_text) is not None

    recorded_hashes = {}
    if hash_file.is_file():
        for line in hash_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2 or re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]) is None:
                continue
            recorded_hashes[Path(parts[1].lstrip("* ")).name] = parts[0].lower()

    artifacts = dict(filtered_files)
    if filter_report.is_file() and filter_report.stat().st_size > 0:
        artifacts[filter_report.name] = filter_report
    expected_hash_names = expected_names | {"filter_report.csv"}
    hashes_ok = set(recorded_hashes) == expected_hash_names
    if hashes_ok:
        hashes_ok = all(
            recorded_hashes[name] == file_sha256(artifacts[name])
            for name in expected_hash_names
        )

    ok = names_ok and report_ok and commit_ok and hashes_ok
    actual = (
        f"filtered={len(filtered_files)}/24, report={len(report_rows)}/24, "
        f"commit={'yes' if commit_ok else 'no'}, "
        f"sha256={'25/25' if hashes_ok else 'invalid'}"
    )
    return ok, actual, manifest_dir


def build_checks(project_root, results_root, endef_root):
    checks = []
    p1_files = [
        project_root / f"data/qwen_adv/{dataset}_{suffix}_paper_v2.pkl"
        for dataset in ("politifact", "gossipcop", "weibo21")
        for suffix in ("train_adv_filtered", "p_entail")
    ]
    add_check(checks, "P1", "filtered + p_entail 数据", sum(path.is_file() and path.stat().st_size > 0 for path in p1_files), 6, project_root / "data/qwen_adv")
    add_file_check(checks, "P1", "最终数据审计", results_root / "data_preparation/after.json")
    specs = [
        ("P2", "LSTM result", results_root / "lstm", "lstm_*_result.json", 6),
        ("P2", "BERT checkpoint", results_root / "models", "*_bert_baseline.pt", 9),
        ("P2", "BERT history", results_root / "models", "*_bert_baseline_history.json", 9),
        ("P2", "BERT 对抗 JSON", results_root / "bert_adv", "bert_adv_*.json", 6),
        ("P2", "DeepSeek JSON", results_root / "llm_baseline", "*.json", 11),
        ("P2", "ENDEF 英文对抗 JSON", results_root / "endef_adv", "endef_adv_*.json", 12),
        ("P3", "surface_all checkpoint", results_root / "models", "*_surface_all.pt", 9),
        ("P3", "surface_all history", results_root / "models", "*_surface_all_history.json", 9),
        ("P4", "surface_all 对抗 JSON", results_root / "surface_adv", "*_surface_all.json", 6),
        ("P5", "SocialDebias-Adv 数据来源", project_root / "data/socialdebias_adv", "*_test_adv_*.pkl", 24),
        ("P5", "SocialDebias-Adv 过滤集", project_root / "data/socialdebias_adv/filtered", "*.pkl", 24),
        ("P6", "IG 三 seed JSON", results_root / "explanation", "politifact_surface_all_seed*.json", 3),
        ("P7", "消融 history", results_root / "models", "*_abl_*_history.json", 42),
        ("P7", "消融对抗 JSON", results_root / "ablation_adv", "ablation_adv_*_abl_*.json", 42),
        ("P8", "17 维 history", results_root / "models", "*_surface17_full_history.json", 3),
        ("P8", "17 维对抗 JSON", results_root / "ablation_adv", "*_surface17_full_seed*.json", 3),
        ("P9", "NLI history", results_root / "models", "*_nli_*_history.json", 45),
        ("P10", "Adaptive history", results_root / "models", "*_surface_fixed_history.json", 9),
        ("P10", "Adaptive history", results_root / "models", "*_surface_adaptive_history.json", 9),
        ("P10", "Adaptive 对抗 JSON", results_root / "surface_adv", "*_surface_fixed.json", 6),
        ("P10", "Adaptive 对抗 JSON", results_root / "surface_adv", "*_surface_adaptive.json", 6),
    ]
    for stage, item, directory, pattern, expected in specs:
        add_check(checks, stage, item, len(list(directory.glob(pattern))), expected, directory / pattern)

    raw_dir = project_root / "data/socialdebias_adv"
    raw_count = sum(
        path.is_file() and path.stat().st_size > 0
        for path in raw_dir.glob("*_test_adv_*.pkl")
    )
    release_ok, release_actual, release_path = validate_socialdebias_adv_release(
        project_root, results_root
    )
    source_check = next(
        check for check in checks
        if check["stage"] == "P5" and check["item"] == "SocialDebias-Adv 数据来源"
    )
    if raw_count == 24:
        source_check.update({
            "actual": "raw=24/24",
            "expected": "raw=24/24 或可验证发布集=24/24",
            "ok": True,
            "path": str(raw_dir / "*_test_adv_*.pkl"),
        })
    else:
        source_check.update({
            "actual": f"raw={raw_count}/24; published release: {release_actual}",
            "expected": "raw=24/24 或可验证发布集=24/24",
            "ok": release_ok,
            "path": str(release_path),
        })
    if endef_root:
        specs = [
            ("ENDEF English backup", endef_root / "ENDEF_en/backup_ckpt", 12),
            ("ENDEF Chinese backup", endef_root / "ENDEF_ch/backup_ckpt", 6),
        ]
        for item, directory, expected in specs:
            add_check(checks, "P2", item, len(list(directory.glob("*.pkl"))), expected, directory)
    required_files = [
        ("P2", "LSTM summary", results_root / "lstm_summary.csv"),
        ("P2", "DeepSeek summary", results_root / "llm_baseline/summary.csv"),
        ("P3", "surface_all clean summary", results_root / "surface_all_clean_summary.csv"),
        ("P4", "主结果 summary", results_root / "main_summary.csv"),
        ("P5", "过滤报告", project_root / "data/socialdebias_adv/filtered/filter_report.csv"),
        ("P5", "三方评估", results_root / "socialdebias_adv_eval.csv"),
        ("P6", "IG summary", results_root / "explanation/explanation_3seed_summary.csv"),
        ("P7", "消融 clean summary", results_root / "ablation_clean_summary.csv"),
        ("P7", "消融 adv summary", results_root / "ablation_adv/ablation_adv_summary.csv"),
        ("P8", "8/17 clean summary", results_root / "surface_8_vs_17_clean_summary.csv"),
        ("P9", "NLI summary", results_root / "nli_mechanism_summary.csv"),
        ("P10", "Adaptive clean summary", results_root / "adaptive_clean_summary.csv"),
        ("P10", "fixed adv summary", results_root / "surface_fixed_adv_summary.csv"),
        ("P10", "adaptive adv summary", results_root / "surface_adaptive_adv_summary.csv"),
        ("P11", "人工评估 manifest", results_root / "human_eval/final/human_eval_input_manifest.json"),
        ("P11", "人工评估 metrics", results_root / "human_eval/final/human_eval_metrics.json"),
        ("P11", "人工评估逐样本", results_root / "human_eval/final/human_eval_per_sample.csv"),
        ("P11", "人工-模型对齐", results_root / "human_eval/final/human_eval_model_alignment.csv"),
    ]
    for stage, item, path in required_files:
        add_file_check(checks, stage, item, path)

    audit_path = results_root / "data_preparation/after.json"
    if audit_path.exists():
        try:
            audit = read_json(audit_path)
            ready = audit.get("paper_v2_data_ready") is True and audit.get("training_ready") is True
            add_boolean_check(
                checks, "P1", "数据审计 ready 标志", ready,
                f"paper_v2={audit.get('paper_v2_data_ready')}, training={audit.get('training_ready')}",
                "both True", audit_path,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            add_boolean_check(checks, "P1", "数据审计 ready 标志", False, str(exc), "both True", audit_path)

    summary_specs = [
        ("P3", "surface_all summary 三 seed", results_root / "surface_all_clean_summary.csv", "n"),
        ("P4", "主结果 summary 三 seed", results_root / "main_summary.csv", "n"),
        ("P6", "IG summary 三 seed", results_root / "explanation/explanation_3seed_summary.csv", "n_seeds"),
        ("P7", "消融 clean summary 三 seed", results_root / "ablation_clean_summary.csv", "n"),
        ("P7", "消融 adv summary 三 seed", results_root / "ablation_adv/ablation_adv_summary.csv", "n"),
        ("P8", "8/17 summary 三 seed", results_root / "surface_8_vs_17_clean_summary.csv", "n"),
        ("P9", "NLI summary 三 seed", results_root / "nli_mechanism_summary.csv", "n"),
        ("P10", "Adaptive clean summary 三 seed", results_root / "adaptive_clean_summary.csv", "n"),
        ("P10", "fixed adv summary 三 seed", results_root / "surface_fixed_adv_summary.csv", "n"),
        ("P10", "adaptive adv summary 三 seed", results_root / "surface_adaptive_adv_summary.csv", "n"),
    ]
    for stage, item, path, field in summary_specs:
        rows = read_csv(path)
        valid = [row for row in rows if number(row.get(field)) == 3]
        add_boolean_check(
            checks, stage, item, bool(rows) and len(valid) == len(rows),
            f"{len(valid)}/{len(rows)} rows", "all rows n=3", path,
        )

    social_path = results_root / "socialdebias_adv_eval.csv"
    social_rows = [
        row for row in read_csv(social_path)
        if row.get("file") == "__dataset_summary__" and row.get("seed") in ("mean", "single")
    ]
    social_groups = {(row.get("dataset"), row.get("method")) for row in social_rows}
    expected_social = {
        (dataset, method)
        for dataset in ("politifact", "gossipcop", "weibo21")
        for method in ("BERT", "DeepSeek", "SocialDebias")
    }
    add_boolean_check(
        checks, "P5", "三方数据集汇总覆盖", social_groups == expected_social,
        f"{len(social_groups)}/9 groups", "9 groups", social_path,
    )

    human_manifest = results_root / "human_eval/final/human_eval_input_manifest.json"
    if human_manifest.exists():
        try:
            payload = read_json(human_manifest)
            add_boolean_check(
                checks, "P11", "人工评估 50 对", payload.get("rows") == 100 and payload.get("pairs") == 50,
                f"rows={payload.get('rows')}, pairs={payload.get('pairs')}",
                "rows=100, pairs=50", human_manifest,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            add_boolean_check(checks, "P11", "人工评估 50 对", False, str(exc), "rows=100, pairs=50", human_manifest)
    alignment_path = results_root / "human_eval/final/human_eval_model_alignment.csv"
    alignment_rows = read_csv(alignment_path)
    add_boolean_check(
        checks, "P11", "人工-IG 对齐覆盖", len(alignment_rows) == 50,
        len(alignment_rows), 50, alignment_path,
    )
    return checks


def csv_section(lines, title, path, columns, rename=None, predicate=None):
    lines.append(f"### {title}")
    lines.append("")
    rows = read_csv(path)
    if predicate:
        rows = [row for row in rows if predicate(row)]
    if not rows:
        lines.append(f"> 缺失或无可用记录：`{path}`")
        lines.append("")
        return []
    rename = rename or {}
    def display(row, column):
        value = row.get(column, "")
        parsed = number(value)
        if parsed is None:
            return value
        if column == "n":
            return str(int(parsed))
        return fmt(parsed)

    table_rows = [[display(row, column) for column in columns] for row in rows]
    lines.extend(markdown_table([rename.get(column, column) for column in columns], table_rows))
    lines.append("")
    lines.append(f"来源：`{path}`")
    lines.append("")
    return rows


def summarize_social_adv_detail(rows, group_keys, metric, weighted=False):
    """Aggregate per-file SocialDebias rows into per-seed and three-seed results."""
    per_seed = defaultdict(list)
    for row in rows:
        if row.get("file") == "__dataset_summary__":
            continue
        if row.get("method") != "SocialDebias":
            continue
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            continue
        value = number(row.get(metric))
        weight = number(row.get("n"))
        if value is None:
            continue
        key = tuple(row.get(field, "") for field in group_keys)
        per_seed[(key, seed)].append((value, weight))

    grouped = defaultdict(list)
    for (key, _seed), values in per_seed.items():
        if weighted and all(weight is not None and weight > 0 for _, weight in values):
            total_weight = sum(weight for _, weight in values)
            seed_value = sum(value * weight for value, weight in values) / total_weight
        else:
            seed_value = statistics.fmean(value for value, _ in values)
        grouped[key].append(seed_value)

    result = {}
    for key, values in grouped.items():
        mean, std = mean_std(values)
        result[key] = {"mean": mean, "std": std, "n_seeds": len(values)}
    return result


def build_artifact_mapping(project_root: Path, results_root: Path):
    """Single source of truth for artifact-to-thesis-section/table mapping."""
    return [
        ["P0", "§5.2.1", "表5-1",
         "PolitiFact、GossipCop、Weibo21 与 SocialDebias-Adv 的数据划分和规模",
         project_root / "data", "属于实验口径核对；不与结果指标混合比较"],
        ["P0", "§5.2.2–§5.2.4、附录A.2–A.3", "表5-2",
         "基线定义、评估指标、随机种子、训练超参数、软硬件环境",
         results_root / "manifests", "核对新旧配置差异；不计算性能差值"],
        ["P1", "§4.3.1–§4.4.2", "表4-3、表5-12、图5.4",
         "训练对抗集三级过滤、NLI p_entail、双重软标签机制",
         project_root / "data/qwen_adv", "旧表4-3/5-12；同配置项计算差值"],
        ["P2", "§5.2.2、§5.3.1", "表5-2～表5-5、图5.1～5.2",
         "BiLSTM、BERT、DeepSeek、ENDEF 基线的 Clean/Adv/F1 Drop/ASR",
         results_root / "lstm_summary.csv", "旧表5-3～5-5；方法与数据切分不同时标注不可直接比较"],
        ["P3", "§3.4–§3.8、§5.3.1", "表5-3～表5-5、图5.1～5.2",
         "统一 surface_all 主模型在三数据集上的三随机种子 Clean 结果",
         results_root / "surface_all_clean_summary.csv", "旧论文 surface/surface_all 混用；新表统一使用 surface_all"],
        ["P4", "§5.3.1", "表5-3、表5-4、图5.1",
         "PolitiFact/GossipCop 的 SheepDog adv_A/B/C/D 对抗鲁棒性",
         results_root / "main_summary.csv", "旧表5-3/5-4；比较 Clean、Avg Adv、Drop、ASR"],
        ["P5", "§4.2.1", "表4-1",
         "Qwen 与 DeepSeek 两类攻击源的非对称攻击强度",
         results_root / "socialdebias_adv_eval.csv", "按数据集和攻击源重算 SocialDebias Accuracy"],
        ["P5", "§4.2.2、附录B", "表4-2",
         "四风格×双语提示词、API采样参数、长度限制与输出清洗配置",
         project_root / "run_socialdebias_adv.sh", "属于生成配置核对；不能用新模型名配旧生成结果"],
        ["P5", "§4.5.1", "表4-4、图4.2",
         "SocialDebias-Adv 生成量、过滤后保留量与保留率",
         project_root / "data/socialdebias_adv/filtered/filter_report.csv", "旧表4-4；比较各数据集保留数量和总保留率"],
        ["P5", "§4.5.2", "表4-5",
         "不同 LLM 来源与四种改写风格的质量过滤保留率",
         project_root / "data/socialdebias_adv/filtered/filter_report.csv", "旧表4-5；比较来源/风格保留率"],
        ["P5", "§5.3.2", "表5-6",
         "四种 tone 攻击下 SocialDebias 的 ASR 与最强攻击风格",
         results_root / "socialdebias_adv_eval.csv", "旧表5-6；逐数据集、逐风格比较 ASR"],
        ["P5", "§5.3.3", "表5-7",
         "BERT、DeepSeek、SocialDebias 在自建对抗集上的三方评估",
         results_root / "socialdebias_adv_eval.csv", "旧表5-7；比较 Accuracy/F1/ASR"],
        ["P6", "§5.4.1–§5.4.3", "表5-8、图5.3",
         "IG 的 Top-10 重合度、Spearman 相关与 JS 散度（三随机种子）",
         results_root / "explanation/explanation_3seed_summary.csv", "旧表5-8基线实现有误；新表整体替换"],
        ["P7", "§5.5.1", "表5-10",
         "GRL、InfoNCE、Consist、表层特征、对抗增强、标签平滑消融",
         results_root / "ablation_adv/ablation_adv_summary.csv", "旧表5-10；仅同名且同开关配置可直接比较"],
        ["P8", "§3.3.1、§5.5.1", "表5-11",
         "8维情绪特征与17维情绪+句法+词汇特征对照",
         results_root / "surface_8_vs_17_clean_summary.csv", "旧表5-11；比较 Clean F1 与对抗 Drop"],
        ["P9", "§4.4.1、§5.5.2", "表4-3、表5-12、图5.4",
         "NLI 硬标签、分类软化、对比加权、floor 与 Soft14 机制拆分",
         results_root / "nli_mechanism_summary.csv", "旧论文仅 Soft14 可严格对应；其余为本轮新增机制拆分"],
        ["P10", "§3.8.3、§5.5.3", "表5-13",
         "固定 λ 与自适应 λ 的跨数据集 Clean/Adv/触发行为",
         results_root / "adaptive_clean_summary.csv", "旧表5-13；比较 Clean、Avg Adv 和 F1 Drop"],
        ["P11", "§5.4.4", "表5-9、附录C",
         "50对人工真假判断、置信度、Human ASR、关键词稳定性及 IG 对齐",
         results_root / "human_eval/final/human_eval_metrics.json", "旧表5-9和附录C；按既有单标注者口径比较"],
        ["P12", "§5.6、§6.1–§6.3、摘要与创新点", "无独立表",
         "根据全部新结果重写实验结论、适用边界、不足和未来工作",
         results_root / "论文实验结果总验收与旧论文对照.md", "只引用本轮验收通过结果，不混合旧种子"],
    ]


def build_report(project_root, results_root, legacy, endef_root):
    checks = build_checks(project_root, results_root, endef_root)
    histories, unreadable_histories = collect_histories(results_root / "models")
    history_summary = summarize_histories(histories)
    main_rows = current_main_rows(results_root, history_summary, legacy)
    passed = sum(check["ok"] for check in checks if check["required"])
    total = sum(check["required"] for check in checks)
    overall = passed == total
    artifact_mapping = build_artifact_mapping(project_root, results_root)

    lines = [
        "# SocialDebias paper-v2 完整实验验收与旧论文对照",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        f"项目目录：`{project_root}`  ",
        f"结果目录：`{results_root}`  ",
        f"旧结果来源：{legacy.get('source', 'unknown')}  ",
        "",
        f"**总验收：{'通过' if overall else '未通过'}（{passed}/{total} 项）**",
        "",
        f"> {legacy.get('warning', '')}",
        "",
        "## 一、实验数据与论文章节、表格总索引",
        "",
    ]
    lines.extend(markdown_table(
        ["阶段", "论文章节", "表格/图", "实验内容", "本轮数据来源", "与旧论文对比口径"],
        [[stage, section, table, content, f"`{source}`", comparison]
         for stage, section, table, content, source, comparison in artifact_mapping],
    ))
    lines.extend([
        "",
        "> 使用顺序：先看本索引定位论文位置，再看下一节验收状态，最后到第4章/第5章分表读取本轮值、旧值和差值。",
        "",
        "## 二、全实验产物验收（P1–P11）",
        "",
    ])
    lines.extend(markdown_table(
        ["阶段", "验收项", "实际", "期望", "状态", "路径"],
        [[
            check["stage"], check["item"], check["actual"], check["expected"],
            "✅" if check["ok"] else "❌", f"`{check['path']}`",
        ] for check in checks],
    ))
    lines.append("")
    if unreadable_histories:
        lines.append("无法读取的 history：")
        lines.extend(f"- `{item}`" for item in unreadable_histories)
        lines.append("")

    lines.extend(["## 三、第 4 章：对抗数据与 NLI 机制", ""])

    social_detail_rows = read_csv(results_root / "socialdebias_adv_eval.csv")
    source_acc = summarize_social_adv_detail(
        social_detail_rows, ("dataset", "source"), "acc", weighted=True
    )
    old_source_acc = {
        (row["dataset"], row["source"]): row
        for row in legacy.get("table_4_1", [])
    }
    lines.append("### P5 / §4.2.1 / 表 4-1：异构攻击源强度")
    lines.append("")
    source_rendered = []
    for dataset in ("politifact", "gossipcop", "weibo21"):
        for source in ("qwen", "deepseek"):
            current = source_acc.get((dataset, source), {})
            old = old_source_acc.get((dataset, source), {}).get("acc")
            source_rendered.append([
                dataset, source, current.get("n_seeds", 0),
                fmt_ms(current.get("mean"), current.get("std")),
                fmt(old), fmt_delta(current.get("mean"), old),
            ])
    lines.extend(markdown_table(
        ["数据集", "攻击源", "n", "本轮 Accuracy", "旧论文 Accuracy", "差值"],
        source_rendered,
    ))
    lines.extend(["", f"来源：`{results_root / 'socialdebias_adv_eval.csv'}`", ""])

    lines.append("### P9 / §4.4.1 / 表 4-3：NLI 双软标签与 floor 机制")
    lines.append("")
    nli_rows = read_csv(results_root / "nli_mechanism_summary.csv")
    old_nli = {(row["dataset"], row["method"]): row for row in legacy.get("table_4_3", [])}
    rendered = []
    for row in nli_rows:
        current = number(row.get("test_f1_mean"))
        old = old_nli.get((row.get("dataset"), row.get("suffix")), {}).get("test_f1")
        rendered.append([
            row.get("dataset"), row.get("suffix"), row.get("n"),
            fmt_ms(current, number(row.get("test_f1_std"))), fmt(old),
            fmt_delta(current, old),
        ])
    if rendered:
        lines.extend(markdown_table(
            ["数据集", "配置", "n", "本轮 Test F1", "旧论文 F1", "差值"], rendered
        ))
    else:
        lines.append(f"> 缺失：`{results_root / 'nli_mechanism_summary.csv'}`")
    lines.extend(["", "旧论文只保留了 `nli_soft14` 的可比值；其余四项是本轮新增的可审计机制拆分。", ""])

    lines.append("### P5 / §4.5.1 / 表 4-4：SocialDebias-Adv 三级过滤")
    lines.append("")
    filter_rows = read_csv(project_root / "data/socialdebias_adv/filtered/filter_report.csv")
    filter_groups = defaultdict(lambda: [0, 0])
    for row in filter_rows:
        name = row.get("file", "")
        dataset = next((item for item in ("politifact", "gossipcop", "weibo21") if item in name), "unknown")
        filter_groups[dataset][0] += int(float(row.get("n_in", 0)))
        filter_groups[dataset][1] += int(float(row.get("n_out", 0)))
    old_filter = {row["dataset"]: row for row in legacy.get("table_4_4", [])}
    filter_rendered = []
    total_in = total_out = 0
    for dataset in ("politifact", "gossipcop", "weibo21"):
        n_in, n_out = filter_groups.get(dataset, (0, 0))
        total_in += n_in; total_out += n_out
        rate = n_out / n_in if n_in else None
        old_out = old_filter.get(dataset, {}).get("retained")
        filter_rendered.append([dataset, n_in or "—", n_out or "—", fmt_pct(rate), old_out or "—", "—" if not n_out or old_out is None else f"{n_out-old_out:+d}"])
    old_all = old_filter.get("all", {})
    filter_rendered.append(["合计", total_in or "—", total_out or "—", fmt_pct(total_out / total_in if total_in else None), old_all.get("retained", "—"), "—" if not total_out else f"{total_out-int(old_all.get('retained', 0)):+d}"])
    lines.extend(markdown_table(["数据集", "生成量", "本轮保留", "本轮保留率", "旧论文保留", "数量差"], filter_rendered))
    lines.extend(["", f"来源：`{project_root / 'data/socialdebias_adv/filtered/filter_report.csv'}`", ""])

    lines.append("### P5 / §4.5.2 / 表 4-5：不同来源与风格的过滤保留率")
    lines.append("")
    quality_groups = defaultdict(lambda: [0, 0])
    for row in filter_rows:
        name = row.get("file", "")
        source = "qwen" if "qwen" in name else "deepseek" if "deepseek" in name else "unknown"
        tone = next((item for item in (
            "emotionally_triggering", "sensational", "objective", "neutral"
        ) if item in name), "unknown")
        n_in = int(float(row.get("n_in", 0)))
        n_out = int(float(row.get("n_out", 0)))
        quality_groups[("LLM来源", source)][0] += n_in
        quality_groups[("LLM来源", source)][1] += n_out
        quality_groups[("改写风格", tone)][0] += n_in
        quality_groups[("改写风格", tone)][1] += n_out
    old_quality = {
        (row["dimension"], row["category"]): row
        for row in legacy.get("table_4_5", [])
    }
    quality_rendered = []
    for dimension, categories in (
        ("LLM来源", ("deepseek", "qwen")),
        ("改写风格", ("neutral", "objective", "sensational", "emotionally_triggering")),
    ):
        for category in categories:
            n_in, n_out = quality_groups.get((dimension, category), (0, 0))
            current_rate = n_out / n_in if n_in else None
            old_rate = old_quality.get((dimension, category), {}).get("keep_rate")
            quality_rendered.append([
                dimension, category, n_in or "—", n_out or "—",
                fmt_pct(current_rate), fmt_pct(old_rate),
                fmt_delta(current_rate, old_rate, percentage_points=True),
            ])
    lines.extend(markdown_table(
        ["分析维度", "类别", "生成量", "保留量", "本轮保留率", "旧论文保留率", "差值"],
        quality_rendered,
    ))
    lines.extend(["", f"来源：`{project_root / 'data/socialdebias_adv/filtered/filter_report.csv'}`", ""])

    lines.extend(["## 四、第 5 章：主实验与机制实验", ""])
    for dataset, table in (("politifact", "表 5-3"), ("gossipcop", "表 5-4"), ("weibo21", "表 5-5")):
        lines.append(f"### P2–P4 / §5.3.1 / {table}：{dataset} 主结果")
        lines.append("")
        selected = [row for row in main_rows if row["dataset"] == dataset]
        table_rows = []
        for row in selected:
            table_rows.append([
                row["method"], row.get("n", "—"),
                fmt_ms(row.get("clean_f1_mean"), row.get("clean_f1_std")),
                fmt(row.get("old_clean_f1")), fmt_delta(row.get("clean_f1_mean"), row.get("old_clean_f1")),
                fmt_ms(row.get("avg_adv_f1_mean"), row.get("avg_adv_f1_std")),
                fmt(row.get("old_avg_adv_f1")),
                fmt_pp(row.get("f1_drop_mean"), row.get("f1_drop_std")),
                fmt_pp(row.get("old_f1_drop")),
                fmt_pct(row.get("avg_asr_mean")), row.get("note", ""),
            ])
        if table_rows:
            lines.extend(markdown_table(
                ["方法", "n", "本轮 Clean F1", "旧 Clean", "ΔClean", "本轮 Avg Adv F1", "旧 Avg Adv", "本轮 Drop", "旧 Drop", "本轮 ASR", "说明"],
                table_rows,
            ))
        else:
            lines.append("> 没有可用的本轮主结果。")
        lines.append("")

    tone_asr = summarize_social_adv_detail(
        social_detail_rows, ("dataset", "tone"), "asr", weighted=False
    )
    old_tone_asr = {
        (row["dataset"], row["tone"]): row
        for row in legacy.get("table_5_6", [])
    }
    lines.append("### P5 / §5.3.2 / 表 5-6：四种改写风格的 ASR")
    lines.append("")
    tone_rendered = []
    for dataset in ("politifact", "gossipcop", "weibo21"):
        dataset_values = []
        for tone in ("neutral", "objective", "sensational", "emotionally_triggering"):
            current = tone_asr.get((dataset, tone), {})
            old = old_tone_asr.get((dataset, tone), {}).get("asr")
            current_mean = current.get("mean")
            if current_mean is not None:
                dataset_values.append((tone, current_mean))
            tone_rendered.append([
                dataset, tone, current.get("n_seeds", 0),
                fmt_pct(current_mean), fmt_pct(old),
                fmt_delta(current_mean, old, percentage_points=True),
            ])
        if dataset_values:
            strongest = max(dataset_values, key=lambda item: item[1])[0]
            tone_rendered.append([dataset, "本轮最强攻击", "—", strongest, "—", "—"])
    lines.extend(markdown_table(
        ["数据集", "风格", "n", "本轮 ASR", "旧论文 ASR", "差值"],
        tone_rendered,
    ))
    lines.extend(["", f"来源：`{results_root / 'socialdebias_adv_eval.csv'}`", ""])

    lines.append("### P5 / §5.3.3 / 表 5-7：SocialDebias-Adv 三方评估")
    lines.append("")
    social_rows = [
        row for row in read_csv(results_root / "socialdebias_adv_eval.csv")
        if row.get("file") == "__dataset_summary__" and row.get("seed") in ("mean", "single")
    ]
    old_social = {(row["dataset"], row["method"]): row for row in legacy.get("table_5_7", [])}
    social_rendered = []
    for row in sorted(social_rows, key=lambda item: (item.get("dataset", ""), item.get("method", ""))):
        acc = number(row.get("acc")); old = old_social.get((row.get("dataset"), row.get("method")), {}).get("acc")
        social_rendered.append([
            row.get("dataset"), row.get("method"), row.get("seed"),
            fmt_ms(acc, number(row.get("acc_std"))), fmt(old), fmt_delta(acc, old),
            fmt_ms(number(row.get("f1")), number(row.get("f1_std"))),
            fmt_pct(number(row.get("asr"))),
        ])
    if social_rendered:
        lines.extend(markdown_table(["数据集", "方法", "seed", "本轮 Acc", "旧 Acc", "差值", "本轮 F1", "本轮 ASR"], social_rendered))
    else:
        lines.append(f"> 缺失数据集汇总行：`{results_root / 'socialdebias_adv_eval.csv'}`")
    lines.append("")

    lines.append("### P6 / §5.4.3 / 表 5-8：三 seed IG 解释一致性")
    lines.append("")
    explanation_rows = read_csv(results_root / "explanation/explanation_3seed_summary.csv")
    old_expl = {row["model"]: row for row in legacy.get("table_5_8", [])}
    expl_rendered = []
    for row in explanation_rows:
        model = row.get("model")
        old = old_expl.get(model, {})
        expl_rendered.append([
            model, row.get("n_seeds"),
            fmt_ms(number(row.get("top_k_overlap_mean")), number(row.get("top_k_overlap_std"))), fmt(old.get("top_k_overlap")),
            fmt_ms(number(row.get("spearman_mean")), number(row.get("spearman_std"))), fmt(old.get("spearman")),
            fmt_ms(number(row.get("js_divergence_mean")), number(row.get("js_divergence_std"))), fmt(old.get("js_divergence")),
        ])
    if expl_rendered:
        lines.extend(markdown_table(["模型", "n", "本轮 Top-K", "旧 Top-K", "本轮 Spearman", "旧 Spearman", "本轮 JS", "旧 JS"], expl_rendered))
    else:
        lines.append("> 缺失三 seed 解释汇总。")
    lines.extend(["", "旧表使用未微调/错误基线和旧指标实现；新表必须整体替换，不能只替换 SocialDebias 一行。", ""])

    lines.append("### P11 / §5.4.4 / 表 5-9：人工评估")
    lines.append("")
    human_path = results_root / "human_eval/final/human_eval_metrics.json"
    if human_path.exists():
        human = read_json(human_path).get("human_metrics", {})
        old_human = legacy.get("human_eval", {})
        human_metrics = [
            ("样本对", human.get("n_samples"), old_human.get("n_samples"), False),
            ("Clean Acc", human.get("human_acc_clean"), old_human.get("human_acc_clean"), True),
            ("Adv Acc", human.get("human_acc_adv"), old_human.get("human_acc_adv"), True),
            ("Acc Drop", human.get("human_acc_drop"), old_human.get("human_acc_drop"), True),
            ("Human ASR", human.get("human_asr"), old_human.get("human_asr"), True),
            ("关键词 Jaccard", human.get("jaccard_orig_adv_mean"), old_human.get("jaccard_orig_adv_mean"), False),
        ]
        lines.extend(markdown_table(
            ["指标", "本轮", "旧论文", "差值"],
            [[name, fmt_pct(new) if pct else fmt(new), fmt_pct(old) if pct else fmt(old), fmt_delta(new, old, pct)] for name, new, old, pct in human_metrics],
        ))
    else:
        lines.append(f"> 缺失：`{human_path}`")
    lines.extend(["", "该文件按单份既有最终标注处理，不报告 Cohen's Kappa；IG 对齐是 seed 42 案例分析。", ""])

    csv_section(lines, "P7 / §5.5.1 / 表 5-10：六组件消融（Clean）", results_root / "ablation_clean_summary.csv", ["dataset", "suffix", "n", "test_f1_mean", "test_f1_std"])
    ablation_adv_rows = csv_section(lines, "P7 / §5.5.1 / 表 5-10：六组件消融（对抗）", results_root / "ablation_adv/ablation_adv_summary.csv", ["dataset", "variant", "n", "clean_f1_mean", "avg_adv_f1_mean", "f1_drop_mean", "avg_asr_mean"])
    old_ablation = {
        (row["dataset"], row["variant"]): row
        for row in legacy.get("table_5_10", [])
    }
    ablation_comparison = []
    for row in ablation_adv_rows:
        old = old_ablation.get((row.get("dataset"), row.get("variant")))
        if not old:
            continue
        current_clean = number(row.get("clean_f1_mean"))
        current_adv = number(row.get("avg_adv_f1_mean"))
        current_drop = number(row.get("f1_drop_mean"))
        ablation_comparison.append([
            row.get("dataset"), row.get("variant"), row.get("n"),
            fmt(current_clean), fmt(old.get("clean_f1")), fmt_delta(current_clean, old.get("clean_f1")),
            fmt(current_adv), fmt(old.get("avg_adv_f1")), fmt_delta(current_adv, old.get("avg_adv_f1")),
            fmt_pp(current_drop), fmt_pp(old.get("f1_drop")),
        ])
    if ablation_comparison:
        lines.append("#### 表 5-10 与旧论文同名配置对照")
        lines.append("")
        lines.extend(markdown_table(
            ["数据集", "配置", "n", "本轮 Clean", "旧 Clean", "ΔClean", "本轮 Adv", "旧 Adv", "ΔAdv", "本轮 Drop", "旧 Drop"],
            ablation_comparison,
        ))
        lines.extend(["", "旧表中的 `no_both` 与本轮任一单组件消融不等价，因此不强行计算差值。", ""])
    csv_section(lines, "P8 / §3.3.1、§5.5.1 / 表 5-11：8 维与 17 维表层特征", results_root / "surface_8_vs_17_clean_summary.csv", ["dataset", "suffix", "n", "test_f1_mean", "test_f1_std"])
    surface_dimension_rows = csv_section(
        lines, "P8 / §3.3.1、§5.5.1 / 表 5-11：8 维与 17 维对抗结果",
        results_root / "ablation_adv/ablation_adv_summary.csv",
        ["dataset", "variant", "n", "clean_f1_mean", "avg_adv_f1_mean", "f1_drop_mean", "avg_asr_mean"],
        predicate=lambda row: row.get("dataset") == "politifact"
        and row.get("variant") in ("full", "surface17_full"),
    )
    old_dimensions = {
        (row["dataset"], row["variant"]): row
        for row in legacy.get("table_5_11", [])
    }
    dimension_comparison = []
    for row in surface_dimension_rows:
        old = old_dimensions.get((row.get("dataset"), row.get("variant")))
        if not old:
            continue
        current_clean = number(row.get("clean_f1_mean"))
        current_drop = number(row.get("f1_drop_mean"))
        dimension_comparison.append([
            row.get("dataset"), old.get("dimension"), row.get("n"),
            fmt(current_clean), fmt(old.get("clean_f1")), fmt_delta(current_clean, old.get("clean_f1")),
            fmt_pp(current_drop), fmt_pp(old.get("f1_drop")),
            fmt_delta(current_drop, old.get("f1_drop"), percentage_points=True),
        ])
    if dimension_comparison:
        lines.append("#### 表 5-11 与旧论文对照")
        lines.append("")
        lines.extend(markdown_table(
            ["数据集", "维度", "n", "本轮 Clean", "旧 Clean", "ΔClean", "本轮 Drop", "旧 Drop", "ΔDrop"],
            dimension_comparison,
        ))
        lines.append("")
    csv_section(lines, "P9 / §5.5.2 / 表 5-12：NLI 机制完整结果", results_root / "nli_mechanism_summary.csv", ["dataset", "suffix", "n", "test_f1_mean", "test_f1_std", "test_bias_acc_mean"])
    csv_section(lines, "P10 / §5.5.3 / 表 5-13：Adaptive λ Clean 对照", results_root / "adaptive_clean_summary.csv", ["dataset", "suffix", "n", "test_f1_mean", "test_f1_std", "test_bias_acc_mean"])
    csv_section(lines, "P10 / §5.5.3 / 表 5-13：固定 λ 英文对抗", results_root / "surface_fixed_adv_summary.csv", ["dataset", "model", "split", "f1_mean", "f1_std", "asr_mean", "n"])
    csv_section(lines, "P10 / §5.5.3 / 表 5-13：Adaptive λ 英文对抗", results_root / "surface_adaptive_adv_summary.csv", ["dataset", "model", "split", "f1_mean", "f1_std", "asr_mean", "n"])

    lines.append("### P10 / §5.5.3 / 表 5-13：Adaptive λ 与旧论文方向对照")
    lines.append("")
    adaptive_clean = read_csv(results_root / "adaptive_clean_summary.csv")
    adaptive_adv_rows = []
    for suffix in ("surface_fixed", "surface_adaptive"):
        grouped, _ = collect_adv_group(
            (results_root / "surface_adv").glob(f"surface_adv_*_{suffix}.json"),
            forced_model=suffix,
        )
        for row in grouped:
            adaptive_adv_rows.append({**row, "suffix": suffix})
    adaptive_adv_index = {
        (row["dataset"], row["suffix"]): row for row in adaptive_adv_rows
    }
    old_adaptive = old_index(legacy)
    old_adaptive_table = {
        (row["dataset"], row["suffix"]): row
        for row in legacy.get("table_5_13", [])
    }
    adaptive_rendered = []
    for row in adaptive_clean:
        dataset = row.get("dataset")
        suffix = row.get("suffix")
        adv = adaptive_adv_index.get((dataset, suffix), {})
        old = old_adaptive_table.get((dataset, suffix), {})
        if not old and suffix == "surface_adaptive":
            old = old_adaptive.get((dataset, "SocialDebias Adaptive"), {})
        clean = number(row.get("test_f1_mean"))
        adaptive_rendered.append([
            dataset, suffix, row.get("n"),
            fmt_ms(clean, number(row.get("test_f1_std"))),
            fmt(old.get("clean_f1")), fmt_delta(clean, old.get("clean_f1")),
            fmt_ms(adv.get("avg_adv_f1_mean"), adv.get("avg_adv_f1_std")),
            fmt(old.get("avg_adv_f1")),
            fmt_pp(adv.get("f1_drop_mean"), adv.get("f1_drop_std")),
            fmt_pp(old.get("f1_drop")),
        ])
    if adaptive_rendered:
        lines.extend(markdown_table(
            ["数据集", "配置", "n", "本轮 Clean", "旧 Clean", "ΔClean", "本轮 Avg Adv", "旧 Avg Adv", "本轮 Drop", "旧 Drop"],
            adaptive_rendered,
        ))
    else:
        lines.append("> 缺失 Adaptive λ 汇总。")
    lines.append("")

    lines.extend(["## 五、第 6 章：结论回填与差异清单", ""])
    changed = []
    for row in main_rows:
        old = row.get("old_clean_f1")
        new = row.get("clean_f1_mean")
        if old is not None and new is not None:
            changed.append((row["dataset"], row["method"], new - old, row.get("same_config", True)))
    if changed:
        lines.extend(markdown_table(
            ["数据集", "方法", "Clean F1 差值", "可比性", "论文动作"],
            [[dataset, method, f"{delta:+.4f}", "同名配置" if comparable else "配置改变", "用本轮值重写"] for dataset, method, delta, comparable in changed],
        ))
        lines.append("")
    missing_required = [check for check in checks if check["required"] and not check["ok"]]
    if missing_required:
        lines.append("### 尚未通过的验收项")
        lines.append("")
        lines.extend(f"- **{item['stage']} / {item['item']}**：实际 `{item['actual']}`，期望 `{item['expected']}`；路径 `{item['path']}`" for item in missing_required)
        lines.append("")
    else:
        lines.extend(["### 验收结论", "", "P1–P11 的必要机器可读产物全部齐全，可以冻结结果目录并开始修改论文。", ""])
    lines.extend([
        "### 论文修改规则",
        "",
        "1. 表 4-3、4-4、5-3～5-13 只使用本报告列出的 paper-v2 结果。",
        "2. 旧实现与新实现不同的行只作追溯，不把两轮 seed 混合求均值。",
        "3. 主表所有神经网络结果必须 `n=3`；单次 API 与人工案例明确标注样本口径。",
        "4. 结论方向与旧论文冲突时，以本轮结果重写摘要、创新点和第 6 章，不反向调参迁就旧数字。",
        "5. 报告中任何 ❌ 项修复前，不把对应表格写入最终论文。",
        "",
    ])
    machine = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "project_root": str(project_root),
        "results_root": str(results_root),
        "overall_pass": overall,
        "passed_checks": passed,
        "total_checks": total,
        "checks": checks,
        "main_results": main_rows,
        "history_summary": history_summary,
        "artifact_mapping": [
            {
                "stage": row[0], "section": row[1], "table_or_figure": row[2],
                "content": row[3], "source": str(row[4]), "comparison": row[5],
            }
            for row in artifact_mapping
        ],
        "unreadable_histories": unreadable_histories,
    }
    return "\n".join(lines), machine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--results_root", type=Path,
        default=PROJECT_ROOT / "results/paper_v2_20260719",
    )
    parser.add_argument(
        "--legacy_reference", type=Path,
        default=PROJECT_ROOT / "paper_reference/legacy_results_v1.json",
    )
    parser.add_argument("--endef_root", type=Path, default=Path("/autodl-fs/data/ENDEF-SIGIR2022"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json_output", type=Path, default=None)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results_root = args.results_root.resolve()
    legacy = read_json(args.legacy_reference.resolve())
    endef_root = args.endef_root.resolve() if args.endef_root.exists() else None
    report, machine = build_report(project_root, results_root, legacy, endef_root)
    output = args.output or results_root / "论文实验结果总验收与旧论文对照.md"
    json_output = args.json_output or results_root / "论文实验结果总验收与旧论文对照.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="utf-8")
    json_output.write_text(json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Markdown report: {output}")
    print(f"Machine report:  {json_output}")
    print(f"Acceptance: {'PASS' if machine['overall_pass'] else 'FAIL'} ({machine['passed_checks']}/{machine['total_checks']})")
    if args.require_complete and not machine["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
