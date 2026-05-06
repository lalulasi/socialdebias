"""
汇总 LLM 基线 11 个测试集结果为 CSV，方便论文写作。

输出列：dataset, variant, n_samples, acc, f1_macro, parse_fail_rate, asr
"""
import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(records):
    """返回 (acc, f1, parse_fail_rate, n_valid)。"""
    valid = [r for r in records if r["pred"] != -1]
    n_total = len(records)
    n_valid = len(valid)
    parse_fail = (n_total - n_valid) / n_total if n_total > 0 else 0
    if n_valid == 0:
        return None, None, parse_fail, 0
    y_true = [r["label"] for r in valid]
    y_pred = [r["pred"] for r in valid]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1, parse_fail, n_valid


def compute_asr(clean_records, adv_records):
    """
    ASR (Attack Success Rate) = 在 clean 上预测正确的样本中，
    在 adv 上预测翻转的比例。
    要求两组 records 按 idx 对齐。
    """
    clean_map = {r["idx"]: r for r in clean_records if r["pred"] != -1}
    flip_count = 0
    correct_in_clean = 0
    for adv_r in adv_records:
        idx = adv_r["idx"]
        if idx not in clean_map:
            continue
        clean_r = clean_map[idx]
        if clean_r["pred"] == clean_r["label"]:
            correct_in_clean += 1
            if adv_r["pred"] != -1 and adv_r["pred"] != clean_r["pred"]:
                flip_count += 1
    if correct_in_clean == 0:
        return None
    return flip_count / correct_in_clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)

    # 收集所有 JSON
    results = {}  # (dataset, variant) -> records
    for fp in sorted(in_dir.glob("*.json")):
        stem = fp.stem  # e.g. "politifact_clean", "gossipcop_adv_A"
        if "_" not in stem:
            continue
        # 解析 dataset 与 variant
        parts = stem.split("_", 1)
        dataset = parts[0]
        variant = parts[1] if len(parts) > 1 else "clean"
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        results[(dataset, variant)] = data.get("records", [])

    # 输出 CSV
    rows = []
    fieldnames = ["dataset", "variant", "n_samples",
                  "acc", "f1_macro", "parse_fail_rate", "asr"]

    # 按数据集分组，每组先输出 clean 行算指标，再输出 adv 行（含 ASR）
    datasets_seen = sorted(set(d for d, _ in results.keys()))
    for ds in datasets_seen:
        # clean 必须先处理
        for variant in ["clean", "adv_A", "adv_B", "adv_C", "adv_D"]:
            key = (ds, variant)
            if key not in results:
                continue
            records = results[key]
            acc, f1, fail, n = compute_metrics(records)

            asr = None
            if variant.startswith("adv_") and (ds, "clean") in results:
                asr = compute_asr(results[(ds, "clean")], records)

            rows.append({
                "dataset": ds,
                "variant": variant,
                "n_samples": n,
                "acc": f"{acc:.4f}" if acc is not None else "",
                "f1_macro": f"{f1:.4f}" if f1 is not None else "",
                "parse_fail_rate": f"{fail:.4f}",
                "asr": f"{asr:.4f}" if asr is not None else "",
            })

    # 写 CSV
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 终端打印
    print(f"\n[CSV] {args.output}")
    print()
    print(f"{'dataset':<12}{'variant':<10}{'n':<6}{'acc':<8}{'f1':<8}{'fail%':<8}{'asr':<8}")
    print("-" * 60)
    for r in rows:
        print(f"{r['dataset']:<12}{r['variant']:<10}{r['n_samples']:<6}"
              f"{r['acc']:<8}{r['f1_macro']:<8}{r['parse_fail_rate']:<8}{r['asr']:<8}")


if __name__ == "__main__":
    main()
