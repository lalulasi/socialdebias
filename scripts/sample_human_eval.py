#!/usr/bin/env python
"""
scripts/sample_human_eval.py

从 PolitiFact 测试集 + adv_C 对抗变体中分层采样 50 条人工评估样本。
每条样本占 CSV 中 2 行（原文 + adv_C 改写），共 100 行。

用法：
    python scripts/sample_human_eval.py \
        --test_pkl data/sheepdog/news_articles/politifact_test.pkl \
        --adv_pkl  data/sheepdog/adversarial_test/politifact_test_adv_C.pkl \
        --n_samples 50 \
        --seed 42 \
        --output results/human_eval/politifact_human_eval_template.csv
"""
import argparse
import csv
import os
import pickle
import random
from pathlib import Path


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_texts(data):
    """SheepDog pkl 兼容两种字段名：news / rewritten。"""
    if "news" in data:
        return data["news"]
    if "rewritten" in data:
        return data["rewritten"]
    raise KeyError(f"pkl missing 'news' or 'rewritten' key. Got: {list(data.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_pkl",
        type=str,
        default="data/sheepdog/news_articles/politifact_test.pkl",
    )
    parser.add_argument(
        "--adv_pkl",
        type=str,
        default="data/sheepdog/adversarial_test/politifact_test_adv_C.pkl",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Total samples to draw (split equally between real/fake)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="results/human_eval/politifact_human_eval_template.csv",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="hide gold labels/text types and shuffle rows for blind annotation",
    )
    parser.add_argument(
        "--key_output",
        type=str,
        default=None,
        help="answer-key CSV required with --blind",
    )
    args = parser.parse_args()

    test_data = load_pkl(args.test_pkl)
    adv_data = load_pkl(args.adv_pkl)

    orig_texts = get_texts(test_data)
    orig_labels = list(test_data["labels"])
    adv_texts = get_texts(adv_data)
    adv_labels = list(adv_data["labels"])

    assert len(orig_texts) == len(adv_texts), (
        f"Size mismatch: orig {len(orig_texts)} vs adv {len(adv_texts)}"
    )
    assert orig_labels == adv_labels, "Labels mismatch between orig and adv pkl"

    real_idx = [i for i, l in enumerate(orig_labels) if l == 0]
    fake_idx = [i for i, l in enumerate(orig_labels) if l == 1]
    print(
        f"Test set total: {len(orig_labels)} "
        f"(real={len(real_idx)}, fake={len(fake_idx)})"
    )

    rng = random.Random(args.seed)
    n_per_class = args.n_samples // 2
    n_real = min(n_per_class, len(real_idx))
    n_fake = min(args.n_samples - n_real, len(fake_idx))

    sampled_real = rng.sample(real_idx, n_real)
    sampled_fake = rng.sample(fake_idx, n_fake)
    sampled = sorted(sampled_real + sampled_fake)

    records = []
    for idx in sampled:
        label_str = "real" if orig_labels[idx] == 0 else "fake"
        sample_id = f"pf_test_{idx:03d}"
        records.extend([
            {
                "id": sample_id, "label": label_str, "text_type": "original",
                "text": str(orig_texts[idx]).strip(),
            },
            {
                "id": sample_id, "label": label_str, "text_type": "adv_C",
                "text": str(adv_texts[idx]).strip(),
            },
        ])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.blind:
        if not args.key_output:
            raise ValueError("--blind requires --key_output")
        rng.shuffle(records)
        for position, record in enumerate(records, start=1):
            record["blind_id"] = f"blind_{position:03d}"
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                "blind_id", "text", "human_keywords", "human_judgment",
                "confidence", "notes",
            ])
            for record in records:
                writer.writerow([record["blind_id"], record["text"], "", "", "", ""])
        Path(args.key_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.key_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["blind_id", "id", "label", "text_type"])
            for record in records:
                writer.writerow([
                    record["blind_id"], record["id"], record["label"], record["text_type"]
                ])
        print(f"Answer key (do not give to annotators): {args.key_output}")
    else:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                "id", "label", "text_type", "text", "human_keywords",
                "human_judgment", "confidence", "notes",
            ])
            for record in records:
                writer.writerow([
                    record["id"], record["label"], record["text_type"],
                    record["text"], "", "", "", "",
                ])

    print(
        f"Sampled: {len(sampled)} pairs "
        f"({n_real} real + {n_fake} fake), seed={args.seed}"
    )
    print(f"Output: {args.output}")
    print(f"Rows: {len(sampled) * 2 + 1} (1 header + {len(sampled) * 2} data)")


if __name__ == "__main__":
    main()
