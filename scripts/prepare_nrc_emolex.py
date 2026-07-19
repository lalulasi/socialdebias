"""Normalize a licensed NRC EmoLex download for SocialDebias.

This script does not download or redistribute NRC data. Obtain the research
package from the official license page first, then run one of:

  python scripts/prepare_nrc_emolex.py \
    --english-source /path/to/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

  python scripts/prepare_nrc_emolex.py \
    --english-source /path/to/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt \
    --translations /path/to/NRC-Emotion-Lexicon-v0.92-In105Languages.xlsx
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


EMOTIONS = {
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
}


def load_long(path):
    associations = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 3:
                continue
            word, emotion, flag = row[0].strip(), row[1].strip().lower(), row[2].strip()
            if word and emotion in EMOTIONS and flag not in {"", "0", "0.0"}:
                associations.append((word, emotion, "1"))
    if not associations:
        raise ValueError(f"未从英文源文件解析到八类情绪关联: {path}")
    return associations


def write_long(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_rows = sorted(set(rows), key=lambda row: (row[0], row[1]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(unique_rows)
    return unique_rows


def find_column(columns, required_groups):
    normalized = {str(col).strip().lower(): col for col in columns}
    for norm, original in normalized.items():
        if all(any(token in norm for token in group) for group in required_groups):
            return original
    return None


def load_translation_map(path):
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        frame = pd.read_csv(path, sep=sep)

    english_col = find_column(frame.columns, [("english",), ("word", "term")])
    chinese_col = find_column(
        frame.columns,
        [("chinese", "simplified", "简体", "简体中文"), ("word", "translation", "翻译")],
    )
    if english_col is None:
        english_col = find_column(frame.columns, [("english",)])
    if chinese_col is None:
        chinese_col = find_column(frame.columns, [("chinese", "simplified", "简体")])
    if english_col is None or chinese_col is None:
        raise ValueError(
            "无法识别翻译表的英文/简体中文列。"
            f"现有列: {list(frame.columns)}"
        )

    mapping = defaultdict(set)
    for english, chinese in zip(frame[english_col], frame[chinese_col]):
        if pd.isna(english) or pd.isna(chinese):
            continue
        english = str(english).strip().lower()
        for item in str(chinese).replace("；", ";").split(";"):
            item = item.strip()
            if item:
                mapping[english].add(item)
    if not mapping:
        raise ValueError(f"翻译表未解析到有效映射: {path}")
    return mapping, str(english_col), str(chinese_col)


def summarize(rows, label):
    words = {word for word, _, _ in rows}
    counts = Counter(emotion for _, emotion, _ in rows)
    missing = sorted(EMOTIONS - set(counts))
    print(f"[{label}] words={len(words)} associations={len(rows)}")
    print(f"[{label}] emotions={dict(sorted(counts.items()))}")
    if missing:
        raise ValueError(f"{label} 缺少情绪类别: {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--english_source", required=True,
                        help="官方 Wordlevel v0.92 长表 TSV")
    parser.add_argument("--translations", default=None,
                        help="官方多语言翻译表（xlsx/csv/tsv）")
    parser.add_argument(
        "--english_output",
        default="data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    )
    parser.add_argument(
        "--chinese_output",
        default="data/lexicons/NRC-Emotion-Lexicon-ZH.tsv",
    )
    args = parser.parse_args()

    english_rows = load_long(args.english_source)
    written_en = write_long(english_rows, args.english_output)
    summarize(written_en, "NRC_EN")
    print(f"[NRC_EN] output={args.english_output}")

    if args.translations:
        translations, english_col, chinese_col = load_translation_map(args.translations)
        chinese_rows = []
        for english, emotion, flag in english_rows:
            for chinese in translations.get(english.lower(), ()):
                chinese_rows.append((chinese, emotion, flag))
        written_zh = write_long(chinese_rows, args.chinese_output)
        summarize(written_zh, "NRC_ZH")
        print(f"[NRC_ZH] columns={english_col!r} -> {chinese_col!r}")
        print(f"[NRC_ZH] output={args.chinese_output}")
    else:
        print("[NRC_ZH] 未传 --translations，仅准备英文词典")


if __name__ == "__main__":
    main()
