"""
LIAR 数据加载器
对应论文 5.5.X 节：speaker 可信度作为社交特征的可行性验证
"""
import pandas as pd
from typing import List, Dict
import numpy as np


# LIAR label → 二分类映射
LABEL_MAP = {
    "true": 0, "mostly-true": 0, "half-true": 0,         # 真新闻类（0）
    "barely-true": 1, "false": 1, "pants-fire": 1,        # 假新闻类（1）
}

LIAR_COLS = [
    "id", "label", "statement", "topic", "speaker",
    "speaker_job", "state", "party",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count",
    "context"
]


def extract_speaker_features(row: dict) -> np.ndarray:
    """提取 speaker 历史可信度特征（5 维）。"""
    counts = [
        row.get("barely_true_count", 0),
        row.get("false_count", 0),
        row.get("half_true_count", 0),
        row.get("mostly_true_count", 0),
        row.get("pants_fire_count", 0),
    ]
    # log(1+n) 归一化（防止极端值主导）
    feat = np.array([np.log1p(c) for c in counts], dtype=np.float32)
    return feat


def load_liar_split(tsv_path: str) -> List[Dict]:
    df = pd.read_csv(tsv_path, sep="\t", names=LIAR_COLS, header=None)
    # 数值列转 int（处理 NaN）
    for col in ["barely_true_count", "false_count", "half_true_count",
                "mostly_true_count", "pants_fire_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    samples = []
    for _, row in df.iterrows():
        label_str = row["label"]
        if label_str not in LABEL_MAP:
            continue
        samples.append({
            "id": row["id"],
            "text": row["statement"],
            "label": LABEL_MAP[label_str],
            "language": "en",
            "speaker_feat": extract_speaker_features(row.to_dict()),
        })
    return samples


def load_liar_dataset(data_dir: str = "data/liar"):
    train = load_liar_split(f"{data_dir}/train.tsv")
    val = load_liar_split(f"{data_dir}/valid.tsv")
    test = load_liar_split(f"{data_dir}/test.tsv")
    
    print(f"=== LIAR 数据集 ===")
    print(f"  train: {len(train)} (真={sum(1 for s in train if s['label']==0)}, "
          f"假={sum(1 for s in train if s['label']==1)})")
    print(f"  val: {len(val)}")
    print(f"  test: {len(test)}")
    
    return train, val, test


if __name__ == "__main__":
    train, val, test = load_liar_dataset()
    sample = train[0]
    print(f"\n样本示例：")
    print(f"  text: {sample['text'][:100]}")
    print(f"  label: {sample['label']}")
    print(f"  speaker_feat: {sample['speaker_feat']}")