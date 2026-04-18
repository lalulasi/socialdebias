"""
真实数据加载器：加载 SheepDog 预处理的 PolitiFact / GossipCop / LUN 数据。

数据格式（SheepDog 标准）：
    {'news': [str], 'labels': [int]}
    - labels: 0 = 真新闻, 1 = 假新闻
"""
import pickle
import os
import random
from typing import List, Dict, Optional, Tuple

# 数据集配置
DATASET_CONFIG = {
    "politifact": {
        "train": "data/sheepdog/news_articles/politifact_train.pkl",
        "test": "data/sheepdog/news_articles/politifact_test.pkl",
    },
    "gossipcop": {
        "train": "data/sheepdog/news_articles/gossipcop_train.pkl",
        "test": "data/sheepdog/news_articles/gossipcop_test.pkl",
    },
    "lun": {
        "train": "data/sheepdog/news_articles/lun_train.pkl",
        "test": "data/sheepdog/news_articles/lun_test.pkl",
    },
}


def load_sheepdog_pkl(path: str) -> List[Dict]:
    """
    加载 SheepDog 格式的 pkl 文件，转成我们统一的样本列表格式。

    Returns:
        [{'id': ..., 'text': ..., 'label': ...}, ...]
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)

    assert "news" in raw and "labels" in raw, f"Unexpected data format in {path}"
    assert len(raw["news"]) == len(raw["labels"]), "news 和 labels 长度不一致"

    samples = []
    for i, (text, label) in enumerate(zip(raw["news"], raw["labels"])):
        samples.append({
            "id": f"{os.path.basename(path)}_{i}",
            "text": text,
            "label": int(label),
            "language": "en",  # SheepDog 的三个数据集都是英文
        })
    return samples


def split_train_val(
        samples: List[Dict],
        val_ratio: float = 0.15,
        seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    从训练集中切出验证集。
    （SheepDog 只提供 train/test，我们需要从 train 里分出 val 用于早停）
    """
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]  # train, val


def load_dataset(
        dataset_name: str = "politifact",
        val_ratio: float = 0.15,
        max_train_samples: Optional[int] = None,
        seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    加载指定数据集，返回 train/val/test 三个列表。

    Args:
        dataset_name: 'politifact', 'gossipcop', 或 'lun'
        val_ratio: 从训练集中切出多少作为验证集
        max_train_samples: 限制训练样本数（本地调试用，上云端设为 None）
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"未知数据集: {dataset_name}，可选: {list(DATASET_CONFIG.keys())}")

    cfg = DATASET_CONFIG[dataset_name]

    # 加载训练+测试
    print(f"📂 加载 {dataset_name} 训练集: {cfg['train']}")
    train_full = load_sheepdog_pkl(cfg["train"])

    print(f"📂 加载 {dataset_name} 测试集: {cfg['test']}")
    test = load_sheepdog_pkl(cfg["test"])

    # 从训练集切分验证集
    train, val = split_train_val(train_full, val_ratio, seed)

    # 如果指定了 max_train_samples，截断（调试用）
    if max_train_samples is not None and len(train) > max_train_samples:
        random.seed(seed)
        train = random.sample(train, max_train_samples)
        print(f"⚠️  训练集被截断到 {max_train_samples} 条（调试模式）")

    # 打印数据集统计
    print(f"\n📊 数据集统计: {dataset_name}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n_real = sum(1 for s in split if s["label"] == 0)
        n_fake = sum(1 for s in split if s["label"] == 1)
        avg_len = sum(len(s["text"].split()) for s in split) / len(split) if split else 0
        print(f"  {name:5s}: 总数={len(split):4d}, 真={n_real:4d}, 假={n_fake:4d}, 平均长度={avg_len:.0f} 词")

    return train, val, test


if __name__ == "__main__":
    # 测试：加载 PolitiFact 数据集
    train, val, test = load_dataset("politifact", max_train_samples=None)

    # 看一条样本
    print("\n📖 样本示例（真新闻）:")
    real_sample = next(s for s in train if s["label"] == 0)
    print(f"  ID: {real_sample['id']}")
    print(f"  标签: {real_sample['label']} (真)")
    print(f"  文本 (前 300 字符): {real_sample['text'][:300]}")

    print("\n📖 样本示例（假新闻）:")
    fake_sample = next(s for s in train if s["label"] == 1)
    print(f"  ID: {fake_sample['id']}")
    print(f"  标签: {fake_sample['label']} (假)")
    print(f"  文本 (前 300 字符): {fake_sample['text'][:300]}")

    print("\n✅ 真实数据加载器测试通过")