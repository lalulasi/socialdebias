"""
Weibo21 数据加载器（MDFEND 仓库版本）
对应论文：跨语言泛化验证（中文虚假新闻检测）
"""
import pickle
from typing import List, Dict, Tuple


def load_weibo21_split(pkl_path: str) -> List[Dict]:
    """加载 train.pkl / val.pkl / test.pkl 单个文件。"""
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    
    samples = []
    for i, row in df.iterrows():
        samples.append({
            "id": f"weibo21_{i}",
            "text": row["content"],
            "label": int(row["label"]),  # 0=真, 1=假
            "language": "zh",
            "category": row.get("category", "unknown"),
        })
    return samples


def load_weibo21_dataset(
    data_dir: str = "data/weibo21_repo/data",
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """加载 Weibo21 完整数据集。"""
    train = load_weibo21_split(f"{data_dir}/train.pkl")
    val = load_weibo21_split(f"{data_dir}/val.pkl")
    test = load_weibo21_split(f"{data_dir}/test.pkl")
    
    print(f"=== Weibo21 数据集 ===")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n_real = sum(1 for s in split if s["label"] == 0)
        n_fake = sum(1 for s in split if s["label"] == 1)
        avg_len = sum(len(s["text"]) for s in split) / len(split) if split else 0
        print(f"  {name}: {len(split)} (真={n_real}, 假={n_fake}, 平均长度={avg_len:.0f} 字)")
    
    return train, val, test


if __name__ == "__main__":
    train, val, test = load_weibo21_dataset()
    
    # 看一条样本
    print("\n=== 样本示例 ===")
    s = train[0]
    print(f"  label: {s['label']} ({'fake' if s['label']==1 else 'real'})")
    print(f"  category: {s['category']}")
    print(f"  text: {s['text'][:200]}")
