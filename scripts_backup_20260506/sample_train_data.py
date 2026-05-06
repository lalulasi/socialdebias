"""
从原训练数据采样固定数量样本，保证类别均衡。
对应论文 5.5.3 节的"采样改写"论述。

使用:
    python scripts/sample_train_data.py \
        --input data/sheepdog/news_articles/gossipcop_train.pkl \
        --output data/sheepdog/news_articles/gossipcop_train_sampled1000.pkl \
        --n_samples 1000 \
        --seed 42
"""
import argparse
import pickle
import random
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始训练 pkl")
    parser.add_argument("--output", required=True, help="采样输出 pkl")
    parser.add_argument("--n_samples", type=int, default=1000, help="采样总数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"加载: {args.input}")
    with open(args.input, "rb") as f:
        data = pickle.load(f)
    
    news = data["news"]
    labels = data["labels"]
    n_total = len(news)
    print(f"  总数: {n_total}")
    print(f"  类别分布: {Counter(labels)}")

    # 按类别分桶
    real_indices = [i for i, l in enumerate(labels) if l == 0]
    fake_indices = [i for i, l in enumerate(labels) if l == 1]
    print(f"  真新闻: {len(real_indices)}, 假新闻: {len(fake_indices)}")

    # 类别均衡采样
    n_per_class = args.n_samples // 2
    if len(real_indices) < n_per_class or len(fake_indices) < n_per_class:
        print(f"WARNING: 某类别样本不足 {n_per_class}，将按比例采样")
        n_real = min(n_per_class, len(real_indices))
        n_fake = min(n_per_class, len(fake_indices))
    else:
        n_real = n_fake = n_per_class

    random.seed(args.seed)
    sampled_real = random.sample(real_indices, n_real)
    sampled_fake = random.sample(fake_indices, n_fake)
    sampled_idx = sorted(sampled_real + sampled_fake)
    
    sampled_news = [news[i] for i in sampled_idx]
    sampled_labels = [labels[i] for i in sampled_idx]
    
    # ★ 关键：保留 orig_idx 字段，记录在原 pkl 中的索引
    # 这样后续 Qwen 改写时，orig_idx 仍能对应回 5383 条原数据
    output_data = {
        "news": sampled_news,
        "labels": sampled_labels,
        "orig_idx_in_full_train": sampled_idx,  # 在原 5383 中的索引
    }
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(output_data, f)
    
    print(f"\n=== 采样完成 ===")
    print(f"  采样数: {len(sampled_news)}")
    print(f"  类别分布: {Counter(sampled_labels)}")
    print(f"  平均长度: {sum(len(t.split()) for t in sampled_news) / len(sampled_news):.0f} 词")
    print(f"  输出: {args.output}")


if __name__ == "__main__":
    main()