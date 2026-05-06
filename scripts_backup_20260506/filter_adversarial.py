"""
对抗改写质量过滤管道（意见 11 简化版）

两道过滤：
1. 命名实体一致性（spaCy）
2. 语义相似度（SentenceTransformer）

NLI 蕴含校验（第 3 道）留到后续，因为需要下载 roberta-large-mnli 较慢。

使用:
    python scripts/filter_adversarial.py \\
        --original data/sheepdog/news_articles/politifact_train.pkl \\
        --rewritten data/qwen_adv/politifact_train_adv.pkl \\
        --output data/qwen_adv/politifact_train_adv_filtered.pkl
"""
import argparse
import pickle
from pathlib import Path

from tqdm import tqdm


def extract_entities_spacy(text: str, nlp) -> set:
    """提取关键实体。"""
    doc = nlp(text[:10000])  # spaCy 限制长文本
    keep_types = {"PERSON", "GPE", "LOC", "ORG", "DATE", "MONEY", "CARDINAL"}
    return {ent.text.lower().strip() for ent in doc.ents if ent.label_ in keep_types}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True,
                        help="原数据 pkl（news + labels）")
    parser.add_argument("--rewritten", required=True,
                        help="改写数据 pkl（来自 gen_adversarial_local.py）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--entity_threshold", type=float, default=0.7,
                        help="实体 Jaccard 相似度阈值（< 则丢弃）")
    parser.add_argument("--semantic_threshold", type=float, default=0.70,
                        help="语义相似度阈值（< 则丢弃）")
    parser.add_argument("--skip_semantic", action="store_true",
                        help="跳过语义相似度过滤（SentenceTransformer 需要下载）")
    args = parser.parse_args()

    # 加载数据
    print(f"加载原数据: {args.original}")
    with open(args.original, "rb") as f:
        orig = pickle.load(f)
    orig_news = orig["news"]

    print(f"加载改写数据: {args.rewritten}")
    with open(args.rewritten, "rb") as f:
        rewritten = pickle.load(f)

    print(f"  原数据: {len(orig_news)} 条")
    print(f"  改写: {len(rewritten['news'])} 条")

    # 加载 spaCy
    print("\n加载 spaCy（en_core_web_sm）...")
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("  未安装，执行：python -m spacy download en_core_web_sm")
        raise

    # 可选加载 SentenceTransformer
    st_model = None
    if not args.skip_semantic:
        print("加载 SentenceTransformer...")
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"  加载失败 ({e})，自动跳过语义过滤")
            st_model = None

    # 逐条过滤
    results = {
        "news": [], "labels": [], "style": [], "orig_idx": [],
        "entity_score": [], "semantic_score": [],
    }
    stats = {"total": 0, "pass_entity": 0, "pass_semantic": 0, "kept": 0}

    print("\n开始过滤...")
    for i in tqdm(range(len(rewritten["news"]))):
        stats["total"] += 1
        orig_idx = rewritten["orig_idx"][i]
        rw_text = rewritten["news"][i]
        status = rewritten["status"][i]

        # 改写失败的直接丢弃
        if status != "success":
            continue

        orig_text = orig_news[orig_idx]

        # 过滤 1：实体一致性
        orig_ents = extract_entities_spacy(orig_text, nlp)
        rw_ents = extract_entities_spacy(rw_text, nlp)
        ent_score = jaccard_similarity(orig_ents, rw_ents)
        if ent_score < args.entity_threshold:
            continue
        stats["pass_entity"] += 1

        # 过滤 2：语义相似度（可选）
        sem_score = -1.0
        if st_model is not None:
            embs = st_model.encode([orig_text[:5000], rw_text[:5000]], convert_to_numpy=True)
            import numpy as np
            sem_score = float(np.dot(embs[0], embs[1]) / (
                    np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8
            ))
            if sem_score < args.semantic_threshold:
                continue
        stats["pass_semantic"] += 1

        # 通过所有过滤
        results["news"].append(rw_text)
        results["labels"].append(rewritten["labels"][i])
        results["style"].append(rewritten["style"][i])
        results["orig_idx"].append(orig_idx)
        results["entity_score"].append(ent_score)
        results["semantic_score"].append(sem_score)
        stats["kept"] += 1

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n=== 过滤统计 ===")
    print(f"  总数: {stats['total']}")
    print(f"  过实体过滤: {stats['pass_entity']} ({100 * stats['pass_entity'] / stats['total']:.1f}%)")
    print(f"  过语义过滤: {stats['pass_semantic']} ({100 * stats['pass_semantic'] / stats['total']:.1f}%)")
    print(f"  最终保留: {stats['kept']} ({100 * stats['kept'] / stats['total']:.1f}%)")
    print(f"\n输出: {args.output}")


if __name__ == "__main__":
    main()