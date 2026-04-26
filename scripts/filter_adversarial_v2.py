"""
对抗改写质量过滤管道 V2 - 修复实体匹配过严问题

改进:
1. 实体规范化（去掉 just/the/a 等修饰词，归一化日期格式）
2. 只看核心实体（PERSON/GPE/ORG/MONEY），去除 DATE/CARDINAL 噪声
3. 用"实体召回率"代替 Jaccard 相似度（原文实体在改写中的保留率）

使用:
    python scripts/filter_adversarial_v2.py \\
        --original data/sheepdog/news_articles/politifact_train.pkl \\
        --rewritten data/qwen_adv/politifact_train_adv.pkl \\
        --output data/qwen_adv/politifact_train_adv_filtered_v2.pkl
"""
import argparse
import pickle
import re
from pathlib import Path

from tqdm import tqdm


def normalize_entity(s: str) -> str:
    """实体归一化。"""
    s = s.lower().strip()
    # 去掉常见前缀修饰
    for prefix in ["just ", "the ", "a ", "an ", "those ", "these ", "some "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    # 去掉序数词后缀
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s)
    # 去掉多余空格
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_entities_spacy(text: str, nlp) -> set:
    """提取核心实体（PERSON/GPE/ORG/MONEY）。"""
    doc = nlp(text[:10000])
    keep_types = {"PERSON", "GPE", "ORG", "MONEY"}  # 去掉 DATE/CARDINAL/LOC
    ents = set()
    for ent in doc.ents:
        if ent.label_ in keep_types:
            normalized = normalize_entity(ent.text)
            if len(normalized) >= 2:  # 过滤掉单字符实体
                ents.add(normalized)
    return ents


def entity_recall(orig_ents: set, rw_ents: set) -> float:
    """原文实体在改写中的召回率（模糊匹配）。"""
    if not orig_ents:
        return 1.0
    matched = 0
    for orig_ent in orig_ents:
        # 严格匹配
        if orig_ent in rw_ents:
            matched += 1
            continue
        # 模糊匹配（substring 包含）
        if any(orig_ent in rw_ent or rw_ent in orig_ent for rw_ent in rw_ents):
            matched += 1
    return matched / len(orig_ents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--rewritten", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--entity_recall_threshold", type=float, default=0.7,
                        help="原文核心实体召回率（< 则丢弃）")
    parser.add_argument("--semantic_threshold", type=float, default=0.65,
                        help="语义相似度阈值")
    parser.add_argument("--skip_semantic", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="打印每个样本的得分")
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
    print("\n加载 spaCy...")
    import spacy
    nlp = spacy.load("en_core_web_sm")

    # 加载 SentenceTransformer
    st_model = None
    if not args.skip_semantic:
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"加载 SentenceTransformer 失败 ({e})，跳过语义过滤")

    # 逐条过滤
    results = {
        "news": [], "labels": [], "style": [], "orig_idx": [],
        "entity_recall_score": [], "semantic_score": [],
    }
    stats = {
        "total": 0, "status_error": 0,
        "pass_entity": 0, "fail_entity": 0,
        "pass_semantic": 0, "fail_semantic": 0,
        "kept": 0,
    }

    # 记录每个风格的通过情况
    style_stats = {}

    for i in tqdm(range(len(rewritten["news"]))):
        stats["total"] += 1
        orig_idx = rewritten["orig_idx"][i]
        rw_text = rewritten["news"][i]
        status = rewritten["status"][i]
        style = rewritten["style"][i]
        style_stats.setdefault(style, {"total": 0, "kept": 0})
        style_stats[style]["total"] += 1

        # 改写失败的直接丢
        if status != "success":
            stats["status_error"] += 1
            continue

        orig_text = orig_news[orig_idx]

        # 实体召回率
        orig_ents = extract_entities_spacy(orig_text, nlp)
        rw_ents = extract_entities_spacy(rw_text, nlp)
        recall = entity_recall(orig_ents, rw_ents)

        if recall < args.entity_recall_threshold:
            stats["fail_entity"] += 1
            if args.debug:
                print(f"\n[FAIL idx={orig_idx} style={style}] recall={recall:.2f}")
                print(f"  原文实体: {sorted(orig_ents)[:8]}")
                print(f"  改写实体: {sorted(rw_ents)[:8]}")
                print(f"  丢失: {sorted(orig_ents - rw_ents)[:5]}")
            continue
        stats["pass_entity"] += 1

        # 语义相似度
        sem_score = -1.0
        if st_model is not None:
            embs = st_model.encode([orig_text[:5000], rw_text[:5000]], convert_to_numpy=True)
            import numpy as np
            sem_score = float(np.dot(embs[0], embs[1]) / (
                    np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8
            ))
            if sem_score < args.semantic_threshold:
                stats["fail_semantic"] += 1
                continue
        stats["pass_semantic"] += 1

        # 通过
        results["news"].append(rw_text)
        results["labels"].append(rewritten["labels"][i])
        results["style"].append(style)
        results["orig_idx"].append(orig_idx)
        results["entity_recall_score"].append(recall)
        results["semantic_score"].append(sem_score)
        stats["kept"] += 1
        style_stats[style]["kept"] += 1

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n=== 过滤统计 V2 ===")
    print(f"  总数: {stats['total']}")
    print(f"  状态 error: {stats['status_error']}")
    print(f"  实体过滤通过: {stats['pass_entity']} / 失败: {stats['fail_entity']}")
    print(f"  语义过滤通过: {stats['pass_semantic']} / 失败: {stats['fail_semantic']}")
    print(f"  最终保留: {stats['kept']} ({100 * stats['kept'] / stats['total']:.1f}%)")
    print(f"\n=== 风格分布（通过率）===")
    for style, s in sorted(style_stats.items()):
        rate = 100 * s["kept"] / s["total"] if s["total"] > 0 else 0
        print(f"  {style}: {s['kept']}/{s['total']} ({rate:.1f}%)")

    # 每个 orig_idx 的对抗版本数分布
    from collections import Counter
    idx_counts = Counter(results["orig_idx"])
    print(f"\n=== 每条原文的对抗版本数分布 ===")
    print(f"  {Counter(idx_counts.values())}")
    print(f"  覆盖原文数: {len(idx_counts)} / 360")

    print(f"\n输出: {args.output}")


if __name__ == "__main__":
    main()