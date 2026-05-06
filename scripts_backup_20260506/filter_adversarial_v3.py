"""
对抗改写质量过滤管道 V3
- 实体过滤同 V2
- 语义过滤改用本地 BERT（不依赖 sentence_transformers）

使用:
    python scripts/filter_adversarial_v3.py \\
        --original data/sheepdog/news_articles/gossipcop_train_sampled1000.pkl \\
        --rewritten data/qwen_adv/gossipcop_train_adv.pkl \\
        --output data/qwen_adv/gossipcop_train_adv_filtered.pkl
"""
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ============== 实体过滤（同 V2） ==============

def normalize_entity(s: str) -> str:
    s = s.lower().strip()
    for prefix in ["just ", "the ", "a ", "an ", "those ", "these ", "some "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_entities_spacy(text: str, nlp) -> set:
    doc = nlp(text[:10000])
    keep_types = {"PERSON", "GPE", "ORG", "MONEY"}
    ents = set()
    for ent in doc.ents:
        if ent.label_ in keep_types:
            normalized = normalize_entity(ent.text)
            if len(normalized) >= 2:
                ents.add(normalized)
    return ents


def entity_recall(orig_ents: set, rw_ents: set) -> float:
    if not orig_ents:
        return 1.0
    matched = 0
    for orig_ent in orig_ents:
        if orig_ent in rw_ents:
            matched += 1
            continue
        if any(orig_ent in rw_ent or rw_ent in orig_ent for rw_ent in rw_ents):
            matched += 1
    return matched / len(orig_ents)


# ============== 语义过滤（用本地 BERT，无需 sentence_transformers） ==============

class BertSemanticEncoder:
    """用 BERT [CLS] + mean pooling 做句向量。"""
    
    def __init__(self, model_name="bert-base-uncased", device=None):
        from transformers import AutoTokenizer, AutoModel
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print(f"[BertSemanticEncoder] 加载完成 (device={device})")
    
    @torch.no_grad()
    def encode_pair(self, text_a: str, text_b: str, max_length: int = 512):
        """编码两个文本，返回归一化后的向量对。"""
        encoded = self.tokenizer(
            [text_a[:5000], text_b[:5000]],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**encoded)
        # mean pooling
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        sum_embs = (outputs.last_hidden_state * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        embs = sum_embs / sum_mask
        # L2 归一化
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-9)
        return embs.cpu().numpy()


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """归一化后的向量直接点积就是余弦相似度。"""
    return float(np.dot(emb_a, emb_b))


# ============== 主流程 ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--rewritten", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--entity_recall_threshold", type=float, default=0.7)
    parser.add_argument("--semantic_threshold", type=float, default=0.65,
                        help="BERT mean-pooling 余弦阈值（< 则丢弃）")
    parser.add_argument("--skip_semantic", action="store_true")
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

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

    # 加载 BERT 语义编码器
    encoder = None
    if not args.skip_semantic:
        print("加载 BERT 语义编码器...")
        encoder = BertSemanticEncoder(model_name=args.bert_model)

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
    
    style_stats = {}

    for i in tqdm(range(len(rewritten["news"]))):
        stats["total"] += 1
        orig_idx = rewritten["orig_idx"][i]
        rw_text = rewritten["news"][i]
        status = rewritten["status"][i]
        style = rewritten["style"][i]
        style_stats.setdefault(style, {"total": 0, "kept": 0})
        style_stats[style]["total"] += 1
        
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
            continue
        stats["pass_entity"] += 1
        
        # 语义相似度（BERT mean pooling）
        sem_score = -1.0
        if encoder is not None:
            embs = encoder.encode_pair(orig_text, rw_text)
            sem_score = cosine_similarity(embs[0], embs[1])
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

    print(f"\n=== 过滤统计 V3 ===")
    print(f"  总数: {stats['total']}")
    print(f"  状态 error: {stats['status_error']}")
    print(f"  实体过滤通过: {stats['pass_entity']} / 失败: {stats['fail_entity']}")
    print(f"  语义过滤通过: {stats['pass_semantic']} / 失败: {stats['fail_semantic']}")
    print(f"  最终保留: {stats['kept']} ({100*stats['kept']/stats['total']:.1f}%)")
    
    print(f"\n=== 风格分布 ===")
    for style, s in sorted(style_stats.items()):
        rate = 100 * s["kept"] / s["total"] if s["total"] > 0 else 0
        print(f"  {style}: {s['kept']}/{s['total']} ({rate:.1f}%)")
    
    from collections import Counter
    idx_counts = Counter(results["orig_idx"])
    print(f"\n=== 每条原文的对抗版本数 ===")
    print(f"  分布: {Counter(idx_counts.values())}")
    print(f"  覆盖原文数: {len(idx_counts)}")
    
    print(f"\n输出: {args.output}")


if __name__ == "__main__":
    main()