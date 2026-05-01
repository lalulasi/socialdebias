"""
对抗改写质量过滤管道 V4 - 中文版（用于 Weibo21 + DeepSeek 改写数据）

中文适配：
1. 实体提取改用 jieba 分词 + 自定义实体识别（spaCy 中文版需额外安装）
2. 语义编码用 bert-base-chinese
3. 实体匹配用召回率（同 V3）

使用:
    python scripts/filter_adversarial_v4_zh.py \\
        --original data/weibo21_repo/data/train.pkl \\
        --rewritten data/qwen_adv/weibo21_train_adv_deepseek.pkl \\
        --output data/qwen_adv/weibo21_train_adv_filtered.pkl
"""
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ============== 中文实体提取（基于规则 + jieba 词性）==============

def extract_entities_zh(text: str) -> set:
    """提取中文文本中的关键实体（人名、地名、机构名、数字、日期）。"""
    import jieba.posseg as pseg
    
    entities = set()
    text = text[:8000]  # 限制长度
    
    # 用 jieba 分词标注词性
    words = pseg.cut(text)
    for word, flag in words:
        word = word.strip()
        if len(word) < 2:
            continue
        # nr=人名, ns=地名, nt=机构名, t=时间, m=数词
        if flag in ("nr", "ns", "nt"):
            entities.add(word.lower())
    
    # 数字 + 单位（用正则）
    # 匹配 "5万" "20%" "300人" "2020年" 等
    number_patterns = [
        r"\d+\.?\d*[%％]",
        r"\d+\.?\d*[万亿千百十]",
        r"\d+\.?\d*[年月日]",
        r"\d+\.?\d*[人个家次]",
    ]
    for pattern in number_patterns:
        for m in re.finditer(pattern, text):
            entities.add(m.group())
    
    return entities


def entity_recall(orig_ents: set, rw_ents: set) -> float:
    if not orig_ents:
        return 1.0
    matched = 0
    for orig in orig_ents:
        if orig in rw_ents:
            matched += 1
            continue
        # 模糊匹配：substring 包含
        if any(orig in rw or rw in orig for rw in rw_ents):
            matched += 1
    return matched / len(orig_ents)


# ============== 中文 BERT 语义编码 ==============

class BertSemanticEncoderZh:
    def __init__(self, model_name="bert-base-chinese", device=None):
        from transformers import AutoTokenizer, AutoModel
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print(f"[BertSemanticEncoderZh] 加载完成 (device={device})")

    @torch.no_grad()
    def encode_pair(self, text_a: str, text_b: str, max_length: int = 256):
        encoded = self.tokenizer(
            [text_a[:2000], text_b[:2000]],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        sum_embs = (outputs.last_hidden_state * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        embs = sum_embs / sum_mask
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-9)
        return embs.cpu().numpy()


def cosine_similarity(a, b):
    return float(np.dot(a, b))


# ============== 主流程 ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--rewritten", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--entity_recall_threshold", type=float, default=0.6)
    parser.add_argument("--semantic_threshold", type=float, default=0.65)
    parser.add_argument("--skip_semantic", action="store_true")
    parser.add_argument("--bert_model", default="bert-base-chinese")
    args = parser.parse_args()

    print(f"加载原数据: {args.original}")
    # Weibo21 是 DataFrame
    with open(args.original, "rb") as f:
        df = pickle.load(f)
    orig_news = df["content"].tolist()
    
    print(f"加载改写数据: {args.rewritten}")
    with open(args.rewritten, "rb") as f:
        rewritten = pickle.load(f)
    
    print(f"  原数据: {len(orig_news)} 条")
    print(f"  改写: {len(rewritten['news'])} 条")

    # 加载 jieba（首次使用会初始化词典）
    print("\n初始化 jieba...")
    import jieba
    import jieba.posseg as pseg
    # warmup
    list(pseg.cut("测试初始化"))
    print("  完成")

    # BERT 编码器
    encoder = None
    if not args.skip_semantic:
        encoder = BertSemanticEncoderZh(model_name=args.bert_model)

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

    for i in tqdm(range(len(rewritten["news"]))):
        stats["total"] += 1
        orig_idx = rewritten["orig_idx"][i]
        rw_text = rewritten["news"][i]
        status = rewritten["status"][i]
        
        if status != "success":
            stats["status_error"] += 1
            continue
        
        orig_text = orig_news[orig_idx]
        
        # 实体召回
        orig_ents = extract_entities_zh(orig_text)
        rw_ents = extract_entities_zh(rw_text)
        recall = entity_recall(orig_ents, rw_ents)
        
        if recall < args.entity_recall_threshold:
            stats["fail_entity"] += 1
            continue
        stats["pass_entity"] += 1
        
        # 语义相似度
        sem_score = -1.0
        if encoder is not None:
            embs = encoder.encode_pair(orig_text, rw_text)
            sem_score = cosine_similarity(embs[0], embs[1])
            if sem_score < args.semantic_threshold:
                stats["fail_semantic"] += 1
                continue
        stats["pass_semantic"] += 1
        
        results["news"].append(rw_text)
        results["labels"].append(rewritten["labels"][i])
        results["style"].append(rewritten["style"][i])
        results["orig_idx"].append(orig_idx)
        results["entity_recall_score"].append(recall)
        results["semantic_score"].append(sem_score)
        stats["kept"] += 1

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n=== 过滤统计 V4-中文 ===")
    print(f"  总数: {stats['total']}")
    print(f"  状态 error: {stats['status_error']}")
    print(f"  实体过滤通过: {stats['pass_entity']} / 失败: {stats['fail_entity']}")
    print(f"  语义过滤通过: {stats['pass_semantic']} / 失败: {stats['fail_semantic']}")
    print(f"  最终保留: {stats['kept']} ({100*stats['kept']/stats['total']:.1f}%)")
    
    # orig_idx 覆盖
    print(f"  覆盖原文数: {len(set(results['orig_idx']))}")
    print(f"\n输出: {args.output}")


if __name__ == "__main__":
    main()