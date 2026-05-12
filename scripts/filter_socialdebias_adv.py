"""
SocialDebias-Adv 三道质量过滤管道（H.2）

复用训练对抗过滤的同一套阈值与模型栈，保证测试集对抗与训练集对抗的过滤标准一致。

英文路径（politifact_*, gossipcop_*）：
  spaCy en_core_web_sm 实体 + bert-base-uncased 语义相似度 + mDeBERTa-v3-base-mnli-xnli NLI

中文路径（weibo21_*）：
  jieba.posseg (nr/ns/nt/m/t) + bert-base-chinese 语义相似度 + mDeBERTa NLI

阈值：
  entity_recall ≥ 0.6
  semantic_sim ≥ 0.65
  NLI label ∈ {entailment, neutral}（排除 contradiction）

输出：
  - data/socialdebias_adv/filtered/{原文件名}.pkl  保留通过过滤的 records，含各维度分数 + p_entail
  - data/socialdebias_adv/filtered/filter_report.csv  汇总每个文件的保留率/失败原因分布
"""
import argparse
import csv
import os
import pickle
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)


# ============== 实体提取 ==============
def extract_entities_en(text, nlp):
    """英文 spaCy NER 提取实体集合（PERSON/ORG/GPE/DATE/MONEY 等）"""
    doc = nlp(text[:10000])  # 截断防止超长
    return {ent.text.lower().strip() for ent in doc.ents if ent.text.strip()}


def extract_entities_zh(text, pseg):
    """中文 jieba.posseg 提取关键实体（人名/地名/机构/数量/时间）"""
    target_pos = {"nr", "ns", "nt", "m", "t"}
    entities = set()
    for word, flag in pseg.cut(text[:10000]):
        if flag in target_pos and len(word.strip()) > 1:
            entities.add(word.strip())
    return entities


def entity_recall(orig_ents, rw_ents):
    """改写文召回了原文多少实体（模糊匹配）"""
    if not orig_ents:
        return 1.0
    matched = 0
    rw_lower = {e.lower() for e in rw_ents}
    for oe in orig_ents:
        oe_low = oe.lower()
        if oe_low in rw_lower or any(oe_low in re_ or re_ in oe_low for re_ in rw_lower):
            matched += 1
    return matched / len(orig_ents)


# ============== BERT 语义相似度（mean pooling） ==============
class BertSemanticEncoder:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts, batch_size=16, max_length=256):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(self.device)
            out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counted = mask.sum(dim=1).clamp(min=1)
            mean = summed / counted
            all_emb.append(F.normalize(mean, p=2, dim=1).cpu())
        return torch.cat(all_emb, dim=0)

    def cosine(self, a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ============== NLI ==============
class NLIScorer:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device
        # mDeBERTa-v3 标签顺序: 0=entailment, 1=neutral, 2=contradiction
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def score(self, premise, hypothesis, max_length=512):
        enc = self.tokenizer(premise[:2000], hypothesis[:2000],
                             truncation=True, max_length=max_length,
                             return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits[0]
        probs = F.softmax(logits, dim=-1)
        pred_idx = int(probs.argmax().item())
        pred_label = self.id2label[pred_idx].lower()
        return {
            "p_entail": float(probs[0]),
            "p_neutral": float(probs[1]),
            "p_contradict": float(probs[2]),
            "pred_label": pred_label,
        }


# ============== 主过滤逻辑 ==============
def filter_one_file(input_pkl, output_pkl, lang, ent_extractor, sem_encoder, nli):
    """对一个 pkl 应用三道过滤"""
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    records_in = data["records"]
    print(f"\n[{input_pkl.name}] 输入 {len(records_in)} 条 (lang={lang})")

    # 先把所有原文 + 改写文一起 batch 编码（节省时间）
    originals = [r["original"] for r in records_in]
    rewrittens = [r["rewritten"] for r in records_in]
    emb_orig = sem_encoder.encode(originals)
    emb_rw = sem_encoder.encode(rewrittens)

    stats = defaultdict(int)
    out_records = []

    for i, r in enumerate(records_in):
        # 跳过 ERROR 条
        if r["rewritten"] == "ERROR":
            stats["fail_error"] += 1
            continue

        orig, rw = r["original"], r["rewritten"]

        # 道 1：实体召回
        if lang == "en":
            orig_ents = extract_entities_en(orig, ent_extractor)
            rw_ents = extract_entities_en(rw, ent_extractor)
        else:
            orig_ents = extract_entities_zh(orig, ent_extractor)
            rw_ents = extract_entities_zh(rw, ent_extractor)
        recall = entity_recall(orig_ents, rw_ents)

        threshold = 0.7 if lang == "en" else 0.6
        if recall < threshold:
            stats["fail_entity"] += 1
            continue

        # 道 2：语义相似度
        sim = float(F.cosine_similarity(emb_orig[i:i+1], emb_rw[i:i+1]).item())
        if sim < 0.65:
            stats["fail_semantic"] += 1
            continue

        # 道 3：NLI
        nli_res = nli.score(orig, rw)
        if nli_res["pred_label"] == "contradiction":
            stats["fail_nli"] += 1
            continue

        # 通过
        stats["pass"] += 1
        out_records.append({
            **r,  # orig_idx, original, rewritten, label
            "entity_recall": round(recall, 4),
            "semantic_sim": round(sim, 4),
            "p_entail": round(nli_res["p_entail"], 4),
            "p_neutral": round(nli_res["p_neutral"], 4),
            "p_contradict": round(nli_res["p_contradict"], 4),
            "nli_label": nli_res["pred_label"],
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(records_in)}] pass={stats['pass']} "
                  f"fail_ent={stats['fail_entity']} fail_sem={stats['fail_semantic']} "
                  f"fail_nli={stats['fail_nli']} fail_err={stats['fail_error']}")

    # 输出
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump({
            **{k: v for k, v in data.items() if k != "records"},  # 保留元信息
            "records": out_records,
            "filter_thresholds": {
                "entity_recall": threshold, "semantic_sim": 0.65,
                "nli_exclude": "contradiction",
            },
        }, f)

    n_in = len(records_in)
    n_out = len(out_records)
    print(f"  → 保留 {n_out}/{n_in} ({n_out/n_in*100:.1f}%) "
          f"| 失败: entity={stats['fail_entity']} semantic={stats['fail_semantic']} "
          f"nli={stats['fail_nli']} error={stats['fail_error']}")

    return {
        "file": input_pkl.name,
        "lang": lang,
        "n_in": n_in,
        "n_out": n_out,
        "keep_rate": round(n_out / n_in, 4) if n_in > 0 else 0,
        "fail_entity": stats["fail_entity"],
        "fail_semantic": stats["fail_semantic"],
        "fail_nli": stats["fail_nli"],
        "fail_error": stats["fail_error"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/socialdebias_adv")
    parser.add_argument("--output_dir", default="data/socialdebias_adv/filtered")
    parser.add_argument("--bert_en", default="bert-base-uncased")
    parser.add_argument("--bert_zh", default="bert-base-chinese")
    parser.add_argument("--nli_model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(f for f in in_dir.glob("*.pkl")
                   if not f.name.startswith("_") and "filtered" not in str(f.parent.name).lower()
                   or f.parent == in_dir)
    files = [f for f in files if not f.name.startswith("_")]
    if not files:
        raise RuntimeError(f"未找到 pkl: {in_dir}")
    print(f"待过滤 {len(files)} 个文件")

    # 模型加载（共享给所有文件）
    print(f"\n[加载] spaCy en_core_web_sm")
    import spacy
    nlp_en = spacy.load("en_core_web_sm")

    print(f"[加载] jieba")
    import jieba
    import jieba.posseg as pseg
    jieba.initialize()

    print(f"[加载] BERT (en): {args.bert_en}")
    enc_en = BertSemanticEncoder(args.bert_en, device)

    print(f"[加载] BERT (zh): {args.bert_zh}")
    enc_zh = BertSemanticEncoder(args.bert_zh, device)

    print(f"[加载] NLI: {args.nli_model}")
    nli = NLIScorer(args.nli_model, device)

    # 主循环
    report = []
    for i, fp in enumerate(files):
        if fp.name.startswith("weibo21"):
            lang = "zh"; ent = pseg; enc = enc_zh
        else:
            lang = "en"; ent = nlp_en; enc = enc_en

        out_path = out_dir / fp.name
        if out_path.exists():
            print(f"\n[{i+1}/{len(files)}] {fp.name} 已存在，跳过")
            with open(out_path, "rb") as f:
                cached = pickle.load(f)
            with open(fp, "rb") as f:
                orig_data = pickle.load(f)
            report.append({
                "file": fp.name, "lang": lang,
                "n_in": len(orig_data["records"]),
                "n_out": len(cached["records"]),
                "keep_rate": round(len(cached["records"]) / max(1, len(orig_data["records"])), 4),
                "fail_entity": "-", "fail_semantic": "-", "fail_nli": "-", "fail_error": "-",
            })
            continue

        print(f"\n[{i+1}/{len(files)}] 处理: {fp.name}")
        rep = filter_one_file(fp, out_path, lang, ent, enc, nli)
        report.append(rep)

    # 汇总 CSV
    csv_path = out_dir / "filter_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)

    print(f"\n===== 汇总 =====")
    print(f"{'文件':<55}{'lang':<6}{'n_in':>6}{'n_out':>7}{'keep%':>8}")
    for r in report:
        kr = r["keep_rate"] * 100 if isinstance(r["keep_rate"], float) else r["keep_rate"]
        print(f"{r['file']:<55}{r['lang']:<6}{r['n_in']:>6}{r['n_out']:>7}{kr:>7.1f}%")
    print(f"\n报告写入: {csv_path}")


if __name__ == "__main__":
    main()
