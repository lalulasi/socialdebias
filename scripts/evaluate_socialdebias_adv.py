"""
SocialDebias-Adv 三方鲁棒性评估（H.3）

输入:  data/socialdebias_adv/filtered/*.pkl    (24 个过滤后的对抗 pkl)
ckpt:  results/models/  (BERT baseline + SD surface / sd_surface_adaptive)
输出:  results/socialdebias_adv_eval.csv  (24 测试集 × 3 方法 × Acc/F1/ASR)

三方:
  1. BERT 基线        (PolitiFact/GossipCop/Weibo21 各 3 seed 平均)
  2. SocialDebias    (PF/GC: surface 3 seed; Weibo21: sd_surface_adaptive 3 seed)
  3. DeepSeek 零样本  (调 API)

ASR 计算: 与原始测试集预测对比，"翻转率" = 在原始上预测对、在改写上预测错的比例
"""
import argparse
import csv
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from modeling.social_debias import SocialDebiasModel
from utils.surface_features import SurfaceFeatureExtractor


# ============== 数据集类（轻量内联，不依赖 SurfaceAugmentedDataset） ==============
class _SimpleEvalDataset(torch.utils.data.Dataset):
    """评估专用 Dataset：文本 + label + 预提取的 surface_feat"""
    def __init__(self, texts, labels, tokenizer, surface_feats=None,
                 max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.surface_feats = surface_feats  # numpy array [N, 8] or None
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }
        if self.surface_feats is not None:
            item["surface_feat"] = torch.tensor(self.surface_feats[i], dtype=torch.float32)
        return item


# ============== 模型加载 ==============
def load_bert_ckpt(ckpt_path, lang, device):
    """加载 BERT baseline ckpt（用 SocialDebiasModel 兼容架构 or 单独 BERT 分类器）。
    
    根据用户实际 baseline ckpt 结构判断。这里假设 baseline 是简单的 BertForSequenceClassification。
    """
    from transformers import BertForSequenceClassification
    model_name = "bert-base-uncased" if lang == "en" else "bert-base-chinese"
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_sd_ckpt(ckpt_path, lang, device, surface_feat_dim=8):
    """加载 SocialDebiasModel ckpt"""
    bert_name = "bert-base-uncased" if lang == "en" else "bert-base-chinese"
    model = SocialDebiasModel(
        model_name=bert_name,
        num_classes=2,
        surface_feat_dim=surface_feat_dim,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ============== BERT 推理 ==============
@torch.no_grad()
def infer_bert(model, dataloader, device):
    preds = []
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


# ============== SD 推理 ==============
@torch.no_grad()
def infer_sd(model, dataloader, device):
    preds = []
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        sf = batch.get("surface_feat", None)
        if sf is not None:
            sf = sf.to(device)
        out = model(ids, mask, surface_feat=sf)
        logits = out["fact_logits"]
        preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


# ============== DeepSeek 零样本 ==============
PROMPT_EN = ('You are a fact-checker. Determine if the following news is real or fake.\n'
             'Reply with ONLY one word: "real" or "fake".\n\nNews:\n{text}\n\nAnswer:')
PROMPT_ZH = '你是一个新闻真伪判别员。判断以下新闻是真还是假。只回答"真"或"假"。\n\n新闻：\n{text}\n\n答：'


def call_deepseek(prompt, api_key, max_retries=3):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 8,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                time.sleep(2)
        except requests.exceptions.RequestException:
            time.sleep(2)
    return "ERROR"


def parse_ds_response(raw, lang):
    if not raw or raw == "ERROR":
        return -1
    t = raw.lower()
    if lang == "en":
        if "fake" in t: return 1
        if "real" in t: return 0
    else:
        if "假" in t: return 1
        if "真" in t: return 0
    return -1


def infer_deepseek(texts, lang, api_key):
    """DeepSeek API 推理。返回 preds（-1 表示解析失败）"""
    tmpl = PROMPT_EN if lang == "en" else PROMPT_ZH
    preds = []
    for i, t in enumerate(texts):
        raw = call_deepseek(tmpl.format(text=t[:2000]), api_key)
        preds.append(parse_ds_response(raw, lang))
        if (i + 1) % 50 == 0:
            print(f"    DeepSeek [{i+1}/{len(texts)}]")
    return preds


# ============== 指标计算 ==============
def compute_metrics(y_true, y_pred):
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
    if not valid:
        return None, None
    yt = [t for t, _ in valid]
    yp = [p for _, p in valid]
    return accuracy_score(yt, yp), f1_score(yt, yp, average="macro")


def compute_asr(orig_preds, adv_preds, labels):
    """ASR = 在原始上预测对、在改写上预测错（且非 -1）的比例"""
    correct_in_orig = 0
    flipped = 0
    for op, ap, lb in zip(orig_preds, adv_preds, labels):
        if op == lb:  # 原始上预测对
            correct_in_orig += 1
            if ap != -1 and ap != op:
                flipped += 1
    return flipped / correct_in_orig if correct_in_orig > 0 else None


# ============== 主流程 ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered_dir", default="data/socialdebias_adv/filtered")
    parser.add_argument("--orig_dir", default="data/sheepdog/news_articles",
                        help="原始 test pkl 路径（用于 ASR 计算）")
    parser.add_argument("--weibo21_orig", default="data/weibo21_repo/data/test.pkl")
    parser.add_argument("--ckpt_dir", default="results/models")
    parser.add_argument("--output", default="results/socialdebias_adv_eval.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--skip_deepseek", action="store_true",
                        help="跳过 DeepSeek API（省钱）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ckpt 后缀
    SD_SUFFIX = {
        "politifact": "surface",
        "gossipcop":  "surface",
        "weibo21":    "sd_surface_adaptive",
    }

    api_key = os.environ.get("DEEPSEEK_API_KEY") if not args.skip_deepseek else None
    if not args.skip_deepseek and not api_key:
        print("[WARN] DEEPSEEK_API_KEY 未设置，将跳过 DeepSeek 评估")
        api_key = None

    # 加载 surface 特征提取器（中英文共用一个实例，词典内置中英 emotion 词混合）
    print("[加载] SurfaceFeatureExtractor")
    feat_extractor = SurfaceFeatureExtractor()

    # 处理每个 filtered pkl
    filtered_files = sorted(Path(args.filtered_dir).glob("*.pkl"))
    if not filtered_files:
        raise RuntimeError(f"未找到过滤后 pkl: {args.filtered_dir}")
    print(f"待评估 {len(filtered_files)} 个文件")

    # 缓存原始测试集预测（用于 ASR）
    orig_preds_cache = {}  # key: (dataset, method, seed) -> [preds on orig test]

    # ============== 第一步：跑原始测试集预测（ASR 基线） ==============
    print("\n========== 第 1 步：原始测试集基线预测 ==========")
    orig_pkls = {
        "politifact": f"{args.orig_dir}/politifact_test.pkl",
        "gossipcop":  f"{args.orig_dir}/gossipcop_test.pkl",
        "weibo21":    args.weibo21_orig,
    }
    orig_data = {}  # key: dataset -> (texts, labels, surface_feats)
    for ds, pkl in orig_pkls.items():
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        if isinstance(d, dict):
            txt_key = "news" if "news" in d else "content"
            lbl_key = "labels" if "labels" in d else "label"
            texts = list(d[txt_key])
            labels = [int(x) for x in d[lbl_key]]
        else:  # DataFrame (Weibo21)
            txt_col = "content" if "content" in d.columns else "news"
            lbl_col = "label" if "label" in d.columns else "labels"
            texts = d[txt_col].tolist()
            labels = [int(x) for x in d[lbl_col]]
        # GossipCop / Weibo21 全量推理太慢，采样到与 filtered pkl 对齐
        # 但 filtered pkl 是按 orig_idx 选的，无法直接对齐，所以直接全量
        # 实际改写的样本只占测试集一小部分，按 orig_idx 索引
        orig_data[ds] = {"texts": texts, "labels": labels}
    print(f"  原始测试集加载完毕：{[(k, len(v['texts'])) for k, v in orig_data.items()]}")

    # ============== 第二步：循环评估每个 filtered pkl ==============
    rows = []
    for fp in filtered_files:
        with open(fp, "rb") as f:
            data = pickle.load(f)
        records = data["records"]
        if not records:
            print(f"\n[跳过] {fp.name}: 空 records")
            continue

        name = fp.stem
        if "politifact" in name:
            ds = "politifact"; lang = "en"
        elif "gossipcop" in name:
            ds = "gossipcop"; lang = "en"
        else:
            ds = "weibo21"; lang = "zh"

        tone = next((t for t in ["neutral", "objective", "sensational", "emotionally_triggering"]
                     if t in name), "unknown")
        source = "qwen" if "qwen" in name else "deepseek"

        adv_texts = [r["rewritten"] for r in records]
        labels = [r["label"] for r in records]
        orig_idxs = [r["orig_idx"] for r in records]

        # 取对应的原始 texts（用于 BERT/SD 在原始上的预测，算 ASR）
        orig_subset_texts = [orig_data[ds]["texts"][i] for i in orig_idxs]

        # 提取 surface features（单条 extract 循环 + stack 成 [N, 17/8] numpy）
        adv_feats = np.stack([feat_extractor.extract(t) for t in adv_texts]).astype(np.float32)
        orig_subset_feats = np.stack([feat_extractor.extract(t) for t in orig_subset_texts]).astype(np.float32)

        bert_name = "bert-base-uncased" if lang == "en" else "bert-base-chinese"
        tokenizer = AutoTokenizer.from_pretrained(bert_name)

        adv_ds_obj = _SimpleEvalDataset(adv_texts, labels, tokenizer, adv_feats, args.max_length)
        orig_ds_obj = _SimpleEvalDataset(orig_subset_texts, labels, tokenizer, orig_subset_feats, args.max_length)
        adv_loader = DataLoader(adv_ds_obj, batch_size=args.batch_size)
        orig_loader = DataLoader(orig_ds_obj, batch_size=args.batch_size)

        print(f"\n========== [{fp.name}] ds={ds} tone={tone} src={source} N={len(records)} ==========")

        # ----- BERT 三 seed -----
        bert_advs, bert_origs = [], []
        for seed in args.seeds:
            ckpt = Path(args.ckpt_dir) / f"socialdebias_{ds}_{lang}_seed{seed}_bert.pt"
            if not ckpt.exists():
                print(f"  [skip] BERT seed={seed} ckpt 不存在: {ckpt.name}")
                continue
            model = load_bert_ckpt(str(ckpt), lang, device)
            bert_advs.append(infer_bert(model, adv_loader, device))
            bert_origs.append(infer_bert(model, orig_loader, device))
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 多 seed 投票（majority vote）
        if bert_advs:
            bert_adv_pred = [int(np.median([s[i] for s in bert_advs])) for i in range(len(adv_texts))]
            bert_orig_pred = [int(np.median([s[i] for s in bert_origs])) for i in range(len(adv_texts))]
            acc, f1 = compute_metrics(labels, bert_adv_pred)
            asr = compute_asr(bert_orig_pred, bert_adv_pred, labels)
            rows.append({"file": fp.name, "dataset": ds, "tone": tone, "source": source,
                         "method": "BERT", "n": len(adv_texts),
                         "acc": f"{acc:.4f}" if acc else "", "f1": f"{f1:.4f}" if f1 else "",
                         "asr": f"{asr:.4f}" if asr is not None else ""})
            print(f"  BERT:     acc={acc:.4f} f1={f1:.4f} asr={asr:.4f}" if acc else "  BERT: 无有效预测")

        # ----- SD 三 seed -----
        sd_advs, sd_origs = [], []
        sd_suffix = SD_SUFFIX[ds]
        for seed in args.seeds:
            ckpt = Path(args.ckpt_dir) / f"socialdebias_{ds}_{lang}_seed{seed}_{sd_suffix}.pt"
            if not ckpt.exists():
                print(f"  [skip] SD seed={seed} ckpt 不存在: {ckpt.name}")
                continue
            model = load_sd_ckpt(str(ckpt), lang, device, surface_feat_dim=8)
            sd_advs.append(infer_sd(model, adv_loader, device))
            sd_origs.append(infer_sd(model, orig_loader, device))
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if sd_advs:
            sd_adv_pred = [int(np.median([s[i] for s in sd_advs])) for i in range(len(adv_texts))]
            sd_orig_pred = [int(np.median([s[i] for s in sd_origs])) for i in range(len(adv_texts))]
            acc, f1 = compute_metrics(labels, sd_adv_pred)
            asr = compute_asr(sd_orig_pred, sd_adv_pred, labels)
            rows.append({"file": fp.name, "dataset": ds, "tone": tone, "source": source,
                         "method": "SocialDebias", "n": len(adv_texts),
                         "acc": f"{acc:.4f}" if acc else "", "f1": f"{f1:.4f}" if f1 else "",
                         "asr": f"{asr:.4f}" if asr is not None else ""})
            print(f"  SD:       acc={acc:.4f} f1={f1:.4f} asr={asr:.4f}" if acc else "  SD: 无有效预测")

        # ----- DeepSeek 零样本 -----
        if api_key:
            ds_adv_pred = infer_deepseek(adv_texts, lang, api_key)
            ds_orig_pred = infer_deepseek(orig_subset_texts, lang, api_key)
            acc, f1 = compute_metrics(labels, ds_adv_pred)
            asr = compute_asr(ds_orig_pred, ds_adv_pred, labels)
            rows.append({"file": fp.name, "dataset": ds, "tone": tone, "source": source,
                         "method": "DeepSeek", "n": len(adv_texts),
                         "acc": f"{acc:.4f}" if acc else "", "f1": f"{f1:.4f}" if f1 else "",
                         "asr": f"{asr:.4f}" if asr is not None else ""})
            print(f"  DeepSeek: acc={acc:.4f} f1={f1:.4f} asr={asr:.4f}" if acc else "  DeepSeek: 无有效预测")

        # 增量保存
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file","dataset","tone","source","method","n","acc","f1","asr"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n========== 完成 ==========")
    print(f"结果 CSV: {args.output}")
    print(f"总行数: {len(rows)}")


if __name__ == "__main__":
    main()
