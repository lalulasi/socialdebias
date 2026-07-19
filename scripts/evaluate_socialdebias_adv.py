"""
SocialDebias-Adv 三方鲁棒性评估（H.3）

输入:  data/socialdebias_adv/filtered/*.pkl    (24 个过滤后的对抗 pkl)
ckpt:  results/models/  (BERT baseline + SD surface / sd_surface_adaptive)
输出:  results/socialdebias_adv_eval.csv（逐 seed、跨 seed mean/std、数据集汇总）

三方:
  1. BERT 基线        (PolitiFact/GossipCop/Weibo21 各 3 seed 平均)
  2. SocialDebias    (PF: surface；GC/Weibo21: surface_all；均为 3 seed)
  3. DeepSeek 零样本  (调 API)

ASR 计算: 与原始测试集预测对比，"翻转率" = 在原始上预测对、在改写上预测错的比例
"""
import argparse
import csv
import os
import pickle
import random
import time
from collections import defaultdict
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
from modeling.social_debias import SocialDebiasModel, infer_bottleneck_dim
from scripts.train_bert_baseline import BertBaselineModel


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
    """Load the exact baseline architecture used during training."""
    model_name = "bert-base-uncased" if lang == "en" else "bert-base-chinese"
    model = BertBaselineModel(model_name=model_name).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_sd_ckpt(ckpt_path, lang, device):
    """加载 SocialDebiasModel ckpt"""
    bert_name = "bert-base-uncased" if lang == "en" else "bert-base-chinese"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = SocialDebiasModel(
        model_name=bert_name,
        num_classes=2,
        hidden_dim=config.get("hidden_dim", 384),
        bottleneck_dim=infer_bottleneck_dim(state, config),
        surface_feat_dim=config.get("surface_feat_dim", 8),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ============== BERT 推理 ==============
@torch.no_grad()
def infer_bert(model, dataloader, device):
    preds = []
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        output = model(input_ids=ids, attention_mask=mask)
        logits = output.logits if hasattr(output, "logits") else output
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


def call_deepseek(prompt, api_key, model, max_retries=3):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 8,
        "thinking": {"type": "disabled"},
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


def infer_deepseek(texts, lang, api_key, model):
    """DeepSeek API 推理。返回 preds（-1 表示解析失败）"""
    tmpl = PROMPT_EN if lang == "en" else PROMPT_ZH
    preds = []
    for i, t in enumerate(texts):
        raw = call_deepseek(tmpl.format(text=t[:2000]), api_key, model)
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


def metric_triplet(labels, adv_preds, orig_preds):
    acc, f1 = compute_metrics(labels, adv_preds)
    return acc, f1, compute_asr(orig_preds, adv_preds, labels)


def append_mean_row(rows, file_name, dataset, tone, source, method, metrics):
    """Append the paper's across-seed mean/std rather than majority voting."""
    values = np.asarray(metrics, dtype=np.float64)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    rows.append({
        "file": file_name, "dataset": dataset, "tone": tone, "source": source,
        "method": method, "seed": "mean", "n": "",
        "acc": f"{means[0]:.4f}", "f1": f"{means[1]:.4f}",
        "asr": f"{means[2]:.4f}", "acc_std": f"{stds[0]:.4f}",
        "f1_std": f"{stds[1]:.4f}", "asr_std": f"{stds[2]:.4f}",
    })


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
    parser.add_argument("--deepseek_model", default="deepseek-v4-flash")
    parser.add_argument(
        "--sd_suffix", default=None,
        help=("统一指定三个数据集使用的 SocialDebias checkpoint 后缀；"
              "正式论文主实验建议传 surface_all，未传时保留历史数据集映射"),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ckpt 后缀
    SD_SUFFIX = {
        "politifact": "surface",
        "gossipcop":  "surface_all",
        "weibo21":    "surface_all",
    }

    api_key = os.environ.get("DEEPSEEK_API_KEY") if not args.skip_deepseek else None
    if not args.skip_deepseek and not api_key:
        print("[WARN] DEEPSEEK_API_KEY 未设置，将跳过 DeepSeek 评估")
        api_key = None

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
    aggregate = defaultdict(lambda: {"labels": [], "adv": [], "orig": []})
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

        bert_name = "bert-base-uncased" if lang == "en" else "bert-base-chinese"
        tokenizer = AutoTokenizer.from_pretrained(bert_name)

        # 最终预测只读取事实分支；surface 特征只在训练时通过偏置分支
        # 影响共享编码器，因此推理不需要再次送入 surface_feat。
        adv_ds_obj = _SimpleEvalDataset(adv_texts, labels, tokenizer, None, args.max_length)
        orig_ds_obj = _SimpleEvalDataset(orig_subset_texts, labels, tokenizer, None, args.max_length)
        adv_loader = DataLoader(adv_ds_obj, batch_size=args.batch_size)
        orig_loader = DataLoader(orig_ds_obj, batch_size=args.batch_size)

        print(f"\n========== [{fp.name}] ds={ds} tone={tone} src={source} N={len(records)} ==========")

        # ----- BERT 三 seed -----
        bert_metrics = []
        for seed in args.seeds:
            ckpt = Path(args.ckpt_dir) / f"socialdebias_{ds}_{lang}_seed{seed}_bert_baseline.pt"
            if not ckpt.exists():
                print(f"  [skip] BERT seed={seed} ckpt 不存在: {ckpt.name}")
                continue
            model = load_bert_ckpt(str(ckpt), lang, device)
            adv_pred = infer_bert(model, adv_loader, device)
            orig_pred = infer_bert(model, orig_loader, device)
            metrics = metric_triplet(labels, adv_pred, orig_pred)
            bert_metrics.append(metrics)
            rows.append({"file": fp.name, "dataset": ds, "tone": tone, "source": source,
                         "method": "BERT", "seed": seed, "n": len(labels),
                         "acc": f"{metrics[0]:.4f}", "f1": f"{metrics[1]:.4f}",
                         "asr": f"{metrics[2]:.4f}", "acc_std": "", "f1_std": "", "asr_std": ""})
            agg = aggregate[(ds, "BERT", seed)]
            agg["labels"].extend(labels); agg["adv"].extend(adv_pred); agg["orig"].extend(orig_pred)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if bert_metrics:
            append_mean_row(rows, fp.name, ds, tone, source, "BERT", bert_metrics)

        # ----- SD 三 seed -----
        sd_metrics = []
        sd_suffix = args.sd_suffix or SD_SUFFIX[ds]
        for seed in args.seeds:
            ckpt = Path(args.ckpt_dir) / f"socialdebias_{ds}_{lang}_seed{seed}_{sd_suffix}.pt"
            if not ckpt.exists():
                print(f"  [skip] SD seed={seed} ckpt 不存在: {ckpt.name}")
                continue
            model = load_sd_ckpt(str(ckpt), lang, device)
            adv_pred = infer_sd(model, adv_loader, device)
            orig_pred = infer_sd(model, orig_loader, device)
            metrics = metric_triplet(labels, adv_pred, orig_pred)
            sd_metrics.append(metrics)
            rows.append({"file": fp.name, "dataset": ds, "tone": tone, "source": source,
                         "method": "SocialDebias", "seed": seed, "n": len(labels),
                         "acc": f"{metrics[0]:.4f}", "f1": f"{metrics[1]:.4f}",
                         "asr": f"{metrics[2]:.4f}", "acc_std": "", "f1_std": "", "asr_std": ""})
            agg = aggregate[(ds, "SocialDebias", seed)]
            agg["labels"].extend(labels); agg["adv"].extend(adv_pred); agg["orig"].extend(orig_pred)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if sd_metrics:
            append_mean_row(rows, fp.name, ds, tone, source, "SocialDebias", sd_metrics)

        # ----- DeepSeek 零样本 -----
        if api_key:
            ds_adv_pred = infer_deepseek(adv_texts, lang, api_key, args.deepseek_model)
            ds_orig_pred = infer_deepseek(orig_subset_texts, lang, api_key, args.deepseek_model)
            acc, f1 = compute_metrics(labels, ds_adv_pred)
            asr = compute_asr(ds_orig_pred, ds_adv_pred, labels)
            rows.append({"file": fp.name, "dataset": ds, "tone": tone, "source": source,
                         "method": "DeepSeek", "seed": "single", "n": len(adv_texts),
                         "acc": f"{acc:.4f}" if acc is not None else "",
                         "f1": f"{f1:.4f}" if f1 is not None else "",
                         "asr": f"{asr:.4f}" if asr is not None else "",
                         "acc_std": "", "f1_std": "", "asr_std": ""})
            agg = aggregate[(ds, "DeepSeek", "single")]
            agg["labels"].extend(labels); agg["adv"].extend(ds_adv_pred); agg["orig"].extend(ds_orig_pred)
            print(f"  DeepSeek: acc={acc:.4f} f1={f1:.4f} asr={asr:.4f}" if acc else "  DeepSeek: 无有效预测")

        # 增量保存
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file","dataset","tone","source","method","seed","n","acc","f1","asr","acc_std","f1_std","asr_std"])
            writer.writeheader()
            writer.writerows(rows)

    # 论文表 5-7 使用整个数据集（所有 tone/source）的逐种子指标均值。
    dataset_seed_metrics = defaultdict(list)
    for (ds, method, seed), values in aggregate.items():
        metrics = metric_triplet(values["labels"], values["adv"], values["orig"])
        dataset_seed_metrics[(ds, method)].append(metrics)
        rows.append({"file": "__dataset_summary__", "dataset": ds, "tone": "all", "source": "all",
                     "method": method, "seed": seed, "n": len(values["labels"]),
                     "acc": f"{metrics[0]:.4f}", "f1": f"{metrics[1]:.4f}",
                     "asr": f"{metrics[2]:.4f}", "acc_std": "", "f1_std": "", "asr_std": ""})
    for (ds, method), metrics in dataset_seed_metrics.items():
        if method != "DeepSeek":
            append_mean_row(rows, "__dataset_summary__", ds, "all", "all", method, metrics)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file","dataset","tone","source","method","seed","n","acc","f1","asr","acc_std","f1_std","asr_std"])
        writer.writeheader(); writer.writerows(rows)

    print(f"\n========== 完成 ==========")
    print(f"结果 CSV: {args.output}")
    print(f"总行数: {len(rows)}")


if __name__ == "__main__":
    main()
