"""
DeepSeek zero-shot baseline evaluation.

针对一个测试集（pkl 文件），逐条调用 DeepSeek API 做真假判断，
输出 Accuracy / F1-macro / ASR（仅对抗集）+ 每条样本的预测明细。

用法：
    python scripts/eval_llm_baseline.py \
        --pkl data/sheepdog/news_articles/politifact_test.pkl \
        --lang en \
        --sample 0 \
        --seed 42 \
        --output results/llm_baseline/politifact_clean.json
"""
import argparse
import json
import os
import pickle
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score


PROMPT_EN = """You are a fact-checker. Determine if the following news is real or fake.
Reply with ONLY one word: "real" or "fake". No explanation.

News:
{text}

Answer:"""

PROMPT_ZH = """你是一个新闻真伪判别员。请判断以下新闻是真实的还是虚假的。
只回答一个词："真" 或 "假"。不要解释。

新闻：
{text}

答："""


DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


def call_deepseek(prompt: str, api_key: str, max_retries: int = 3) -> str:
    """调用 DeepSeek API，失败重试。返回 raw response 文本。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers,
                                 json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  [429] retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [API ERROR {resp.status_code}] {resp.text[:200]}")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"  [NET ERROR] {e}")
            time.sleep(2)
    return "ERROR"


def parse_response(raw: str, lang: str) -> int:
    """
    解析 LLM 输出为 0/1 标签。
    约定：0 = real, 1 = fake (与 SheepDog 一致；调用前会确认)。
    返回 -1 表示无法解析。
    """
    if not raw or raw == "ERROR":
        return -1

    text = raw.lower().strip()
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)  # 去标点

    if lang == "en":
        if "fake" in text:
            return 1
        if "real" in text:
            return 0
    else:  # zh
        if "假" in text:
            return 1
        if "真" in text:
            return 0

    return -1


def load_pkl(path: str):
    """统一加载 dict 或 DataFrame，返回 (texts, labels)。"""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        for txt_key in ["news", "text", "content"]:
            if txt_key in data:
                texts = data[txt_key]
                break
        else:
            raise ValueError(f"未找到文本字段，可用键: {list(data.keys())}")
        for lbl_key in ["labels", "label"]:
            if lbl_key in data:
                labels = data[lbl_key]
                break
        else:
            raise ValueError(f"未找到标签字段，可用键: {list(data.keys())}")
    elif isinstance(data, pd.DataFrame):
        text_col = "content" if "content" in data.columns else "news"
        label_col = "label" if "label" in data.columns else "labels"
        texts = data[text_col].tolist()
        labels = data[label_col].tolist()
    else:
        raise ValueError(f"未知数据格式: {type(data)}")

    return list(texts), [int(x) for x in labels]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="测试集 pkl 路径")
    parser.add_argument("--lang", choices=["en", "zh"], required=True)
    parser.add_argument("--sample", type=int, default=0,
                        help="采样数（0 表示全量）")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每 N 条保存一次 checkpoint")
    parser.add_argument("--max_chars", type=int, default=2000,
                        help="新闻文本截断字符数（防止 prompt 过长）")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 环境变量未设置")

    print(f"[加载] {args.pkl}")
    texts, labels = load_pkl(args.pkl)
    print(f"  样本数: {len(texts)}, 类别分布: 0={labels.count(0)}, 1={labels.count(1)}")

    if args.sample > 0 and args.sample < len(texts):
        rng = random.Random(args.seed)
        idx = rng.sample(range(len(texts)), args.sample)
        idx.sort()
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
        print(f"  采样 {args.sample} 条 (seed={args.seed})")
    else:
        idx = list(range(len(texts)))
        print(f"  全量评估 {len(texts)} 条")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    start_i = 0
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("pkl") == args.pkl and cached.get("sample") == args.sample:
            records = cached.get("records", [])
            start_i = len(records)
            if start_i >= len(texts):
                print(f"[完成] 已有完整结果，直接跳过: {out_path}")
                _print_metrics(records)
                return
            print(f"[续跑] 从 {start_i}/{len(texts)} 继续")

    prompt_tmpl = PROMPT_EN if args.lang == "en" else PROMPT_ZH

    t0 = time.time()
    for i in range(start_i, len(texts)):
        text = texts[i][:args.max_chars]
        label = labels[i]
        prompt = prompt_tmpl.format(text=text)

        raw = call_deepseek(prompt, api_key)
        pred = parse_response(raw, args.lang)

        records.append({
            "idx": idx[i],
            "label": label,
            "pred": pred,
            "raw": raw,
        })

        if (i + 1) % args.save_every == 0 or (i + 1) == len(texts):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "pkl": args.pkl,
                    "lang": args.lang,
                    "sample": args.sample,
                    "seed": args.seed,
                    "records": records,
                }, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - t0
            done = i + 1 - start_i
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(texts) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(texts)}] {rate:.2f} req/s, ETA {eta/60:.1f}min")

    print(f"\n[完成] 用时 {(time.time()-t0)/60:.1f} 分钟")
    _print_metrics(records)


def _print_metrics(records):
    """打印 acc, f1, parse 失败率。"""
    valid = [r for r in records if r["pred"] != -1]
    invalid = len(records) - len(valid)
    if not valid:
        print("  [WARN] 无有效预测")
        return

    y_true = [r["label"] for r in valid]
    y_pred = [r["pred"] for r in valid]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"  Acc: {acc:.4f}")
    print(f"  F1-macro: {f1:.4f}")
    print(f"  解析失败: {invalid}/{len(records)} ({invalid/len(records)*100:.1f}%)")


if __name__ == "__main__":
    main()
