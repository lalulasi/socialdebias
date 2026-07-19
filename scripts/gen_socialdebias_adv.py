"""
SocialDebias-Adv: Qwen + DeepSeek 双源对抗测试集生成

复用 SheepDog Section 5.1 的 4 tone prompt：
  neutral / objective / sensational / emotionally_triggering

输入：测试集 pkl
输出：每个 (dataset, tone, source) 一份 pkl，
      格式: {"original": [...], "rewritten": [...], "labels": [...], "orig_idx": [...]}

特性：串行调用 / 增量保存 / 失败重试 / 断点续跑
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


# ============== Prompt 模板 ==============
PROMPTS = {
    "en": {
        "neutral": "Rewrite the following article in a neutral tone. Output ONLY the rewritten article, with no preamble, meta-commentary, or formatting markers:\n\n{text}",
        "objective": "Rewrite the following article in an objective and professional tone. Output ONLY the rewritten article, with no preamble, meta-commentary, or formatting markers:\n\n{text}",
        "sensational": "Rewrite the following article in a sensational tone. Output ONLY the rewritten article, with no preamble, meta-commentary, or formatting markers:\n\n{text}",
        "emotionally_triggering": "Rewrite the following article in an emotionally triggering tone. Output ONLY the rewritten article, with no preamble, meta-commentary, or formatting markers:\n\n{text}",
    },
    "zh": {
        "neutral": "请用中立的语气改写下面这条新闻，保留事实内容。直接输出改写后的新闻正文，不要任何引导语、说明或格式标记：\n\n{text}",
        "objective": "请用客观专业的语气改写下面这条新闻，保留事实内容。直接输出改写后的新闻正文，不要任何引导语、说明或格式标记：\n\n{text}",
        "sensational": "请用耸动的语气改写下面这条新闻，保留事实内容。直接输出改写后的新闻正文，不要任何引导语、说明或格式标记：\n\n{text}",
        "emotionally_triggering": "请用情绪煽动的语气改写下面这条新闻，保留事实内容。直接输出改写后的新闻正文，不要任何引导语、说明或格式标记：\n\n{text}",
    },
}


# ============== API 调用 ==============
# Qwen API region URL（新加坡 / 北京 / 弗吉尼亚）
QWEN_URLS = {
    "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
    "cn":   "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "us":   "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
}


def call_qwen(prompt, api_key, model="qwen3.6-plus", region="intl", max_retries=3):
    url = QWEN_URLS[region]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2048,
        "enable_thinking": False
    }
    return _call_with_retry(url, headers, payload, max_retries)


def call_deepseek(prompt, api_key, model="deepseek-v4-flash", max_retries=3):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2048,
        "thinking": {"type": "disabled"}
    }
    return _call_with_retry(url, headers, payload, max_retries)


def _call_with_retry(url, headers, payload, max_retries):
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                print(f"  [API ERR {resp.status_code}] {resp.text[:200]}")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"  [NET ERR] {e}")
            time.sleep(2)
    return "ERROR"


# ============== 输出清理 ==============

_PREAMBLE_PATTERNS_EN = [
    r"^(?:sure|certainly|here|okay|ok|alright)[^\n]{0,80}?[:.]\s*",
    r"^(?:rewritten|here(?:'s| is)?(?:\s+the)?)[^\n]{0,80}?[:.]\s*",
    r"^\*\*[^*\n]+\*\*\s*\n",  # **粗体标题** 单行
]
_PREAMBLE_PATTERNS_ZH = [
    r"^(?:好的|当然|以下是|这是|改写后(?:的)?(?:新闻|内容|文章|正文)?)[^\n]{0,40}?[:：]\s*",
    r"^改写[:：]\s*",
]


def clean_rewritten(text, lang):
    """移除 LLM 输出中常见的引导语。"""
    if not text or text == "ERROR":
        return text
    t = text.strip()
    patterns = _PREAMBLE_PATTERNS_EN if lang == "en" else _PREAMBLE_PATTERNS_ZH
    for _ in range(2):  # 最多剥两层
        changed = False
        for p in patterns:
            new_t = re.sub(p, "", t, count=1, flags=re.IGNORECASE)
            if new_t != t:
                t = new_t.strip()
                changed = True
                break
        if not changed:
            break
    return t


# ============== 数据加载 ==============
def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        for k in ["news", "text", "content"]:
            if k in data:
                texts = list(data[k]); break
        else:
            raise ValueError(f"text key not found: {list(data.keys())}")
        for k in ["labels", "label"]:
            if k in data:
                labels = list(data[k]); break
        else:
            raise ValueError(f"label key not found: {list(data.keys())}")
    elif isinstance(data, pd.DataFrame):
        text_col = "content" if "content" in data.columns else "news"
        label_col = "label" if "label" in data.columns else "labels"
        texts = data[text_col].tolist()
        labels = data[label_col].tolist()
    else:
        raise ValueError(f"unknown pkl format: {type(data)}")
    return texts, [int(x) for x in labels]


# ============== 主流程 ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--lang", choices=["en", "zh"], required=True)
    parser.add_argument("--tone", required=True,
                        choices=["neutral", "objective", "sensational", "emotionally_triggering"])
    parser.add_argument("--source", required=True, choices=["qwen", "deepseek"])
    parser.add_argument("--qwen_region", default="intl", choices=["intl", "cn", "us"],
                        help="Qwen API 节点：intl=新加坡(默认), cn=北京, us=弗吉尼亚")
    parser.add_argument("--qwen_model", default="qwen3.6-plus")
    parser.add_argument("--deepseek_model", default="deepseek-v4-flash")
    parser.add_argument("--sample", type=int, default=0, help="0=全量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_chars", type=int, default=2000)
    args = parser.parse_args()

    # API key
    if args.source == "qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        source_model = args.qwen_model
        call_fn = lambda p, k: call_qwen(
            p, k, model=source_model, region=args.qwen_region
        )
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        source_model = args.deepseek_model
        call_fn = lambda p, k: call_deepseek(p, k, model=source_model)
    if not api_key:
        raise RuntimeError(f"{args.source} API key 环境变量未设置")

    # 数据
    print(f"[加载] {args.pkl}")
    texts, labels = load_pkl(args.pkl)

    if args.sample > 0 and args.sample < len(texts):
        rng = random.Random(args.seed)
        idx_list = sorted(rng.sample(range(len(texts)), args.sample))
        texts = [texts[i] for i in idx_list]
        labels = [labels[i] for i in idx_list]
        print(f"  采样 {args.sample}/{len(texts)+args.sample} 条 (seed={args.seed})")
    else:
        idx_list = list(range(len(texts)))
        print(f"  全量 {len(texts)} 条")

    # 断点续跑
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    start_i = 0
    if out_path.exists():
        with open(out_path, "rb") as f:
            cached = pickle.load(f)
        if (cached.get("pkl") == args.pkl and cached.get("sample") == args.sample
                and cached.get("tone") == args.tone and cached.get("source") == args.source
                and cached.get("model") == source_model):
            records = cached.get("records", [])
            start_i = len(records)
            if start_i >= len(texts):
                print(f"[完成] 已有完整结果: {out_path}")
                return
            print(f"[续跑] 从 {start_i}/{len(texts)} 继续")

    prompt_tmpl = PROMPTS[args.lang][args.tone]

    # 主循环
    t0 = time.time()
    err_count = 0
    for i in range(start_i, len(texts)):
        text = texts[i][:args.max_chars]
        prompt = prompt_tmpl.format(text=text)
        rewritten = call_fn(prompt, api_key)
        rewritten = clean_rewritten(rewritten, args.lang)

        if rewritten == "ERROR":
            err_count += 1

        records.append({
            "orig_idx": idx_list[i],
            "original": text,
            "rewritten": rewritten,
            "label": labels[i],
        })

        if (i + 1) % args.save_every == 0 or (i + 1) == len(texts):
            with open(out_path, "wb") as f:
                pickle.dump({
                    "pkl": args.pkl, "lang": args.lang, "tone": args.tone,
                    "source": args.source, "model": source_model,
                    "sample": args.sample, "seed": args.seed,
                    "records": records,
                }, f)
            elapsed = time.time() - t0
            done = i + 1 - start_i
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(texts) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(texts)}] {rate:.2f}/s ETA {eta/60:.1f}min err={err_count}")

    print(f"\n[完成] 用时 {(time.time()-t0)/60:.1f} min, 错误 {err_count}/{len(records)}")


if __name__ == "__main__":
    main()
