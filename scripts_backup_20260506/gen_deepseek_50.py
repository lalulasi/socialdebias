"""
意见 13 验证性实验：用 DeepSeek API 改写 PolitiFact 训练集前 50 条
对比 Qwen vs DeepSeek 两个 LLM 的风格分布差异
"""
import os
import sys
import time
import pickle
import json
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts.rewrite_prompts import build_prompt

# DeepSeek API
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("ERROR: 未设置 DEEPSEEK_API_KEY")
    sys.exit(1)

API_URL = "https://api.deepseek.com/v1/chat/completions"


def call_deepseek(prompt: str, model: str = "deepseek-chat") -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2000,
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR: {type(e).__name__}: {str(e)[:100]}]"


def main():
    # 加载 PolitiFact 前 50 条
    with open("data/sheepdog/news_articles/politifact_train.pkl", "rb") as f:
        data = pickle.load(f)
    
    n = 50
    news_list = data["news"][:n]
    labels = data["labels"][:n]

    # 用 neutral 风格（和 Qwen 实验对照）
    style = "neutral"
    
    results = {"news": [], "labels": [], "style": [], "orig_idx": [], "status": []}
    
    print(f"开始改写 {n} 条 (DeepSeek-Chat, style={style})...")
    start = time.time()
    
    for i, (text, label) in enumerate(tqdm(list(zip(news_list, labels)))):
        prompt = build_prompt(text, style)
        rewritten = call_deepseek(prompt)
        
        status = "success"
        if rewritten.startswith("[ERROR"):
            status = "error"
        elif len(rewritten) < 50:
            status = "too_short"
        
        results["news"].append(rewritten)
        results["labels"].append(int(label))
        results["style"].append(style)
        results["orig_idx"].append(i)
        results["status"].append(status)
        
        # 简单 rate limit
        time.sleep(0.5)
    
    elapsed = time.time() - start
    
    # 保存
    out_path = Path("data/qwen_adv/politifact_train_adv_deepseek_50.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    
    # 统计
    from collections import Counter
    print(f"\n=== 完成 ===")
    print(f"总耗时: {elapsed:.0f}s ({elapsed/n:.1f}s / 条)")
    print(f"状态分布: {Counter(results['status'])}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()