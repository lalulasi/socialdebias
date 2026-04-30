"""
DeepSeek API 改写 Weibo21 训练集（用于意见 14 软标签 NLI 实验）
- 全量 5751 条 × 1 风格（neutral）
- 并发 10
- 断点续传
"""
import os
import sys
import time
import pickle
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts.rewrite_prompts import build_prompt

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("ERROR: 未设置 DEEPSEEK_API_KEY")
    sys.exit(1)

API_URL = "https://api.deepseek.com/v1/chat/completions"


def call_deepseek(prompt: str, model: str = "deepseek-chat", timeout: int = 60) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1000,
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR: {type(e).__name__}: {str(e)[:100]}]"


def rewrite_one(args):
    """单个改写任务（线程池调用）"""
    idx, text, label, style = args
    prompt = build_prompt(text, style)
    rewritten = call_deepseek(prompt)
    
    status = "success"
    if rewritten.startswith("[ERROR"):
        status = "error"
    elif len(rewritten) < 30:
        status = "too_short"
    
    return {
        "idx": idx,
        "rewritten": rewritten,
        "label": int(label),
        "style": style,
        "status": status,
    }


def main():
    # 加载 Weibo21 训练集
    import pandas as pd
    with open("data/weibo21_repo/data/train.pkl", "rb") as f:
        df = pickle.load(f)
    
    n = len(df)
    print(f"加载 Weibo21 训练集: {n} 条")

    style = "neutral"
    output_path = Path("data/qwen_adv/weibo21_train_adv_deepseek.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 断点续传：加载已有结果
    if output_path.exists():
        with open(output_path, "rb") as f:
            existing = pickle.load(f)
        # 已完成的 idx 集合
        done_idx = set(existing["orig_idx"])
        print(f"断点续传：已完成 {len(done_idx)} 条")
    else:
        existing = {"news": [], "labels": [], "style": [], "orig_idx": [], "status": []}
        done_idx = set()

    # 构造任务列表（跳过已完成的）
    tasks = []
    for i, row in df.iterrows():
        if i in done_idx:
            continue
        tasks.append((i, row["content"], row["label"], style))
    
    print(f"待处理：{len(tasks)} 条")
    if not tasks:
        print("全部完成")
        return

    # 并发改写
    start = time.time()
    results_buffer = []
    save_every = 200  # 每 200 条保存一次
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(rewrite_one, t) for t in tasks]
        
        with tqdm(total=len(tasks), desc="DeepSeek 改写") as pbar:
            for future in as_completed(futures):
                r = future.result()
                results_buffer.append(r)
                pbar.update(1)
                
                # 定期保存
                if len(results_buffer) >= save_every:
                    for r in results_buffer:
                        existing["news"].append(r["rewritten"])
                        existing["labels"].append(r["label"])
                        existing["style"].append(r["style"])
                        existing["orig_idx"].append(r["idx"])
                        existing["status"].append(r["status"])
                    with open(output_path, "wb") as f:
                        pickle.dump(existing, f)
                    results_buffer = []

    # 最后一次保存
    for r in results_buffer:
        existing["news"].append(r["rewritten"])
        existing["labels"].append(r["label"])
        existing["style"].append(r["style"])
        existing["orig_idx"].append(r["idx"])
        existing["status"].append(r["status"])
    with open(output_path, "wb") as f:
        pickle.dump(existing, f)

    elapsed = time.time() - start
    
    # 统计
    from collections import Counter
    print(f"\n=== 完成 ===")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"总数: {len(existing['news'])}")
    print(f"状态分布: {Counter(existing['status'])}")
    print(f"输出: {output_path}")


if __name__ == "__main__":
    main()