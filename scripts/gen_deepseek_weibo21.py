"""
使用 DeepSeek API 改写 Weibo21 训练集。
默认生成 neutral 风格文本，并支持断点续传。
"""
import argparse
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
from prompts.rewrite_prompts import build_prompt_zh as build_prompt

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("ERROR: 未设置 DEEPSEEK_API_KEY")
    sys.exit(1)

API_URL = "https://api.deepseek.com/v1/chat/completions"


def call_deepseek(prompt: str, model: str = "deepseek-v4-flash", timeout: int = 60) -> str:
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
        "thinking": {"type": "disabled"},
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR: {type(e).__name__}: {str(e)[:100]}]"


def rewrite_one(args):
    """单个改写任务（线程池调用）"""
    idx, text, label, style, model = args
    prompt = build_prompt(text, style)
    rewritten = call_deepseek(prompt, model=model)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/weibo21_repo/data/train.pkl")
    parser.add_argument("--output", default="data/qwen_adv/weibo21_train_adv_deepseek.pkl")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--style", default="neutral")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--retry_non_success", action="store_true",
                        help="已有 status 非 success 的记录也重新请求")
    args = parser.parse_args()

    # 加载 Weibo21 训练集
    import pandas as pd
    with open(args.input, "rb") as f:
        df = pickle.load(f)
    
    n = len(df)
    print(f"加载 Weibo21 训练集: {n} 条")

    style = args.style
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 断点续传：加载已有结果
    if output_path.exists():
        with open(output_path, "rb") as f:
            existing = pickle.load(f)
        status_by_idx = dict(zip(existing["orig_idx"], existing["status"]))
        done_idx = {
            idx for idx, status in status_by_idx.items()
            if status == "success" or not args.retry_non_success
        }
        print(f"断点续传：跳过 {len(done_idx)} 条，"
              f"重试失败={args.retry_non_success}")
    else:
        existing = {"news": [], "labels": [], "style": [], "orig_idx": [], "status": []}
        done_idx = set()

    # 构造任务列表（跳过已完成的）
    tasks = []
    for i, row in df.iterrows():
        if i in done_idx:
            continue
        tasks.append((i, row["content"], row["label"], style, args.model))
    
    print(f"待处理：{len(tasks)} 条")
    if not tasks:
        print("全部完成")
        return

    # 并发改写
    start = time.time()
    results_buffer = []
    save_every = args.save_every

    def merge_results(items):
        positions = {idx: pos for pos, idx in enumerate(existing["orig_idx"])}
        for result in items:
            idx = result["idx"]
            values = {
                "news": result["rewritten"],
                "labels": result["label"],
                "style": result["style"],
                "orig_idx": idx,
                "status": result["status"],
            }
            if idx in positions:
                pos = positions[idx]
                for key, value in values.items():
                    existing[key][pos] = value
            else:
                positions[idx] = len(existing["orig_idx"])
                for key, value in values.items():
                    existing[key].append(value)
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(rewrite_one, t) for t in tasks]
        
        with tqdm(total=len(tasks), desc="DeepSeek 改写") as pbar:
            for future in as_completed(futures):
                r = future.result()
                results_buffer.append(r)
                pbar.update(1)
                
                # 定期保存
                if len(results_buffer) >= save_every:
                    merge_results(results_buffer)
                    with open(output_path, "wb") as f:
                        pickle.dump(existing, f)
                    results_buffer = []

    # 最后一次保存
    merge_results(results_buffer)
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
