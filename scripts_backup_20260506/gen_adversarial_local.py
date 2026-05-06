"""
本地 Qwen3 对抗改写脚本（方案 β：4 风格 × 全量）

使用:
    python scripts/gen_adversarial_local.py \\
        --input data/sheepdog/news_articles/politifact_train.pkl \\
        --output data/qwen_adv/politifact_train_adv.pkl \\
        --styles neutral academic report simplified \\
        --num_workers 2

支持断点续传：中途挂了直接重跑，会跳过已完成的。
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts.rewrite_prompts import build_prompt, STYLE_PROMPTS


OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama(prompt: str, model: str = "qwen3:8b", timeout: int = 180) -> str:
    """调用本地 Ollama API，同步返回完整文本。"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 2000,   # 限制最大输出 token
        },
        "think": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.Timeout:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {type(e).__name__}: {str(e)[:100]}]"


def clean_qwen_output(text: str) -> str:
    """清理 Qwen 输出的 <think>...</think> 块及多余前缀。"""
    # 移除 think 标签（以防万一 no_think 没起效）
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^/no_think\s*", "", text, flags=re.IGNORECASE)

    # 移除常见的前缀
    for prefix in [
        "REWRITTEN NEWS:", "REWRITTEN:",
        "Here is the rewritten news:", "Here's the rewritten news:",
        "The rewritten news is:",
    ]:
        if text.lstrip().lower().startswith(prefix.lower()):
            text = text.lstrip()[len(prefix):].lstrip()

    return text.strip()


def rewrite_one(idx: int, original: str, label: int, style: str, model: str) -> dict:
    """单条改写任务。"""
    prompt = build_prompt(original, style)
    raw = call_ollama(prompt, model=model)
    cleaned = clean_qwen_output(raw)

    # 简单质量检查
    status = "success"
    if cleaned.startswith("[TIMEOUT]") or cleaned.startswith("[ERROR"):
        status = "error"
    elif len(cleaned) < 50:
        status = "too_short"
    elif len(cleaned) > len(original) * 3:
        status = "too_long"

    return {
        "idx": idx,
        "style": style,
        "label": label,
        "original": original,
        "rewritten": cleaned,
        "raw": raw if status != "success" else None,
        "status": status,
    }


def load_checkpoint(checkpoint_path: Path) -> dict:
    """加载已完成的任务记录。"""
    if not checkpoint_path.exists():
        return {}
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(results: dict, checkpoint_path: Path):
    """保存当前进度。"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    tmp.replace(checkpoint_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="原数据 pkl 路径（含 news 和 labels 字段）")
    parser.add_argument("--output", required=True,
                        help="输出 pkl 路径")
    parser.add_argument("--styles", nargs="+",
                        default=["neutral", "academic", "report", "simplified"],
                        help="要生成的风格列表")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="并发数（Mac M5 推荐 2）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 条（测试用）")
    parser.add_argument("--save_every", type=int, default=20,
                        help="每 N 条保存一次 checkpoint")
    args = parser.parse_args()

    # 验证风格
    for s in args.styles:
        if s not in STYLE_PROMPTS:
            print(f"未知风格: {s}")
            print(f"可用: {list(STYLE_PROMPTS.keys())}")
            sys.exit(1)

    # 加载原数据
    print(f"加载原数据: {args.input}")
    with open(args.input, "rb") as f:
        data = pickle.load(f)
    news_list = data["news"]
    labels = data["labels"]
    n_total = len(news_list)
    if args.limit:
        news_list = news_list[:args.limit]
        labels = labels[:args.limit]
    print(f"  总数: {n_total}，本次处理: {len(news_list)}")
    print(f"  风格: {args.styles}")
    print(f"  并发: {args.num_workers}")

    # 断点续传
    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(".checkpoint.pkl")
    results = load_checkpoint(checkpoint_path)
    print(f"  已完成: {len(results)} 条（断点续传）")

    # 构造任务清单
    tasks = []
    for idx, (news, label) in enumerate(zip(news_list, labels)):
        for style in args.styles:
            key = f"{idx}_{style}"
            if key not in results:
                tasks.append((idx, news, label, style))
    print(f"  待处理: {len(tasks)} 条")
    if len(tasks) == 0:
        print("  没有新任务，直接退出整理输出")
    else:
        # 并发执行
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_task = {
                executor.submit(rewrite_one, idx, news, label, style, args.model): (idx, style)
                for idx, news, label, style in tasks
            }

            count = 0
            pbar = tqdm(as_completed(future_to_task), total=len(tasks), desc="改写中")
            for future in pbar:
                idx, style = future_to_task[future]
                try:
                    result = future.result()
                    key = f"{idx}_{style}"
                    results[key] = result
                    count += 1

                    # 实时显示速率
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        "rate": f"{rate:.2f}条/s",
                        "ETA": f"{(len(tasks) - count) / rate / 60:.0f}min" if rate > 0 else "?",
                    })

                    # 定期保存
                    if count % args.save_every == 0:
                        save_checkpoint(results, checkpoint_path)
                except Exception as e:
                    print(f"\n[ERROR] idx={idx} style={style}: {e}")

        # 最终保存 checkpoint
        save_checkpoint(results, checkpoint_path)

    # 整理输出成最终 pkl 格式
    print(f"\n整理输出到 {args.output}")
    output = {
        "news": [],
        "labels": [],
        "style": [],
        "orig_idx": [],
        "status": [],
    }
    stats = {s: 0 for s in ["success", "error", "too_short", "too_long"]}

    for key, r in results.items():
        output["news"].append(r["rewritten"])
        output["labels"].append(r["label"])
        output["style"].append(r["style"])
        output["orig_idx"].append(r["idx"])
        output["status"].append(r["status"])
        stats[r["status"]] = stats.get(r["status"], 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\n=== 质量统计 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n总计 {len(output['news'])} 条改写")
    print(f"输出文件: {args.output}")
    print(f"断点文件（可删除）: {checkpoint_path}")


if __name__ == "__main__":
    main()