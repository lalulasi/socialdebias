"""
DashScope API 对抗改写脚本（方案 β：4 风格 × 全量）—— Ollama 版改造为阿里云百炼 API

使用:
    export DASHSCOPE_API_KEY=你的key
    # base_url 默认新加坡国际公共地址；如需切换 export DASHSCOPE_BASE_URL=...
    python scripts/gen_adversarial_dashscope.py \
        --input data/sheepdog/news_articles/gossipcop_train.pkl \
        --output data/qwen_adv/gossipcop_train_adv.pkl \
        --styles neutral academic report simplified \
        --model qwen-plus \
        --num_workers 8

支持断点续传：中途挂了直接重跑，会跳过已完成的。
与 Ollama 版完全兼容：checkpoint 格式、输出 pkl 格式、命令行参数均不变。
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts.rewrite_prompts import build_prompt, STYLE_PROMPTS


# ============== DashScope (阿里云百炼) 配置 ==============
# base_url 优先读环境变量 DASHSCOPE_BASE_URL，未设置则用新加坡国际公共地址。
# 常见可选：
#   国际公共（新加坡）: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
#   国内公共（北京）:   https://dashscope.aliyuncs.com/compatible-mode/v1
#   业务空间专属:       https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
# 切换：export DASHSCOPE_BASE_URL=你要用的地址
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# 全局 client（线程安全，复用同一连接池，不要在每次调用里 new）
_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not _API_KEY:
    print("ERROR: 未设置环境变量 DASHSCOPE_API_KEY")
    print("  export DASHSCOPE_API_KEY=你的key")
    sys.exit(1)

# 防止 key 含非 ASCII 字符（复制时易混入中文标点/全角空格，导致 ascii 编码报错）
if not _API_KEY.isascii():
    print("ERROR: DASHSCOPE_API_KEY 含非 ASCII 字符，请重新设置纯英文数字的 key")
    sys.exit(1)

print(f"base_url: {DASHSCOPE_BASE_URL}")
client = OpenAI(api_key=_API_KEY, base_url=DASHSCOPE_BASE_URL)


def call_dashscope(prompt: str, model: str = "qwen-plus", timeout: int = 180,
                   max_retries: int = 3) -> str:
    """调用 DashScope（OpenAI 兼容模式），同步返回完整文本。带限流重试。"""
    last_err = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.9,
                max_tokens=2000,
                timeout=timeout,
                # 关闭 Qwen3 思考模式（等价于 Ollama 的 "think": False）
                extra_body={"enable_thinking": False},
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # 限流 / 超时 / 临时故障 → 指数退避重试
            if any(k in msg for k in ["rate", "limit", "429", "timeout", "timed out",
                                       "503", "502", "overload"]):
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
            return f"[ERROR: {type(e).__name__}: {str(e)[:100]}]"
    return f"[ERROR: {type(last_err).__name__}: {str(last_err)[:100]}]"


def clean_qwen_output(text: str) -> str:
    """清理 Qwen 输出的 <think>...</think> 块及多余前缀。"""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^/no_think\s*", "", text, flags=re.IGNORECASE)

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
    raw = call_dashscope(prompt, model=model)
    cleaned = clean_qwen_output(raw)

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
    if not checkpoint_path.exists():
        return {}
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(results: dict, checkpoint_path: Path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    tmp.replace(checkpoint_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="原数据 pkl 路径（含 news 和 labels 字段）")
    parser.add_argument("--output", required=True, help="输出 pkl 路径")
    parser.add_argument("--styles", nargs="+",
                        default=["neutral", "academic", "report", "simplified"],
                        help="要生成的风格列表")
    # 模型名以你百炼控制台/业务空间实际可用的为准（如 qwen-plus / qwen-turbo / qwen3-8b）
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="并发数（API 可并发，建议 5-10，注意 QPS 限额）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 条（测试用）")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="按 --seed 从原数据确定性抽样 N 条；与 --limit 互斥")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry_non_success", action="store_true",
                        help="断点中 status 非 success 的任务也重新请求")
    parser.add_argument("--save_every", type=int, default=20,
                        help="每 N 条保存一次 checkpoint")
    args = parser.parse_args()

    if args.limit is not None and args.sample_size is not None:
        parser.error("--limit 与 --sample_size 不能同时使用")

    for s in args.styles:
        if s not in STYLE_PROMPTS:
            print(f"未知风格: {s}")
            print(f"可用: {list(STYLE_PROMPTS.keys())}")
            sys.exit(1)

    print(f"加载原数据: {args.input}")
    with open(args.input, "rb") as f:
        data = pickle.load(f)
    news_list = data["news"]
    labels = data["labels"]
    n_total = len(news_list)
    selected_indices = list(range(n_total))
    if args.sample_size is not None:
        if not 0 < args.sample_size <= n_total:
            parser.error(f"--sample_size 必须在 1..{n_total} 之间")
        selected_indices = sorted(
            random.Random(args.seed).sample(selected_indices, args.sample_size)
        )
    elif args.limit is not None:
        selected_indices = selected_indices[:args.limit]
    print(f"  总数: {n_total}，本次处理: {len(selected_indices)}")
    print(f"  风格: {args.styles}")
    print(f"  模型: {args.model}")
    print(f"  并发: {args.num_workers}")

    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(".checkpoint.pkl")
    results = load_checkpoint(checkpoint_path)
    print(f"  已完成: {len(results)} 条（断点续传）")

    tasks = []
    for idx in selected_indices:
        news, label = news_list[idx], labels[idx]
        for style in args.styles:
            key = f"{idx}_{style}"
            if (key not in results
                    or (args.retry_non_success
                        and results[key].get("status") != "success")):
                tasks.append((idx, news, label, style))
    print(f"  待处理: {len(tasks)} 条")
    if len(tasks) == 0:
        print("  没有新任务，直接退出整理输出")
    else:
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

                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        "rate": f"{rate:.2f}条/s",
                        "ETA": f"{(len(tasks) - count) / rate / 60:.0f}min" if rate > 0 else "?",
                    })

                    if count % args.save_every == 0:
                        save_checkpoint(results, checkpoint_path)
                except Exception as e:
                    print(f"\n[ERROR] idx={idx} style={style}: {e}")

        save_checkpoint(results, checkpoint_path)

    print(f"\n整理输出到 {args.output}")
    output = {"news": [], "labels": [], "style": [], "orig_idx": [], "status": []}
    stats = {s: 0 for s in ["success", "error", "too_short", "too_long"]}

    selected_keys = {
        f"{idx}_{style}" for idx in selected_indices for style in args.styles
    }
    for key, r in results.items():
        if key not in selected_keys:
            continue
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
