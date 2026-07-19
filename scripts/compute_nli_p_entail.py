"""
对对抗样本计算 NLI 蕴含概率 p_entail
对应论文 4.2.2 / 4.3.1 节软标签机制

使用:
  python scripts/compute_nli_p_entail.py \\
      --original data/sheepdog/news_articles/politifact_train.pkl \\
      --rewritten data/qwen_adv/politifact_train_adv_filtered_v2.pkl \\
      --output data/qwen_adv/politifact_p_entail.pkl \\
      --orig_format pkl_dict
"""
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np


def resolve_nli_label_indices(id2label):
    """Resolve entailment/neutral/contradiction indices from model metadata."""
    normalized = {str(v).lower(): int(k) for k, v in id2label.items()}
    aliases = {
        "entailment": ("entailment", "entail"),
        "neutral": ("neutral",),
        "contradiction": ("contradiction", "contradictory", "contradict"),
    }
    resolved = {}
    for target, names in aliases.items():
        for label, idx in normalized.items():
            if any(name in label for name in names):
                resolved[target] = idx
                break
    if len(resolved) != 3:
        raise ValueError(
            f"NLI 模型 id2label 无法解析三分类标签: {id2label}"
        )
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="原数据 pkl")
    parser.add_argument("--rewritten", required=True, help="过滤后的对抗 pkl")
    parser.add_argument("--output", required=True, help="输出 p_entail pkl")
    parser.add_argument("--orig_format", default="pkl_dict",
                        choices=["pkl_dict", "pkl_dataframe"],
                        help="原数据格式：dict（SheepDog）或 dataframe（Weibo21）")
    parser.add_argument("--nli_model",
                        default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--keep_contradictions", action="store_true",
                        help="保留 contradiction；默认按论文三级过滤要求剔除")
    args = parser.parse_args()

    # 加载原数据
    with open(args.original, "rb") as f:
        orig_raw = pickle.load(f)
    
    if args.orig_format == "pkl_dict":
        orig_news = orig_raw["news"]  # SheepDog 格式
    else:
        orig_news = orig_raw["content"].tolist()  # Weibo21 DataFrame
    
    print(f"原数据: {len(orig_news)} 条")

    # 加载对抗数据
    with open(args.rewritten, "rb") as f:
        rewritten = pickle.load(f)
    print(f"对抗数据: {len(rewritten['news'])} 条")

    # 加载 NLI 模型
    print(f"\n加载 NLI 模型: {args.nli_model}")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device)
    model.eval()
    
    print(f"  NLI 标签: {model.config.id2label}")
    label_indices = resolve_nli_label_indices(model.config.id2label)
    print(f"  标签索引: {label_indices}")

    # 批量推理
    n = len(rewritten["news"])
    p_entails, p_neutrals, p_contradicts, pred_labels = [], [], [], []
    
    print(f"\n开始 NLI 推理...")
    with torch.no_grad():
        for batch_start in tqdm(range(0, n, args.batch_size)):
            batch_end = min(batch_start + args.batch_size, n)
            
            premises = []
            hypotheses = []
            for i in range(batch_start, batch_end):
                orig_idx = rewritten["orig_idx"][i]
                premises.append(orig_news[orig_idx])
                hypotheses.append(rewritten["news"][i])
            
            # NLI 输入：premise [SEP] hypothesis
            encoded = tokenizer(
                premises, hypotheses,
                max_length=args.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)
            probs_np = probs.cpu().numpy()
            p_entails.extend(probs_np[:, label_indices["entailment"]].tolist())
            p_neutrals.extend(probs_np[:, label_indices["neutral"]].tolist())
            p_contradicts.extend(probs_np[:, label_indices["contradiction"]].tolist())
            idx_to_name = {idx: name for name, idx in label_indices.items()}
            pred_labels.extend(
                idx_to_name[int(i)] for i in probs_np.argmax(axis=1)
            )

    # 保存
    keep_indices = [
        i for i, label in enumerate(pred_labels)
        if args.keep_contradictions or label != "contradiction"
    ]
    output_data = {}
    for key, values in rewritten.items():
        if isinstance(values, (list, tuple)) and len(values) == n:
            output_data[key] = [values[i] for i in keep_indices]
    output_data.update({
        "p_entail": [p_entails[i] for i in keep_indices],
        "p_neutral": [p_neutrals[i] for i in keep_indices],
        "p_contradict": [p_contradicts[i] for i in keep_indices],
        "nli_label": [pred_labels[i] for i in keep_indices],
        "nli_filter": {
            "exclude": None if args.keep_contradictions else "contradiction",
            "model": args.nli_model,
        },
    })
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    # 统计
    p_arr = np.array(output_data["p_entail"])
    n_dropped = n - len(keep_indices)
    print(f"\n=== p_entail 统计 ===")
    print(f"  NLI contradiction 剔除: {n_dropped}/{n}")
    print(f"  均值: {p_arr.mean():.4f}")
    print(f"  中位数: {np.median(p_arr):.4f}")
    print(f"  std: {p_arr.std():.4f}")
    print(f"  分布:")
    print(f"    p < 0.3: {(p_arr < 0.3).sum()} ({100*(p_arr < 0.3).sum()/len(p_arr):.1f}%)")
    print(f"    0.3-0.5: {((p_arr >= 0.3) & (p_arr < 0.5)).sum()}")
    print(f"    0.5-0.7: {((p_arr >= 0.5) & (p_arr < 0.7)).sum()}")
    print(f"    0.7-0.9: {((p_arr >= 0.7) & (p_arr < 0.9)).sum()}")
    print(f"    p >= 0.9: {(p_arr >= 0.9).sum()} ({100*(p_arr >= 0.9).sum()/len(p_arr):.1f}%)")

    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
