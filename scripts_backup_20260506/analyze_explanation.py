"""
解释一致性分析脚本

输入：一对文本 (原文 P, 改写文 P')
输出：模型对两者的归因分数 + 一致性指标

这个脚本是后续评估的核心组件——可以用来：
1. 对单个样本做可视化分析（答辩时展示）
2. 在测试集上批量计算一致性指标（第五章实验数据）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from transformers import AutoTokenizer

from modeling.social_debias import SocialDebiasModel
from modeling.attributor import BertAttributor
from utils.explanation_metrics import compute_all_metrics
from utils.device import get_device


def analyze_pair(
        original_text: str,
        rewritten_text: str,
        model,
        tokenizer,
        device,
        target_class: int = 1,
        n_steps: int = 50,
        top_k: int = 10,
):
    """分析一对文本的解释一致性"""

    attributor = BertAttributor(model, tokenizer, device, n_steps=n_steps)

    # 对两个版本分别计算归因
    print("\n[1/3] 计算原文归因...")
    result_a = attributor.attribute(original_text, target_class=target_class)

    print("[2/3] 计算改写文归因...")
    result_b = attributor.attribute(rewritten_text, target_class=target_class)

    # 计算一致性指标
    print("[3/3] 计算一致性指标...")
    metrics = compute_all_metrics(
        result_a["tokens"], result_a["scores"],
        result_b["tokens"], result_b["scores"],
        k=top_k,
    )

    return {
        "original": result_a,
        "rewritten": result_b,
        "metrics": metrics,
    }


def print_result(analysis: dict, top_n: int = 8):
    """漂亮地打印分析结果"""
    orig = analysis["original"]
    rewr = analysis["rewritten"]
    metrics = analysis["metrics"]

    print("\n" + "=" * 70)
    print("解释一致性分析报告")
    print("=" * 70)

    # 模型预测情况
    print("\n【模型预测】")
    print(f"  原文  → 预测类={orig['pred_class']} (置信度 {orig['pred_prob']:.3f})")
    print(f"  改写文 → 预测类={rewr['pred_class']} (置信度 {rewr['pred_prob']:.3f})")

    prediction_changed = orig['pred_class'] != rewr['pred_class']
    if prediction_changed:
        print(f"  ⚠️  预测翻转了！（攻击成功）")
    else:
        print(f"  ✓ 预测保持一致")

    # 一致性指标
    print("\n【一致性指标】")
    print(f"  Top-{10} 归因重合度: {metrics['top_k_overlap']:.4f}  (越高越一致)")
    print(f"  Spearman 秩相关:    {metrics['spearman']:.4f}  (越接近 1 越一致)")
    print(f"  JS 散度:           {metrics['js_divergence']:.4f}  (越低越一致)")
    print(f"  共同词元数:        {metrics['common_tokens_count']}")

    # Top-N 重要词元对比
    def top_tokens(result, n):
        scored = list(zip(result["tokens"], result["scores"]))
        return sorted(scored, key=lambda x: abs(x[1]), reverse=True)[:n]

    print(f"\n【原文 Top-{top_n} 关键词元】")
    for tok, score in top_tokens(orig, top_n):
        bar = "█" * min(20, int(abs(score) * 40))
        sign = "+" if score > 0 else "-"
        print(f"  {sign} {tok:20s} {score:+.4f}  {bar}")

    print(f"\n【改写文 Top-{top_n} 关键词元】")
    for tok, score in top_tokens(rewr, top_n):
        bar = "█" * min(20, int(abs(score) * 40))
        sign = "+" if score > 0 else "-"
        print(f"  {sign} {tok:20s} {score:+.4f}  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--n_steps", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    print(f"设备: {device}")

    # 加载模型（这里用未训练的模型做演示；真实场景会加载训练好的权重）
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SocialDebiasModel(
        model_name=args.model_name,
        use_frozen_bert=False,  # 推理不需要冻结 BERT
    ).to(device)
    model.eval()
    # 加载训练好的权重（如果存在）
    ckpt_path = "./results/models/socialdebias_en.pt"
    if os.path.exists(ckpt_path):
        print(f"加载训练好的权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        # 注意：use_frozen_bert=False 的模型只有主 BERT，而保存时 use_frozen_bert=True 的模型有两份
        # 我们只加载主 BERT 相关的权重，忽略 frozen_bert 的权重
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        print("✓ 权重加载成功")
    else:
        print(f"未找到训练好的模型 ({ckpt_path})，使用随机初始化")
    model.eval()
    # 用两条文本测试：一条假新闻 + 它的"中立化改写"
    original = (
        "BREAKING: SHOCKING news about the economy! "
        "This UNBELIEVABLE discovery will CHANGE EVERYTHING you thought you knew. "
        "Click now to see the truth they don't want you to see!"
    )
    rewritten = (
        "According to a recent report, new economic data has been released. "
        "Analysts suggest the findings may lead to some revisions in current understanding. "
        "Further investigation is recommended for a comprehensive evaluation."
    )

    print(f"\n原文长度: {len(original.split())} 词")
    print(f"改写文长度: {len(rewritten.split())} 词")
    print(f"\n开始分析（每次归因约 10-30 秒，总共约 30-60 秒）...")

    analysis = analyze_pair(
        original, rewritten,
        model, tokenizer, device,
        target_class=1,  # 对"假"这个类做归因
        n_steps=args.n_steps,
    )

    print_result(analysis)

    print("\n✅ 分析完成")


if __name__ == "__main__":
    main()