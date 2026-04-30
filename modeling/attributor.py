"""
归因计算器：使用 Captum 的 Layer Integrated Gradients 计算 BERT 模型的词元级归因分数。

为什么选 Integrated Gradients（回应意见 16）：
1. 满足完整性公理（Completeness）：所有归因分数之和等于预测分数差
2. 满足敏感性公理（Sensitivity）：相关特征一定有非零归因
3. 是 BERT 类模型上最常用、最严谨的归因方法之一

注意：IG 需要对每个样本跑多次前向（n_steps=50 次），计算开销大，
只在评估时用，不在训练时用。
"""
import torch
import numpy as np
from captum.attr import LayerIntegratedGradients
from typing import List, Dict, Optional


class BertAttributor:
    """
    BERT 模型的归因计算器。

    用法示例：
        attributor = BertAttributor(model, tokenizer, device)
        result = attributor.attribute("This is a fake news article.", target_class=1)
        # result = {'tokens': ['This', 'is', ...], 'scores': [0.12, -0.03, ...]}
    """

    def __init__(self, model, tokenizer, device, n_steps: int = 50):
        self.model = model
        self.tokenizer = tokenizer
        self.n_steps = n_steps

        # MPS 上 Captum 有 float64 兼容问题，强制用 CPU 做归因
        if device.type == "mps":
            print("⚠️  MPS 上 Captum 有兼容性问题，归因计算切换到 CPU（训练时仍用 MPS）")
            self.device = torch.device("cpu")
            self.model = self.model.to("cpu")
        else:
            self.device = device

        if hasattr(model, 'bert'):
            self.embedding_layer = self.model.bert.embeddings.word_embeddings
            self.bert_model = self.model.bert
        else:
            raise ValueError("模型必须有 .bert 属性")

        self.lig = LayerIntegratedGradients(self._forward_fn, self.embedding_layer)

    def _forward_fn(self, input_ids, attention_mask):
        """
        统一的前向函数：不管是基线 BertClassifier 还是 SocialDebiasModel，
        都只取"真假预测"的 logits 作为归因的目标。
        """
        if hasattr(self.model, 'predict'):
            # SocialDebiasModel 有 predict 方法（只用事实分支）
            return self.model.predict(input_ids, attention_mask)
        else:
            # BertClassifier 直接前向
            return self.model(input_ids, attention_mask)

    def attribute(
            self,
            text: str,
            target_class: Optional[int] = None,
            max_length: int = 256,
    ) -> Dict:
        """
        对单个文本计算归因分数。

        Args:
            text: 原始文本
            target_class: 对哪个类做归因（0=真, 1=假）。None 表示用模型预测的类。
            max_length: 序列最大长度

        Returns:
            {
                'tokens':   [str]  分词结果（去除 special tokens）
                'scores':   [float] 每个 token 的归因分数
                'pred_class': int  模型预测的类
                'pred_prob':  float 预测置信度
            }
        """
        # 1. Tokenize
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # 2. 先跑一次前向，拿到模型的预测
        self.model.eval()
        with torch.no_grad():
            logits = self._forward_fn(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred_class = logits.argmax(dim=-1).item()
            pred_prob = probs[0, pred_class].item()

        # 如果没指定 target，就用模型的预测类
        if target_class is None:
            target_class = pred_class

        # 3. 构造 baseline（用 [PAD] token 作为"无信息基线"）
        # IG 的核心思想：从 baseline 到 real input 做路径积分
        pad_token_id = self.tokenizer.pad_token_id
        baseline_ids = torch.full_like(input_ids, pad_token_id)
        # 保留 [CLS] 和 [SEP] 位置不变（这些不是语义词元）
        baseline_ids[:, 0] = input_ids[:, 0]  # [CLS]
        # 找到 [SEP] 的位置（第一个 pad 之前的位置）
        sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)
        if len(sep_positions[1]) > 0:
            sep_pos = sep_positions[1][0].item()
            baseline_ids[:, sep_pos] = input_ids[:, sep_pos]  # [SEP]

        # 4. 计算 IG 归因
        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target_class,
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )

        # 5. 把每个 token 在 embedding 各维度上的归因聚合成一个标量
        # attributions shape: [1, seq_len, hidden_size]
        token_scores = attributions.sum(dim=-1).squeeze(0).cpu().numpy()

        # 6. 解码 tokens，过滤掉 padding
        token_ids = input_ids.squeeze(0).cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())

        # 找到有效长度（attention_mask 为 1 的位置）
        valid_length = attention_mask.squeeze(0).sum().item()

        # 过滤掉 padding 和 special tokens
        filtered_tokens = []
        filtered_scores = []
        for i in range(valid_length):
            tok = tokens[i]
            if tok in (self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token):
                continue
            filtered_tokens.append(tok)
            filtered_scores.append(float(token_scores[i]))

        return {
            "tokens": filtered_tokens,
            "scores": filtered_scores,
            "pred_class": pred_class,
            "pred_prob": pred_prob,
            "target_class": target_class,
            "convergence_delta": delta.item(),  # IG 收敛误差，越小越好
        }

    def attribute_batch(self, texts: List[str], **kwargs) -> List[Dict]:
        """对一批文本逐个计算归因（IG 本身无法批量）"""
        return [self.attribute(text, **kwargs) for text in texts]


if __name__ == "__main__":
    # 测试：加载训练好的模型 + 一条假新闻，看看归因分数
    import sys

    sys.path.insert(0, ".")
    from transformers import AutoTokenizer
    from modeling.social_debias import SocialDebiasModel
    from utils.device import get_device

    device = get_device()
    print(f"设备: {device}")

    # 加载未训练的模型做快速测试
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SocialDebiasModel(use_frozen_bert=False).to(device)
    model.eval()

    # 用一条假新闻测试
    text = "BREAKING: shocking news about the economy! This unbelievable discovery will change everything."

    print(f"\n输入文本: {text}")
    print("\n计算归因（IG n_steps=50，大约需要 10-30 秒）...")

    attributor = BertAttributor(model, tokenizer, device, n_steps=50)
    result = attributor.attribute(text, target_class=1)  # 对"假"这个类做归因

    print(f"\n预测类别: {result['pred_class']} (置信度 {result['pred_prob']:.3f})")
    print(f"归因目标类: {result['target_class']}")
    print(f"收敛误差: {result['convergence_delta']:.4f}")
    print(f"\n词元级归因分数（top 5 正贡献 + top 5 负贡献）:")

    # 按分数绝对值排序显示
    scored = list(zip(result['tokens'], result['scores']))
    scored_by_abs = sorted(scored, key=lambda x: abs(x[1]), reverse=True)
    print("Top 10 最重要的词元:")
    for tok, score in scored_by_abs[:10]:
        sign = "+" if score > 0 else "-"
        print(f"  {sign} {tok:20s} : {score:+.4f}")

    print("\n✅ 归因计算成功")