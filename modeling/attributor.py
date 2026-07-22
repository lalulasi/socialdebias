"""Token-level attribution with Captum Layer Integrated Gradients."""
import torch
import numpy as np
from captum.attr import IntegratedGradients
from typing import List, Dict, Optional


class BertAttributor:
    """
    BERT 模型的归因计算器。

    用法示例：
        attributor = BertAttributor(model, tokenizer, device)
        result = attributor.attribute("This is a fake news article.", target_class=1)
        # result = {'tokens': ['This', 'is', ...], 'scores': [0.12, -0.03, ...]}
    """

    def __init__(
            self,
            model,
            tokenizer,
            device,
            n_steps: int = 50,
            internal_batch_size: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size

        if device.type == "mps":
            print("Captum attribution is running on CPU because MPS has dtype limitations.")
            self.device = torch.device("cpu")
            self.model = self.model.to("cpu")
        else:
            self.device = device

        if hasattr(model, 'bert'):
            self.embedding_layer = self.model.bert.embeddings.word_embeddings
            self.bert_model = self.model.bert
        else:
            raise ValueError("模型必须有 .bert 属性")

        self.ig = IntegratedGradients(self._forward_embeds)

    def _forward_fn(self, input_ids, attention_mask):
        """
        Return classification logits for both baseline and SocialDebias models.
        """
        if hasattr(self.model, 'predict'):
            # SocialDebiasModel 有 predict 方法（只用事实分支）
            return self.model.predict(input_ids, attention_mask)
        else:
            # BertClassifier 直接前向
            return self.model(input_ids, attention_mask)

    def _forward_embeds(self, input_embeds, attention_mask):
        """Forward from embedding vectors so the IG baseline can be zero."""
        outputs = self.bert_model(
            inputs_embeds=input_embeds, attention_mask=attention_mask
        )
        shared = outputs.last_hidden_state[:, 0, :]
        if hasattr(self.model, "fact_projector"):
            return self.model.fact_classifier(self.model.fact_projector(shared))
        if hasattr(self.model, "dropout"):
            shared = self.model.dropout(shared)
        return self.model.classifier(shared)

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
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self._forward_fn(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred_class = logits.argmax(dim=-1).item()
            pred_prob = probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        input_embeds = self.embedding_layer(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)
        attributions, delta = self.ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            target=target_class,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
            return_convergence_delta=True,
        )

        token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

        token_ids = input_ids.squeeze(0).cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())

        valid_length = attention_mask.squeeze(0).sum().item()

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
            "convergence_delta": float(delta.reshape(-1).mean().item()),
        }

    def attribute_batch(self, texts: List[str], **kwargs) -> List[Dict]:
        """Attribute a list of texts one by one."""
        return [self.attribute(text, **kwargs) for text in texts]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from transformers import AutoTokenizer
    from modeling.social_debias import SocialDebiasModel
    from utils.device import get_device

    device = get_device()
    print(f"设备: {device}")

    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SocialDebiasModel(use_frozen_bert=False).to(device)
    model.eval()

    text = "BREAKING: shocking news about the economy! This unbelievable discovery will change everything."

    print(f"\n输入文本: {text}")
    print("\n计算归因（IG n_steps=50，大约需要 10-30 秒）...")

    attributor = BertAttributor(model, tokenizer, device, n_steps=50)
    result = attributor.attribute(text, target_class=1)

    print(f"\n预测类别: {result['pred_class']} (置信度 {result['pred_prob']:.3f})")
    print(f"归因目标类: {result['target_class']}")
    print(f"收敛误差: {result['convergence_delta']:.4f}")
    print(f"\n词元级归因分数（top 5 正贡献 + top 5 负贡献）:")

    scored = list(zip(result['tokens'], result['scores']))
    scored_by_abs = sorted(scored, key=lambda x: abs(x[1]), reverse=True)
    print("Top 10 最重要的词元:")
    for tok, score in scored_by_abs[:10]:
        sign = "+" if score > 0 else "-"
        print(f"  {sign} {tok:20s} : {score:+.4f}")

    print("\n归因计算成功")
