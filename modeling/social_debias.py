"""
SocialDebias 双分支模型

架构：
    Text → BERT Encoder → h (共享表示)
                           ├─→ 事实分支 (Fact Branch) → y_fact (真假预测)
                           └─→ GRL → 偏置分支 (Bias Branch) → y_bias (真假预测)

训练信号：
    - 事实分支：正常的分类损失，驱动共享表示学"有用"的特征
    - 偏置分支：也是分类损失，但经过 GRL，驱动共享表示"忘记"表层偏差
    - 语义一致性：事实分支表示要接近冻结的 BERT 表示（防止误删核心语义）
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from modeling.grl import GradientReversalLayer


class SocialDebiasModel(nn.Module):
    """
    双分支解耦架构，回应老师意见 5、6、9。
    """

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            num_classes: int = 2,
            hidden_dim: int = 384,
            dropout: float = 0.1,
            grl_lambda: float = 1.0,
            use_frozen_bert: bool = True,  # 是否用冻结 BERT 作为语义锚
    ):
        super().__init__()

        # 主干 BERT（可训练）
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size  # 768

        # 语义锚 BERT（冻结，用于计算语义一致性损失，回应意见 6）
        self.use_frozen_bert = use_frozen_bert
        if use_frozen_bert:
            self.frozen_bert = AutoModel.from_pretrained(model_name)
            # 冻结所有参数
            for p in self.frozen_bert.parameters():
                p.requires_grad = False
            self.frozen_bert.eval()

        # 事实分支：h → 事实特征 → 真假分类
        self.fact_projector = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fact_classifier = nn.Linear(hidden_dim, num_classes)

        # 偏置分支：h → GRL → 偏置特征 → 真假分类（会被 GRL 反向）
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.bias_projector = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bias_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        前向传播，返回多个输出供训练时计算不同的损失。

        Returns:
            dict 包含:
                'fact_logits':  [B, num_classes]  事实分支预测
                'bias_logits':  [B, num_classes]  偏置分支预测
                'shared_repr':  [B, bert_hidden]  共享 BERT 表示
                'fact_repr':    [B, hidden_dim]   事实分支的中间表示
                'bias_repr':    [B, hidden_dim]   偏置分支的中间表示
                'frozen_repr':  [B, bert_hidden]  冻结 BERT 表示（如果启用）
        """
        # 主干 BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        shared = outputs.last_hidden_state[:, 0, :]  # [CLS] 向量

        # 事实分支（正常传播）
        fact_repr = self.fact_projector(shared)
        fact_logits = self.fact_classifier(fact_repr)

        # 偏置分支（经过 GRL）
        bias_input = self.grl(shared)
        bias_repr = self.bias_projector(bias_input)
        bias_logits = self.bias_classifier(bias_repr)

        result = {
            "fact_logits": fact_logits,
            "bias_logits": bias_logits,
            "shared_repr": shared,
            "fact_repr": fact_repr,
            "bias_repr": bias_repr,
        }

        # 冻结 BERT 的表示（用于语义一致性损失）
        if self.use_frozen_bert:
            with torch.no_grad():
                frozen_outputs = self.frozen_bert(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                frozen = frozen_outputs.last_hidden_state[:, 0, :]
            result["frozen_repr"] = frozen

        return result

    def predict(self, input_ids, attention_mask):
        """
        推理时只用事实分支（去偏后的预测），回应老师意见——
        训练时对抗去偏，推理时只用干净的那条分支。
        """
        outputs = self.forward(input_ids, attention_mask)
        return outputs["fact_logits"]


def compute_losses(outputs, labels, weights=None):
    """
    计算多任务损失。

    Args:
        outputs: SocialDebiasModel 的输出 dict
        labels: [B] 真实标签
        weights: dict, 各损失的权重，默认 {fact: 1.0, bias: 1.0, consist: 0.5}

    Returns:
        total_loss, loss_dict（各分项损失，用于日志）
    """
    if weights is None:
        weights = {"fact": 1.0, "bias": 1.0, "consist": 0.5}

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    losses = {}

    # 1. 事实分支分类损失（正常交叉熵）
    losses["L_fact"] = ce(outputs["fact_logits"], labels)

    # 2. 偏置分支分类损失（会通过 GRL 反向，最终产生对抗效果）
    losses["L_bias"] = ce(outputs["bias_logits"], labels)

    # 3. 语义一致性损失（如果启用）
    if "frozen_repr" in outputs:
        # 用余弦相似度作为一致性度量
        # 注意 fact_repr 和 frozen_repr 维度不同，需要先做个投射
        # 这里简化：直接用共享表示和冻结表示的相似度
        shared = outputs["shared_repr"]
        frozen = outputs["frozen_repr"]
        cos_sim = nn.functional.cosine_similarity(shared, frozen, dim=-1)
        # 我们希望相似度接近 1，所以损失 = 1 - sim
        losses["L_consist"] = (1 - cos_sim).mean()
    else:
        losses["L_consist"] = torch.tensor(0.0, device=labels.device)

    # 总损失
    total = (
            weights["fact"] * losses["L_fact"] +
            weights["bias"] * losses["L_bias"] +
            weights["consist"] * losses["L_consist"]
    )

    losses["L_total"] = total
    return total, losses


if __name__ == "__main__":
    # 前向传播测试
    import sys

    sys.path.insert(0, ".")
    from utils.device import get_device

    device = get_device()
    print(f"设备: {device}")

    print("构建模型（首次运行会加载两份 BERT，约 800MB）...")
    model = SocialDebiasModel(
        model_name="bert-base-uncased",
        use_frozen_bert=True,
    ).to(device)

    # 冻结 BERT 保持在 CPU 以节省显存？不，保持在同一个设备
    # 注意：frozen_bert 的参数已经 requires_grad=False，不占优化器显存，只占模型本身的显存

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"可训练参数: {n_params / 1e6:.1f}M")
    print(f"冻结参数: {n_frozen / 1e6:.1f}M")

    # 模拟输入
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)

    # 前向
    outputs = model(input_ids, attention_mask)
    print(f"\n输出形状检查:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")

    # 损失计算
    total_loss, loss_dict = compute_losses(outputs, labels)
    print(f"\n损失分量:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    # 反向传播测试
    total_loss.backward()
    print("\n✅ 双分支模型前向 + 反向传播成功")