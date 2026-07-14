"""SocialDebias dual-branch classifier."""
import torch
import torch.nn as nn
from transformers import AutoModel
from modeling.grl import GradientReversalLayer


class SocialDebiasModel(nn.Module):
    """BERT encoder with a fact branch, an adversarial bias branch, and an optional semantic anchor."""

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            num_classes: int = 2,
            hidden_dim: int = 384,
            dropout: float = 0.1,
            grl_lambda: float = 1.0,
            use_frozen_bert: bool = True,
            use_comment_encoder: bool = False,
            comment_model_name: str = "bert-base-chinese",
            surface_feat_dim: int = 0,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size

        self.use_frozen_bert = use_frozen_bert
        if use_frozen_bert:
            self.frozen_bert = AutoModel.from_pretrained(model_name)
            for p in self.frozen_bert.parameters():
                p.requires_grad = False
            self.frozen_bert.eval()

        self.fact_projector = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fact_classifier = nn.Linear(hidden_dim, num_classes)

        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.surface_feat_dim = surface_feat_dim
        self.bias_projector = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bias_classifier = nn.Linear(hidden_dim, num_classes)

        self.use_comment_encoder = use_comment_encoder
        if use_comment_encoder:
            from modeling.comment_encoder import CommentEncoder
            self.comment_encoder = CommentEncoder(model_name=comment_model_name)
            self.comment_fusion_gate = nn.Linear(bert_hidden * 2, 1)

        if surface_feat_dim > 0:
            surface_hidden = 128
            self.surface_proj = nn.Sequential(
                nn.Linear(surface_feat_dim, surface_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(surface_hidden, bert_hidden),
            )
            self.fusion_gate = nn.Linear(bert_hidden * 2, 1)

    def forward(self, input_ids, attention_mask, surface_feat=None, comment_input_ids=None, comment_attention_mask=None, comment_mask=None):
        """
        Return logits and intermediate representations used by the training losses.

        Returns:
            dict 包含:
                'fact_logits':  [B, num_classes]  事实分支预测
                'bias_logits':  [B, num_classes]  偏置分支预测
                'shared_repr':  [B, bert_hidden]  共享 BERT 表示
                'fact_repr':    [B, hidden_dim]   事实分支的中间表示
                'bias_repr':    [B, hidden_dim]   偏置分支的中间表示
                'frozen_repr':  [B, bert_hidden]  冻结 BERT 表示（如果启用）
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        shared = outputs.last_hidden_state[:, 0, :]

        if self.use_comment_encoder and comment_input_ids is not None:
            comment_repr = self.comment_encoder(
                comment_input_ids, comment_attention_mask, comment_mask
            )
            gate = torch.sigmoid(
                self.comment_fusion_gate(torch.cat([shared, comment_repr], dim=-1))
            )
            shared = gate * shared + (1 - gate) * comment_repr

        fact_repr = self.fact_projector(shared)
        fact_logits = self.fact_classifier(fact_repr)

        bias_input = self.grl(shared)

        if self.surface_feat_dim > 0 and surface_feat is not None:
            surface_proj = self.surface_proj(surface_feat)
            gate = torch.sigmoid(
                self.fusion_gate(torch.cat([bias_input, surface_proj], dim=-1))
            )
            bias_input = gate * bias_input + (1 - gate) * surface_proj

        bias_repr = self.bias_projector(bias_input)
        bias_logits = self.bias_classifier(bias_repr)

        result = {
            "fact_logits": fact_logits,
            "bias_logits": bias_logits,
            "shared_repr": shared,
            "fact_repr": fact_repr,
            "bias_repr": bias_repr,
        }

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
        Inference uses the fact branch.
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

    losses["L_fact"] = ce(outputs["fact_logits"], labels)

    losses["L_bias"] = ce(outputs["bias_logits"], labels)

    if "frozen_repr" in outputs:
        shared = outputs["shared_repr"]
        frozen = outputs["frozen_repr"]
        cos_sim = nn.functional.cosine_similarity(shared, frozen, dim=-1)
        losses["L_consist"] = (1 - cos_sim).mean()
    else:
        losses["L_consist"] = torch.tensor(0.0, device=labels.device)

    total = (
            weights["fact"] * losses["L_fact"] +
            weights["bias"] * losses["L_bias"] +
            weights["consist"] * losses["L_consist"]
    )

    losses["L_total"] = total
    return total, losses


if __name__ == "__main__":
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"可训练参数: {n_params / 1e6:.1f}M")
    print(f"冻结参数: {n_frozen / 1e6:.1f}M")

    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)

    outputs = model(input_ids, attention_mask)
    print(f"\n输出形状检查:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")

    total_loss, loss_dict = compute_losses(outputs, labels)
    print(f"\n损失分量:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    total_loss.backward()
    print("\n双分支模型前向 + 反向传播成功")
