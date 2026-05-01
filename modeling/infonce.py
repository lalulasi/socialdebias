"""
InfoNCE 损失（对比学习）
对应论文大纲 3.4.2 节
"""
import torch
import torch.nn.functional as F


def info_nce_loss(
    anchor_emb: torch.Tensor,    # [B, D] 原文 fact_repr
    positive_emb: torch.Tensor,  # [B, D] 对抗版本 fact_repr
    temperature: float = 0.07,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss (双向对比，类似 SimCLR)。
    
    每个 anchor_i 的正样本是 positive_i，
    负样本是 batch 内其他所有 positive_j (j != i)。
    
    Args:
        anchor_emb:   [B, D] 原文表示
        positive_emb: [B, D] 对抗版本表示
        temperature:  对比温度（推荐 0.07-0.1）
        normalize:    是否 L2 归一化（推荐 True）
    Returns:
        scalar loss
    """
    if normalize:
        anchor_emb = F.normalize(anchor_emb, dim=-1)
        positive_emb = F.normalize(positive_emb, dim=-1)
    
    batch_size = anchor_emb.size(0)
    
    # 相似度矩阵 [B, B]
    # sim[i][j] = sim(anchor_i, positive_j)
    sim = torch.matmul(anchor_emb, positive_emb.T) / temperature
    
    # 正对在对角线上：anchor_i 配 positive_i
    labels = torch.arange(batch_size, device=anchor_emb.device)
    
    # 双向损失：anchor->positive + positive->anchor
    loss_i = F.cross_entropy(sim, labels)
    loss_j = F.cross_entropy(sim.T, labels)
    
    return (loss_i + loss_j) / 2

def info_nce_loss_weighted(
    anchor_emb,
    positive_emb,
    weights,
    temperature=0.07,
    normalize=True,
):
    """
    带样本权重的 InfoNCE 损失（用于软标签机制）
    
    Args:
        anchor_emb: [B, D] 原文表示
        positive_emb: [B, D] 对抗表示
        weights: [B] 每个样本的权重（如 max(0.5, p_entail)）
        temperature: 对比温度
        normalize: 是否 L2 归一化
    Returns:
        scalar 加权 loss
    """
    import torch
    import torch.nn.functional as F
    
    if normalize:
        anchor_emb = F.normalize(anchor_emb, dim=-1)
        positive_emb = F.normalize(positive_emb, dim=-1)
    
    batch_size = anchor_emb.size(0)
    sim = torch.matmul(anchor_emb, positive_emb.T) / temperature
    labels = torch.arange(batch_size, device=anchor_emb.device)
    
    # 双向损失（不 reduce）
    loss_i = F.cross_entropy(sim, labels, reduction='none')
    loss_j = F.cross_entropy(sim.T, labels, reduction='none')
    
    # 加权平均
    weights = weights.to(anchor_emb.device)
    weighted_loss = (weights * (loss_i + loss_j) / 2).sum() / weights.sum().clamp(min=1e-9)
    return weighted_loss
