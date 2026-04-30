"""
评论编码器：用独立 BERT 对评论列表编码，pooling 后输出固定维度向量
对应论文 3.6 节"评论语义编码"
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class CommentEncoder(nn.Module):
    """
    评论列表 → 768 维社交语义向量
    
    输入: comment_input_ids [B, K, L], comment_attention_mask [B, K, L]
        其中 K = max_comments_per_news (默认 5), L = max_comment_length
    输出: comment_repr [B, 768]
    
    实现: 共享 BERT 编码每条评论 → 取 [CLS] → 跨评论 mean pooling
    """
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        freeze: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
    
    def forward(self, comment_input_ids, comment_attention_mask, comment_mask=None):
        """
        Args:
            comment_input_ids: [B, K, L]
            comment_attention_mask: [B, K, L]  
            comment_mask: [B, K] - 1 表示这条评论存在，0 表示是 padding
                         （处理评论数量不一致：有的新闻有 5 条评论，有的只有 2 条）
        Returns:
            comment_repr: [B, hidden_size]
        """
        B, K, L = comment_input_ids.shape
        
        # Reshape 为 [B*K, L] 一次性 forward
        flat_input = comment_input_ids.reshape(B * K, L)
        flat_mask = comment_attention_mask.reshape(B * K, L)
        
        outputs = self.bert(input_ids=flat_input, attention_mask=flat_mask)
        # [CLS] 表示
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [B*K, hidden]
        
        # Reshape 回 [B, K, hidden]
        cls_repr = cls_repr.reshape(B, K, self.hidden_size)
        
        # 跨评论 pooling（masked mean pooling）
        if comment_mask is not None:
            mask = comment_mask.unsqueeze(-1).float()  # [B, K, 1]
            sum_repr = (cls_repr * mask).sum(dim=1)  # [B, hidden]
            count = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
            comment_repr = sum_repr / count
        else:
            comment_repr = cls_repr.mean(dim=1)  # [B, hidden]
        
        return comment_repr