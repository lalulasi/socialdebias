"""
BiLSTM 文本分类器，作为对比基线（意见 19）。
呼应开题报告里的"序列模型"承诺。
"""
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embedding: torch.Tensor = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_embedding is not None:
            self.embedding.weight.data.copy_(pretrained_embedding)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)            # [B, L, E]
        lstm_out, _ = self.lstm(emb)               # [B, L, 2H]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).bool()
            lstm_out = lstm_out.masked_fill(~mask, float("-inf"))

        pooled, _ = lstm_out.max(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits