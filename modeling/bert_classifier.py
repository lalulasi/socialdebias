"""
最简 BERT 分类器：作为后续所有方法的基础类。

设计原则：
1. 结构尽量简单，让你能完全理解每一行
2. 后面所有复杂模型（双分支、对比学习、社交融合）都基于这个类扩展
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class BertClassifier(nn.Module):
    """
    最基础的 BERT 二分类器。

    架构: BERT → CLS pooling → Dropout → Linear → logits
    """

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            num_classes: int = 2,
            dropout: float = 0.1,
    ):
        super().__init__()

        # 加载预训练 BERT
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            logits: [batch_size, num_classes]
        """
        # BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 取 [CLS] 的隐状态作为整句表示
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # 分类
        x = self.dropout(cls_hidden)
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    # 测试：模型能否前向传播
    import sys

    sys.path.insert(0, ".")  # 让 Python 能找到 utils
    from utils.device import get_device

    device = get_device()
    print(f"使用设备: {device}")

    print("加载 BERT 模型（首次运行会下载，约 400MB）...")
    model = BertClassifier(model_name="bert-base-uncased")
    model = model.to(device)

    # 模拟一个 batch 的输入
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)

    print(f"\n输入形状: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")

    # 前向传播
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    print(f"输出形状: logits={logits.shape}")
    print(f"输出设备: {logits.device}")
    print(f"输出样例: {logits[0]}")
    print("\n✅ 模型前向传播成功")