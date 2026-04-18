"""
PyTorch 数据加载器：把原始数据转成模型可用的 batch。

设计原则：
1. 数据格式和真实数据集（PolitiFact、GossipCop、Weibo）保持一致的接口
2. 同一个 Dataset 类，既能加载假数据，也能加载真实数据
3. 自动处理 tokenize、padding、转 tensor 等繁琐步骤
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Optional


class FakeNewsDataset(Dataset):
    """虚假新闻检测数据集"""

    def __init__(
            self,
            data: List[Dict],
            tokenizer,
            max_length: int = 256,
    ):
        """
        Args:
            data: 样本列表，每条包含 'text', 'label' 等字段
            tokenizer: HuggingFace tokenizer
            max_length: 文本最大长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        # 把文本 tokenize 成模型输入
        encoded = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


def create_dataloaders(
        data: List[Dict],
        tokenizer,
        batch_size: int = 8,
        max_length: int = 256,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
):
    """
    把数据切分成 train/val/test，并创建 DataLoader。

    Returns:
        train_loader, val_loader, test_loader
    """
    # 固定随机种子，保证每次切分一样
    torch.manual_seed(seed)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # 简单切分（真实场景应该按时间或者按类别分层）
    indices = torch.randperm(n).tolist()
    train_data = [data[i] for i in indices[:n_train]]
    val_data = [data[i] for i in indices[n_train:n_train + n_val]]
    test_data = [data[i] for i in indices[n_train + n_val:]]

    train_set = FakeNewsDataset(train_data, tokenizer, max_length)
    val_set = FakeNewsDataset(val_data, tokenizer, max_length)
    test_set = FakeNewsDataset(test_data, tokenizer, max_length)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"📊 数据切分: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试：加载假数据并创建 DataLoader
    from dummy_data import load_dummy_dataset

    # 加载数据
    data = load_dummy_dataset()
    print(f"加载了 {len(data)} 条数据")

    # 创建 tokenizer
    print("加载 tokenizer（首次运行会下载 BERT 模型，约 400MB）...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 创建 DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        data, tokenizer, batch_size=4, max_length=128
    )

    # 看一眼第一个 batch 长什么样
    print("\n第一个 batch:")
    for batch in train_loader:
        print(f"  input_ids 形状: {batch['input_ids'].shape}")
        print(f"  attention_mask 形状: {batch['attention_mask'].shape}")
        print(f"  label 形状: {batch['label'].shape}")
        print(f"  label 内容: {batch['label'].tolist()}")
        break

    print("\n✅ DataLoader 测试通过")


def create_dataloaders_auto(
        config,
        tokenizer,
        seed: int = 42,
):
    """
    根据配置自动加载假数据或真实数据。
    这是后续训练脚本统一使用的入口。
    """
    import sys
    sys.path.insert(0, ".")

    if config.use_dummy_data:
        # 加载假数据
        from utils.dummy_data import load_dummy_dataset, generate_dummy_dataset
        import os

        dummy_path = f"./data/dummy/dummy_data_{config.language}.json"
        if not os.path.exists(dummy_path):
            n = config.max_train_samples or 100
            generate_dummy_dataset(n_samples=n, language=config.language)
        data = load_dummy_dataset(path=dummy_path)

        return create_dataloaders(
            data, tokenizer,
            batch_size=config.batch_size,
            max_length=config.max_length,
            seed=seed,
        )
    else:
        # 加载真实数据
        from utils.real_dataloader import load_dataset
        from torch.utils.data import DataLoader

        train_data, val_data, test_data = load_dataset(
            dataset_name=config.dataset_name,
            max_train_samples=config.max_train_samples,
            seed=seed,
        )

        train_set = FakeNewsDataset(train_data, tokenizer, config.max_length)
        val_set = FakeNewsDataset(val_data, tokenizer, config.max_length)
        test_set = FakeNewsDataset(test_data, tokenizer, config.max_length)

        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader