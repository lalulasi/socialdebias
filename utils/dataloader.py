"""Dataset helpers shared by the training and evaluation scripts."""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class FakeNewsDataset(Dataset):
    """Tokenized fake-news classification samples."""

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
    torch.manual_seed(seed)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

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

    print(f"Data split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    return train_loader, val_loader, test_loader

class SurfaceAugmentedDataset(Dataset):
    """
    Precomputes and normalizes surface features for SocialDebias.

    用法:
        from utils.surface_features import SurfaceFeatureExtractor

        extractor = SurfaceFeatureExtractor()
        train_set = SurfaceAugmentedDataset(
            train_data, tokenizer, max_length=512,
            surface_extractor=extractor,
        )
        val_set = SurfaceAugmentedDataset(
            val_data, tokenizer, max_length=512,
            surface_extractor=extractor,
            normalizer=(train_set.feat_mean, train_set.feat_std),
        )
    """

    def __init__(
            self,
            data: list,
            tokenizer,
            max_length: int = 512,
            surface_extractor=None,
            normalizer: tuple = None,
    ):
        from tqdm import tqdm
        import numpy as np

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        has_speaker_feat = len(data) > 0 and "speaker_feat" in data[0]
        
        if has_speaker_feat:
            print("[SurfaceAug] using speaker_feat from samples")
            feats = np.stack([s["speaker_feat"] for s in data], axis=0).astype(np.float32)
            if normalizer is None:
                self.feat_mean = feats.mean(axis=0)
                self.feat_std = feats.std(axis=0) + 1e-6
            else:
                self.feat_mean, self.feat_std = normalizer
            self.surface_features = (feats - self.feat_mean) / self.feat_std
            return
        elif surface_extractor is None:
            self.surface_features = None
            self.feat_mean = None
            self.feat_std = None
            return
        else:
            print(f"[SurfaceAug] 预计算 {len(data)} 条样本的表层特征...")
            feats = []
            for s in tqdm(data, desc="提取表层特征"):
                feats.append(surface_extractor.extract(s["text"]))
            feats = np.stack(feats, axis=0).astype(np.float32)

            if normalizer is None:
                self.feat_mean = feats.mean(axis=0)
                self.feat_std = feats.std(axis=0) + 1e-6
                print(f"[SurfaceAug] 训练集归一化: mean={self.feat_mean.shape}, std={self.feat_std.shape}")
            else:
                self.feat_mean, self.feat_std = normalizer

            self.surface_features = (feats - self.feat_mean) / self.feat_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        encoded = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }
        if self.surface_features is not None:
            item["surface_feat"] = torch.tensor(
                self.surface_features[idx], dtype=torch.float32
            )
        return item
