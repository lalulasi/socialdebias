"""Dataset for original/adversarial positive pairs used by InfoNCE."""
import random
import pickle
from collections import defaultdict
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset


class ContrastiveFakeNewsDataset(Dataset):
    """
    返回原文 + 随机选一个对抗版本，作为对比学习的正样本对。

    每个 batch item 包含：
        orig_input_ids:    原文 token
        orig_attention_mask:
        adv_input_ids:     对抗版本 token
        adv_attention_mask:
        label:             标签（原文和对抗共用）
    """

    def __init__(
            self,
            original_samples: List[Dict],
            adversarial_pkl_path: str,
            tokenizer,
            max_length: int = 512,
            styles: Optional[List[str]] = None,
            require_adv: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(adversarial_pkl_path, "rb") as f:
            adv_data = pickle.load(f)

        self.adv_by_idx = defaultdict(list)
        for i, idx in enumerate(adv_data["orig_idx"]):
            if styles and adv_data["style"][i] not in styles:
                continue
            self.adv_by_idx[idx].append(adv_data["news"][i])

        self.samples = []
        n_total = len(original_samples)
        n_skipped = 0
        for sample in original_samples:
            pass

        raise NotImplementedError(
            "请使用 ContrastiveFakeNewsDataset.from_pkl 工厂方法初始化，"
            "用原始 pkl 路径而不是 List[Dict]"
        )

    @classmethod
    def from_pkl(
            cls,
            original_pkl_path: str,
            adversarial_pkl_path: str,
            tokenizer,
            max_length: int = 512,
            styles: Optional[List[str]] = None,
            require_adv: bool = True,
    ):
        """Build the dataset from original and adversarial pkl files."""
        instance = cls.__new__(cls)
        instance.tokenizer = tokenizer
        instance.max_length = max_length

        with open(original_pkl_path, "rb") as f:
            orig = pickle.load(f)

        if hasattr(orig, "columns"):  # 鸭子类型判断 pandas.DataFrame
            if "content" in orig.columns and "label" in orig.columns:
                orig = {
                    "news": orig["content"].tolist(),
                    "labels": orig["label"].tolist(),
                }
            else:
                raise ValueError(
                    f"原始 DataFrame 缺少必要列，期望 'content' 与 'label'，"
                    f"实际列: {list(orig.columns)}"
                )
        with open(adversarial_pkl_path, "rb") as f:
            adv = pickle.load(f)
        has_p_entail = "p_entail" in adv
        instance.has_p_entail = has_p_entail

        adv_by_idx = defaultdict(list)
        for i, idx in enumerate(adv["orig_idx"]):
            if styles and adv["style"][i] not in styles:
                continue
            p_entail = float(adv["p_entail"][i]) if has_p_entail else 1.0
            adv_by_idx[idx].append((adv["news"][i], p_entail))

        instance.samples = []
        n_skipped = 0
        for idx, (text, label) in enumerate(zip(orig["news"], orig["labels"])):
            adv_versions = adv_by_idx.get(idx, [])
            if require_adv and not adv_versions:
                n_skipped += 1
                continue
            sample = {
                "orig_idx": idx,
                "orig_text": text,
                "label": int(label),
                "adv_versions": adv_versions,
            }
            instance.samples.append(sample)

        print(f"[ContrastiveDataset] 加载完成")
        print(f"  原数据: {len(orig['news'])} 条")
        print(f"  对抗数据: {len(adv['news'])} 条")
        print(f"  最终样本: {len(instance.samples)} 条")
        print(f"  跳过（无对抗）: {n_skipped} 条")
        if instance.samples:
            avg_adv = sum(len(s["adv_versions"]) for s in instance.samples) / len(instance.samples)
            print(f"  每条样本平均对抗版本数: {avg_adv:.2f}")

        return instance

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]

        orig_enc = self.tokenizer(
            s["orig_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if s["adv_versions"]:
            # p_entail 必须与本次随机抽到的改写版本一一对应，不能按原文
            # 对所有风格求均值，否则 NLI 样本权重会错配。
            adv_text, p_entail = random.choice(s["adv_versions"])
        else:
            adv_text = s["orig_text"]
            p_entail = 1.0

        adv_enc = self.tokenizer(
            adv_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "orig_input_ids": orig_enc["input_ids"].squeeze(0),
            "orig_attention_mask": orig_enc["attention_mask"].squeeze(0),
            "adv_input_ids": adv_enc["input_ids"].squeeze(0),
            "adv_attention_mask": adv_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(s["label"], dtype=torch.long),
            "orig_idx": s["orig_idx"],
        }
        result["p_entail"] = torch.tensor(p_entail, dtype=torch.float32)
        return result


def create_contrastive_dataloaders(
        config,
        tokenizer,
        seed: int = 42,
        val_ratio: float = 0.15,
):
    """
    Create dataloaders for contrastive training and plain validation/testing.
    """
    import sys
    sys.path.insert(0, ".")
    from torch.utils.data import DataLoader
    from utils.dataloader import FakeNewsDataset  # 复用现有的
    from utils.real_dataloader import load_dataset

    train_data, val_data, test_data = load_dataset(
        dataset_name=config.dataset_name,
        seed=seed,
        val_ratio=val_ratio,
    )

    orig_pkl = f"data/sheepdog/news_articles/{config.dataset_name}_train.pkl"
    adv_pkl = config.adversarial_pkl_path

    train_set = ContrastiveFakeNewsDataset.from_pkl(
        original_pkl_path=orig_pkl,
        adversarial_pkl_path=adv_pkl,
        tokenizer=tokenizer,
        max_length=config.max_length,
        styles=getattr(config, "contrast_styles", None),
    )

    val_set = FakeNewsDataset(val_data, tokenizer, config.max_length)
    test_set = FakeNewsDataset(test_data, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # smoke test
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = ContrastiveFakeNewsDataset.from_pkl(
        original_pkl_path="data/sheepdog/news_articles/politifact_train.pkl",
        adversarial_pkl_path="data/qwen_adv/politifact_train_adv_filtered.pkl",
        tokenizer=tokenizer,
        max_length=512,
    )

    print(f"\n样本量: {len(dataset)}")
    sample = dataset[0]
    print(f"\n第 0 条:")
    print(f"  orig_input_ids shape: {sample['orig_input_ids'].shape}")
    print(f"  adv_input_ids shape: {sample['adv_input_ids'].shape}")
    print(f"  label: {sample['label']}")
    print(f"  原文前 100 token: {tokenizer.decode(sample['orig_input_ids'][:100])}")
    print(f"  对抗前 100 token: {tokenizer.decode(sample['adv_input_ids'][:100])}")
