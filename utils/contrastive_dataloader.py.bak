"""
对比学习数据加载器 - 读 Qwen 对抗改写数据 + 原数据，构造正样本对
对应大纲 3.4.2 节 InfoNCE 损失。

使用:
    from utils.contrastive_dataloader import ContrastiveFakeNewsDataset

    dataset = ContrastiveFakeNewsDataset(
        original_data,         # SheepDog 原训练数据 List[Dict]
        adversarial_data,      # Qwen 改写数据（filtered）
        tokenizer,
        max_length=512,
    )
"""
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
            original_samples: List[Dict],  # [{text, label, ...}]
            adversarial_pkl_path: str,  # Qwen 改写后的 pkl 路径（filter 后的）
            tokenizer,
            max_length: int = 512,
            styles: Optional[List[str]] = None,  # None 表示用所有可用风格
            require_adv: bool = True,  # True: 没有对抗版本的样本会被丢弃
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载对抗数据，按 orig_idx 分组
        with open(adversarial_pkl_path, "rb") as f:
            adv_data = pickle.load(f)

        # 构造 orig_idx -> List[改写文本]
        self.adv_by_idx = defaultdict(list)
        for i, idx in enumerate(adv_data["orig_idx"]):
            if styles and adv_data["style"][i] not in styles:
                continue
            self.adv_by_idx[idx].append(adv_data["news"][i])

        # 过滤原始样本：必须有对抗版本（如果 require_adv=True）
        self.samples = []
        n_total = len(original_samples)
        n_skipped = 0
        for sample in original_samples:
            # 注意：original_samples 来自 load_dataset()，可能没有 orig_idx 字段
            # 我们用样本在原始 pkl 中的位置作为 idx
            # 需要确保 original_samples 的顺序和 pkl 一致
            pass

        # 由于 load_dataset 切了 train/val 后顺序乱了，我们用 'id' 字段或文本匹配
        # 这里采用更稳的策略：用原文 hash 匹配
        # 但更简单的做法是：直接用 orig pkl 文件而非 load_dataset 的输出

        # ★ 推荐用法：传 adversarial_pkl 时同时传原始 pkl 路径
        # 而不是 List[Dict]
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
        """工厂方法：从两个 pkl 路径直接构造数据集。"""
        instance = cls.__new__(cls)
        instance.tokenizer = tokenizer
        instance.max_length = max_length

        # 加载原数据
        with open(original_pkl_path, "rb") as f:
            orig = pickle.load(f)

        # 加载对抗数据（可能含 p_entail 字段）
        with open(adversarial_pkl_path, "rb") as f:
            adv = pickle.load(f)
        has_p_entail = "p_entail" in adv

        # 按 orig_idx 分组对抗数据
        adv_by_idx = defaultdict(list)
        for i, idx in enumerate(adv["orig_idx"]):
            if styles and adv["style"][i] not in styles:
                continue
            adv_by_idx[idx].append(adv["news"][i])

        # 构造样本
        instance.samples = []
        n_skipped = 0
        for idx, (text, label) in enumerate(zip(orig["news"], orig["labels"])):
            adv_versions = adv_by_idx.get(idx, [])
            if require_adv and not adv_versions:
                n_skipped += 1
                continue
            # 如果有 p_entail，按 orig_idx 分组取均值（一个原文有多个对抗版本时）
            p_entail_for_orig = None
            if has_p_entail:
                p_entails = [adv["p_entail"][i] for i, oi in enumerate(adv["orig_idx"]) if oi == idx]
                if p_entails:
                    p_entail_for_orig = sum(p_entails) / len(p_entails)
            
            sample = {
                "orig_idx": idx,
                "orig_text": text,
                "label": int(label),
                "adv_versions": adv_versions,
            }
            if p_entail_for_orig is not None:
                sample["p_entail"] = p_entail_for_orig
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

        # 编码原文
        orig_enc = self.tokenizer(
            s["orig_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 随机选一个对抗版本
        if s["adv_versions"]:
            adv_text = random.choice(s["adv_versions"])
        else:
            adv_text = s["orig_text"]  # fallback

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
        # 如果有 p_entail，返回作为软标签权重
        if "p_entail" in s:
            result["p_entail"] = torch.tensor(s["p_entail"], dtype=torch.float32)
        else:
            result["p_entail"] = torch.tensor(1.0, dtype=torch.float32)
        return result


def create_contrastive_dataloaders(
        config,
        tokenizer,
        seed: int = 42,
        val_ratio: float = 0.15,
):
    """
    创建对比学习的 DataLoader。
    Train 用 ContrastiveDataset，Val/Test 用普通 FakeNewsDataset（无对抗）。
    """
    import sys
    sys.path.insert(0, ".")
    from torch.utils.data import DataLoader
    from utils.dataloader import FakeNewsDataset  # 复用现有的
    from utils.real_dataloader import load_dataset

    # 加载原数据（用于 val/test）
    train_data, val_data, test_data = load_dataset(
        dataset_name=config.dataset_name,
        seed=seed,
        val_ratio=val_ratio,
    )

    # 训练集用对比学习版（直接读 pkl，不用 train_data 列表）
    orig_pkl = f"data/sheepdog/news_articles/{config.dataset_name}_train.pkl"
    adv_pkl = config.adversarial_pkl_path  # 从 config 读

    train_set = ContrastiveFakeNewsDataset.from_pkl(
        original_pkl_path=orig_pkl,
        adversarial_pkl_path=adv_pkl,
        tokenizer=tokenizer,
        max_length=config.max_length,
        styles=getattr(config, "contrast_styles", None),  # None 表示全部
    )

    # ★ 注意：train_set 包含全部 360 条，没有 val 切分
    # val 和 test 用 load_dataset 切出来的部分
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