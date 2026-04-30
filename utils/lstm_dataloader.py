"""
LSTM 基线专用词表构建 + 数据集。
复用 utils.real_dataloader.load_dataset 加载 SheepDog 数据。
"""
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

_TOKEN_RE = re.compile(r"\b\w+\b|[.,!?;:\"']")


def simple_tokenize(text: str, lowercase: bool = True) -> list:
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def build_vocab(texts, max_vocab_size: int = 30000, min_freq: int = 2):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    most_common = [w for w, c in counter.most_common(max_vocab_size - 2) if c >= min_freq]
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for i, w in enumerate(most_common, start=2):
        vocab[w] = i
    return vocab


def encode(text: str, vocab: dict, max_len: int = 128):
    tokens = simple_tokenize(text)
    ids = [vocab.get(t, UNK_IDX) for t in tokens[:max_len]]
    length = len(ids)
    ids = ids + [PAD_IDX] * (max_len - length)
    mask = [1] * length + [0] * (max_len - length)
    return ids, mask


class LSTMTextDataset(Dataset):
    """
    接受 List[Dict]（每个 dict 有 'text' 和 'label'），
    和 load_dataset 返回的样本格式对齐。
    """
    def __init__(self, samples, vocab, max_len: int = 128):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ids, mask = encode(s["text"], self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }


def load_glove(glove_path: str, vocab: dict, embed_dim: int = 300) -> torch.Tensor:
    """加载 GloVe 向量，未登录词保持随机初始化。可选。"""
    embeddings = torch.randn(len(vocab), embed_dim) * 0.1
    embeddings[PAD_IDX] = 0

    if not Path(glove_path).exists():
        print(f"[LSTM] GloVe 文件不存在：{glove_path}，用随机初始化")
        return embeddings

    hit = 0
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                vec = torch.tensor([float(x) for x in parts[1:]])
                if len(vec) == embed_dim:
                    embeddings[vocab[word]] = vec
                    hit += 1
    print(f"[LSTM] GloVe 命中 {hit}/{len(vocab)} 个词")
    return embeddings