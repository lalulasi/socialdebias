"""
CHECKED PyTorch Dataset：把 CHECKED 样本转成模型输入
- 主文本 tokenize（用中文 BERT）
- 评论 tokenize（用中文 BERT，每条独立 + padding 到固定数量 K）
"""
import torch
from torch.utils.data import Dataset


class CheckedDataset(Dataset):
    """
    每条返回:
        input_ids:                 [L]      新闻正文 token
        attention_mask:            [L]
        label:                     scalar
        comment_input_ids:         [K, L_c] 评论 token（K 条评论，L_c 长度）
        comment_attention_mask:    [K, L_c]
        comment_mask:              [K]      1=有效评论，0=padding
    """
    def __init__(
        self,
        samples: list,
        tokenizer,
        max_length: int = 256,           # 微博正文较短
        max_comments: int = 5,
        max_comment_length: int = 64,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_comments = max_comments
        self.max_comment_length = max_comment_length

    def __len__(self):
        return len(self.samples)

    def _tokenize_one(self, text: str, max_length: int):
        enc = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 主文本
        input_ids, attention_mask = self._tokenize_one(s["text"], self.max_length)
        
        # 评论：padding 到 max_comments 条
        comments = s.get("comments", [])
        comment_ids_list = []
        comment_mask_list = []
        comment_valid_mask = []
        
        for i in range(self.max_comments):
            if i < len(comments):
                cid, cmask = self._tokenize_one(comments[i], self.max_comment_length)
                comment_ids_list.append(cid)
                comment_mask_list.append(cmask)
                comment_valid_mask.append(1)
            else:
                # 没有这条评论，用全 0 占位
                comment_ids_list.append(torch.zeros(self.max_comment_length, dtype=torch.long))
                comment_mask_list.append(torch.zeros(self.max_comment_length, dtype=torch.long))
                comment_valid_mask.append(0)
        
        comment_input_ids = torch.stack(comment_ids_list)       # [K, L_c]
        comment_attention_mask = torch.stack(comment_mask_list)  # [K, L_c]
        comment_mask = torch.tensor(comment_valid_mask, dtype=torch.float32)  # [K]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(s["label"], dtype=torch.long),
            "comment_input_ids": comment_input_ids,
            "comment_attention_mask": comment_attention_mask,
            "comment_mask": comment_mask,
        }


if __name__ == "__main__":
    # smoke test
    import sys
    sys.path.insert(0, ".")
    from transformers import AutoTokenizer
    from utils.checked_dataloader import load_checked_dataset

    train, val, test = load_checked_dataset()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    ds = CheckedDataset(train, tokenizer, max_length=256, max_comments=5, max_comment_length=64)
    
    print(f"\n样本数: {len(ds)}")
    sample = ds[0]
    print(f"input_ids: {sample['input_ids'].shape}")
    print(f"comment_input_ids: {sample['comment_input_ids'].shape}")
    print(f"comment_mask: {sample['comment_mask']}")
    print(f"label: {sample['label']}")