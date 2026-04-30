"""
CHECKED 中文虚假新闻数据集加载器（含评论）
对应论文 5.5.X 节：评论语义编码作为社交特征的实现验证
"""
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter


def load_checked_news(json_path: Path, max_comments: int = 5) -> Dict:
    """加载单条 CHECKED 新闻 JSON。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 提取前 max_comments 条评论文本
    comments_raw = data.get("comments", [])
    comment_texts = []
    for c in comments_raw[:max_comments]:
        text = c.get("text", "").strip()
        if text and len(text) > 2:  # 过滤过短评论
            comment_texts.append(text)
    
    # 数值社交特征
    try:
        comment_num = int(data.get("comment_num", "0"))
        repost_num = int(data.get("repost_num", "0"))
        like_num = int(data.get("like_num", "0"))
    except (ValueError, TypeError):
        comment_num = repost_num = like_num = 0
    
    # 彻底清洗 CHECKED 数据中的来源域捷径
    raw_text = data["text"]
    text = raw_text
    # 1. 去【...】标题
    text = re.sub(r"^【[^】]*】", "", text)
    # 2. 去 #话题标签#
    text = re.sub(r"#[^#]+#", "", text)
    # 3. 去 emoji 和常见装饰字符
    text = re.sub(r"[𐀀-􏿿☀-➿]", "", text)
    # 4. 去 @用户名（连续非空白字符到下一个空白）
    text = re.sub(r"@\S+", "", text)
    # 5. 去末尾的 "L...的微博视频" / "L...的秒拍视频" 等 L 引用
    text = re.sub(r"\sL\S{2,30}", "", text)
    # 6. 去多余空白
    text = re.sub(r"\s+", " ", text).strip()
    # 7. 截断到固定长度 100 字符（缩小长度差异）
    text = text[:100]
    
    cleaned_text = text if text and len(text) > 10 else raw_text[:100]
    
    return {
        "id": data["id"],
        "text": cleaned_text,
        "label": 1 if data["label"] == "fake" else 0,  # fake=1, real=0
        "language": "zh",
        "comments": comment_texts,
        "n_comments": comment_num,
        "n_reposts": repost_num,
        "n_likes": like_num,
    }


def load_checked_dataset(
    data_dir: str = "data/checked/dataset",
    max_comments: int = 5,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    balance_topic: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    加载完整 CHECKED 数据集，按比例切分 train/val/test。
    自动平衡类别（先按类别切分再合并）。
    """
    data_path = Path(data_dir)
    fake_dir = data_path / "fake_news"
    real_dir = data_path / "real_news"
    
    print(f"=== CHECKED 数据集加载 ===")
    print(f"  fake_news/: {len(list(fake_dir.glob('*.json')))} 条")
    print(f"  real_news/: {len(list(real_dir.glob('*.json')))} 条")
    
    # 加载所有样本
    fake_samples = []
    for jf in fake_dir.glob("*.json"):
        try:
            fake_samples.append(load_checked_news(jf, max_comments))
        except Exception as e:
            print(f"  跳过 {jf.name}: {e}")
    
    real_samples = []
    for jf in real_dir.glob("*.json"):
        try:
            real_samples.append(load_checked_news(jf, max_comments))
        except Exception as e:
            print(f"  跳过 {jf.name}: {e}")
    
    print(f"  成功加载: fake={len(fake_samples)}, real={len(real_samples)}")
    
    # 主题平衡采样（修复 CHECKED 数据中 real 偏向通报体的问题）
    if balance_topic:
        REPORT_KEYWORDS = ["新增", "病例", "确诊", "例", "本土"]
        def is_report(text):
            return any(kw in text for kw in REPORT_KEYWORDS)
        
        random.seed(seed)
        
        # 按主题分桶
        fake_report = [s for s in fake_samples if is_report(s["text"])]
        fake_rumor = [s for s in fake_samples if not is_report(s["text"])]
        real_report = [s for s in real_samples if is_report(s["text"])]
        real_rumor = [s for s in real_samples if not is_report(s["text"])]
        
        # 各主题取 min(fake, real) 数量
        n_report = min(len(fake_report), len(real_report))
        n_rumor = min(len(fake_rumor), len(real_rumor))
        
        random.shuffle(fake_report); random.shuffle(real_report)
        random.shuffle(fake_rumor); random.shuffle(real_rumor)
        
        fake_samples = fake_report[:n_report] + fake_rumor[:n_rumor]
        real_samples = real_report[:n_report] + real_rumor[:n_rumor]
        
        print(f"  [平衡后] fake={len(fake_samples)}, real={len(real_samples)}")
        print(f"           (通报体 {n_report}+{n_report}, 言论体 {n_rumor}+{n_rumor})")
    
    # 评论统计
    n_with_comments_fake = sum(1 for s in fake_samples if len(s["comments"]) > 0)
    n_with_comments_real = sum(1 for s in real_samples if len(s["comments"]) > 0)
    print(f"  有评论的样本: fake={n_with_comments_fake}/{len(fake_samples)}, "
          f"real={n_with_comments_real}/{len(real_samples)}")
    
    # 按类别切分（保证类别均衡）
    random.seed(seed)
    random.shuffle(fake_samples)
    random.shuffle(real_samples)
    
    def split(samples, val_r, test_r):
        n = len(samples)
        n_test = int(n * test_r)
        n_val = int(n * val_r)
        n_train = n - n_test - n_val
        return samples[:n_train], samples[n_train:n_train+n_val], samples[n_train+n_val:]
    
    fake_tr, fake_va, fake_te = split(fake_samples, val_ratio, test_ratio)
    real_tr, real_va, real_te = split(real_samples, val_ratio, test_ratio)
    
    train = fake_tr + real_tr
    val = fake_va + real_va
    test = fake_te + real_te
    
    # 打乱合并后的训练集
    random.shuffle(train)
    
    print(f"  Train: {len(train)} (fake={len(fake_tr)}, real={len(real_tr)})")
    print(f"  Val: {len(val)} (fake={len(fake_va)}, real={len(real_va)})")
    print(f"  Test: {len(test)} (fake={len(fake_te)}, real={len(real_te)})")
    
    return train, val, test


if __name__ == "__main__":
    train, val, test = load_checked_dataset()
    
    print("\n=== 样本示例 ===")
    sample = train[0]
    print(f"  id: {sample['id']}")
    print(f"  label: {sample['label']} ({'fake' if sample['label']==1 else 'real'})")
    print(f"  text: {sample['text'][:100]}")
    print(f"  comments ({len(sample['comments'])} 条):")
    for i, c in enumerate(sample['comments'][:3]):
        print(f"    [{i}] {c[:80]}")
    print(f"  n_comments={sample['n_comments']}, n_reposts={sample['n_reposts']}, n_likes={sample['n_likes']}")