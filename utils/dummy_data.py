"""
假数据生成器：支持中英文。

设计原则：
1. 中英文样本结构完全一致，只是词汇不同
2. 真假新闻的特征要明显，便于验证模型能学出来
"""
import random
import json
import os
from typing import List, Dict
from typing import Optional

# ==================== 英文词汇库 ====================
FAKE_KEYWORDS_EN = [
    "shocking", "unbelievable", "doctors hate", "secret", "exposed",
    "miracle", "amazing trick", "you won't believe", "banned",
    "they don't want you to know", "explosive", "scandal"
]

REAL_KEYWORDS_EN = [
    "according to", "researchers found", "study published", "data shows",
    "officials confirmed", "report indicates", "analysis suggests",
    "evidence shows", "experts say", "investigation revealed"
]

NEUTRAL_TOPICS_EN = [
    "the economy", "climate change", "education policy", "healthcare reform",
    "technology development", "international relations", "scientific research",
    "urban planning", "public transportation", "agricultural production"
]

# ==================== 中文词汇库 ====================
FAKE_KEYWORDS_ZH = [
    "震惊", "不敢相信", "医生都恨", "内幕曝光", "秘密揭露",
    "神奇方法", "你绝对想不到", "已被封杀", "他们不想让你知道",
    "重磅", "丑闻"
]

REAL_KEYWORDS_ZH = [
    "据报道", "研究人员发现", "研究指出", "数据显示",
    "官方确认", "报告表明", "分析显示",
    "证据表明", "专家称", "调查显示"
]

NEUTRAL_TOPICS_ZH = [
    "经济形势", "气候变化", "教育政策", "医疗改革",
    "科技发展", "国际关系", "科学研究",
    "城市规划", "公共交通", "农业生产"
]


def generate_dummy_sample(label: int, language: str = "en") -> Dict:
    """
    生成一条假数据。

    Args:
        label: 0=真新闻, 1=假新闻
        language: 'en' 或 'zh'
    """
    if language == "en":
        topic = random.choice(NEUTRAL_TOPICS_EN)
        if label == 1:
            keywords = random.sample(FAKE_KEYWORDS_EN, k=2)
            text = (
                f"BREAKING: {keywords[0]} news about {topic}! "
                f"This {keywords[1]} discovery will change everything you thought you knew. "
                f"Click to find out the truth they don't want you to see."
            )
        else:
            keywords = random.sample(REAL_KEYWORDS_EN, k=2)
            text = (
                f"{keywords[0].capitalize()} a recent report on {topic}, "
                f"new findings have emerged. {keywords[1].capitalize()} the situation has "
                f"developed in measurable ways over the past quarter."
            )
        comments = [f"This is comment {i} about {topic}." for i in range(random.randint(2, 5))]

    elif language == "zh":
        topic = random.choice(NEUTRAL_TOPICS_ZH)
        if label == 1:
            keywords = random.sample(FAKE_KEYWORDS_ZH, k=2)
            text = (
                f"【{keywords[0]}】关于{topic}的{keywords[1]}！"
                f"这个发现将彻底改变你的认知，"
                f"点击查看他们不想让你知道的真相。"
            )
        else:
            keywords = random.sample(REAL_KEYWORDS_ZH, k=2)
            text = (
                f"{keywords[0]}近期发布的关于{topic}的报告，"
                f"出现了新的研究发现。{keywords[1]}过去一个季度以来，"
                f"相关情况已发生可量化的变化。"
            )
        comments = [f"这是关于{topic}的第{i}条评论。" for i in range(random.randint(2, 5))]

    else:
        raise ValueError(f"不支持的语言: {language}")

    sample = {
        "id": f"dummy_{language}_{random.randint(10000, 99999)}",
        "text": text,
        "label": label,
        "language": language,
        "comments": comments,
        "user_features": {
            "follower_count": random.randint(10, 100000),
            "friend_count": random.randint(10, 5000),
            "verified": random.choice([0, 1]),
            "account_age_days": random.randint(30, 3650),
        },
        "propagation_stats": {
            "retweet_count": random.randint(0, 1000),
            "reply_count": random.randint(0, 500),
            "spread_velocity": random.random() * 10,
        }
    }
    return sample


def generate_dummy_dataset(
        n_samples: int = 100,
        language: str = "en",
        save_path: Optional[str] = None,
        seed: int = 42
) -> List[Dict]:
    """生成完整的假数据集"""
    random.seed(seed)

    if save_path is None:
        save_path = f"./data/dummy/dummy_data_{language}.json"

    samples = []
    for i in range(n_samples):
        label = i % 2
        sample = generate_dummy_sample(label, language=language)
        samples.append(sample)

    random.shuffle(samples)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"✅ 已生成 {n_samples} 条 {language} 假数据 → {save_path}")
    print(f"   真新闻数: {sum(1 for s in samples if s['label'] == 0)}")
    print(f"   假新闻数: {sum(1 for s in samples if s['label'] == 1)}")

    return samples


def load_dummy_dataset(path: Optional[str] = None, language: str = "en") -> List[Dict]:
    """从文件加载假数据"""
    if path is None:
        path = f"./data/dummy/dummy_data_{language}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # 同时生成中英文两套假数据
    print("=" * 60)
    print("生成英文假数据")
    print("=" * 60)
    en_samples = generate_dummy_dataset(n_samples=100, language="en")

    print("\n" + "=" * 60)
    print("生成中文假数据")
    print("=" * 60)
    zh_samples = generate_dummy_dataset(n_samples=100, language="zh")

    # 看一眼英文样本
    print("\n" + "=" * 60)
    print("英文样本示例")
    print("=" * 60)
    for s in en_samples[:2]:
        print(f"[label={s['label']}] {s['text']}")

    # 看一眼中文样本
    print("\n" + "=" * 60)
    print("中文样本示例")
    print("=" * 60)
    for s in zh_samples[:2]:
        print(f"[label={s['label']}] {s['text']}")