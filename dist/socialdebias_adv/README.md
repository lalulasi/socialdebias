# SocialDebias-Adv: 多源 LLM 改写攻击的虚假新闻对抗测试集

**版本**: v1.0
**发布日期**: 2026-05-06
**来源**: 论文「SocialDebias: 面向 LLM 改写攻击的虚假新闻鲁棒检测」

## 简介

SocialDebias-Adv 是基于 PolitiFact / GossipCop / Weibo21 测试集，使用 Qwen3.5-flash 与
DeepSeek-V3 双源 LLM 在 4 种语言风格（tone）下进行改写得到的对抗测试集，用于评估虚假
新闻检测模型在不同改写攻击下的鲁棒性。

与 SheepDog 原始的 publisher-based 对抗测试集（adv_A/B/C/D）形成正交评估维度：
- SheepDog: publisher 风格混淆（CNN/NYT/National Enquirer/The Sun）
- SocialDebias-Adv: tone 风格扰动（neutral/objective/sensational/emotionally_triggering）

## 数据集结构

```
data/
├── politifact_test_adv_<tone>_<source>.pkl   (8 个: 4 tone × 2 source)
├── gossipcop_test_adv_<tone>_<source>.pkl    (8 个: 4 tone × 2 source)
└── weibo21_test_adv_<tone>_<source>.pkl       (8 个: 4 tone × 2 source)
```

## 风格 (tone)

| Tone | 提示词（英文 / 中文） |
|---|---|
| neutral                | Rewrite in a neutral tone / 用中立的语气改写 |
| objective              | Rewrite in an objective and professional tone / 用客观专业的语气改写 |
| sensational            | Rewrite in a sensational tone / 用耸动的语气改写 |
| emotionally_triggering | Rewrite in an emotionally triggering tone / 用情绪煽动的语气改写 |

## 双源 (source)

| Source | 模型 |
|---|---|
| qwen     | Qwen3.5-flash (阿里云 DashScope API, 新加坡节点) |
| deepseek | DeepSeek-V3   (DeepSeek API)                 |

## 数据格式

每个 pkl 文件结构:
```python
{
    "pkl":    "原始测试集路径",
    "lang":   "en" | "zh",
    "tone":   "neutral" | "objective" | "sensational" | "emotionally_triggering",
    "source": "qwen" | "deepseek",
    "sample": 90 | 200,                    # 抽样数（PolitiFact 全量, GossipCop/Weibo21 200）
    "seed":   42,                          # 抽样随机种子
    "filter_thresholds": {                # 三道过滤阈值
        "entity_recall": 0.6,
        "semantic_sim":  0.65,
        "nli_exclude":   "contradiction"
    },
    "records": [
        {
            "orig_idx":      45,            # 原测试集索引
            "original":      "...",         # 原文
            "rewritten":     "...",         # 改写文
            "label":         0 | 1,          # 0=real, 1=fake
            "entity_recall": 0.85,
            "semantic_sim":  0.93,
            "p_entail":      0.92,           # NLI 蕴含概率
            "p_neutral":     0.06,
            "p_contradict":  0.02,
            "nli_label":     "entailment"
        },
        ...
    ]
}
```

## 数据规模

| 数据集 | 原测试集 | 抽样规模 | 改写总数 | 过滤后保留 | 平均保留率 |
|---|---|---|---|---|---|
| PolitiFact | 90    | 90 全量  | 720    | 489    | 67.9% |
| GossipCop  | 1584  | 200 抽样 | 1600   | 1089    | 68.1% |
| Weibo21    | 1923  | 200 抽样 | 1600   | 1080    | 67.5% |
| **总计**   |       |          | **3920** | **2658** | **67.8%** |

## 三道过滤管道

1. **实体召回率 ≥ 0.6**: 改写文中保留至少 60% 的原文实体
   - 英文: spaCy `en_core_web_sm` 提取实体
   - 中文: jieba 词性标注 (nr/ns/nt/m/t)

2. **语义相似度 ≥ 0.65**: BERT mean-pooling cosine 相似度
   - 英文: `bert-base-uncased`
   - 中文: `bert-base-chinese`

3. **NLI 排除矛盾**: 保留 entailment/neutral，排除 contradiction
   - 模型: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (跨语言)

## 使用示例

```python
import pickle
with open("data/politifact_test_adv_neutral_qwen.pkl", "rb") as f:
    d = pickle.load(f)
records = d["records"]
print(f"保留 {len(records)} 条")
for r in records[:3]:
    print(f"  label={r['label']} p_entail={r['p_entail']:.3f} text={r['rewritten'][:80]}")
```

## 引用

如使用 SocialDebias-Adv，请引用论文（详见 `DATA_CARD.md` 引用章节）。

## 许可

数据集本身遵循 MIT 协议。原始数据集（SheepDog / Weibo21）保持各自原协议。
