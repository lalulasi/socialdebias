# SocialDebias-Adv 数据卡片

## 1. 基本信息

| 字段 | 值 |
|---|---|
| 数据集名称 | SocialDebias-Adv |
| 版本 | v1.0 |
| 发布日期 | 2026-05-06 |
| 协议 | MIT |
| 体量 | 2658 条 (3 dataset × 4 tone × 2 source = 24 个 pkl) |
| 语言 | 英文 (PolitiFact/GossipCop) + 简体中文 (Weibo21) |

## 2. 数据来源

- **PolitiFact** test split (90 条) - SheepDog (Wu et al., KDD 2024)
- **GossipCop** test split (200 抽样) - SheepDog (Wu et al., KDD 2024)
- **Weibo21** test split (200 抽样) - MDFEND (Nan et al., CIKM 2021)

抽样种子固定为 42，可复现。

## 3. 改写流程

1. **改写生成**: Qwen3.5-flash + DeepSeek-V3 双源，4 风格
2. **三道质量过滤**: 实体召回 ≥ 0.6, 语义相似度 ≥ 0.65, NLI 排除矛盾
3. **元信息保留**: 每条样本附 entity_recall/semantic_sim/p_entail/p_neutral/p_contradict 五维分数

## 4. 适用场景

✓ 评估虚假新闻分类器对 LLM 改写攻击的鲁棒性
✓ 多源 LLM 攻击异质性研究
✓ 跨语言（中英）改写攻击对比
✓ tone 维度的攻击难度阶梯研究

✗ 训练分类器（数据规模偏小，仅作测试用）
✗ 文本生成质量评估（已通过 NLI 过滤）

## 5. 已知偏置与局限

1. **改写来源偏置**: 仅使用 Qwen3.5-flash + DeepSeek-V3，未覆盖 GPT-4 / Claude 等
2. **采样偏置**: GossipCop/Weibo21 仅抽样 200 条，可能不完全反映原测试集分布
3. **NER 工具差异**: 中文 jieba.posseg 与英文 spaCy 实体边界不完全一致
4. **过滤阈值固定**: 实体 0.6 / 语义 0.65 是经验值，不同任务可能需调整

## 6. 引用

```bibtex
@misc{socialdebias_adv_2026,
  title  = {SocialDebias-Adv: A Multi-Source LLM Rewriting Adversarial Test Set for Fake News Detection},
  author = {TODO: 作者名},
  year   = {2026},
  note   = {Companion dataset to SocialDebias paper.},
}
```

## 7. 复现

完整生成流程见仓库 `复现指南_v1.x.md` H.1-H.3 阶段：

```bash
# H.1 生成（需 DASHSCOPE_API_KEY + DEEPSEEK_API_KEY）
bash run_socialdebias_adv.sh

# H.2 过滤
python scripts/filter_socialdebias_adv.py \
    --input_dir data/socialdebias_adv \
    --output_dir data/socialdebias_adv/filtered

# H.3 三方评估（可选）
python scripts/evaluate_socialdebias_adv.py \
    --filtered_dir data/socialdebias_adv/filtered \
    --output results/socialdebias_adv_eval.csv
```

## 8. 联系

数据集相关问题: TODO 联系方式
