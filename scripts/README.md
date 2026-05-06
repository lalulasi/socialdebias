# scripts/ 脚本说明

> 论文「SocialDebias: 面向 LLM 改写攻击的虚假新闻鲁棒检测」实验脚本目录
> 最终保留 30 个脚本，按功能分组如下。废弃版本归档在 `scripts/legacy/`。

## 训练入口（7 个）

| 脚本 | 用途 | 论文对应章节 |
|---|---|---|
| `train_baseline.py` | BERT 基线（PolitiFact / GossipCop / Weibo21） | 5.2 / 5.3 |
| `train_lstm.py` | LSTM 基线 | 5.2 |
| `train_checked.py` | CHECKED 中文数据集探索（已放弃，保留记录） | Day 11 总结 |
| `train_socialdebias.py` | SocialDebias 基础双分支版本 | 5.2 |
| `train_socialdebias_contrastive.py` | SD + InfoNCE 对比学习 | 5.5.3 |
| **`train_socialdebias_surface.py`** | **SD + 表层特征 + 软标签（最终版）** | **5.5.4 / 5.5.5** |
| `train_ablation.py` | 6 变体消融实验 | 5.5.1 |

> 论文最终主表（Table 5.1）使用 `train_socialdebias_surface.py`。

## 对抗数据生成（4 个）

| 脚本 | 用途 | 输出 |
|---|---|---|
| `gen_adversarial_local.py` | Qwen 本地批量改写（PolitiFact / GossipCop） | `data/qwen_adv/{dataset}_train_adv.pkl` |
| `gen_deepseek_50.py` | DeepSeek 50 条对照（多 LLM 异构验证） | `data/qwen_adv/politifact_train_adv_deepseek_50.pkl` |
| `gen_deepseek_weibo21.py` | Weibo21 中文 DeepSeek 改写 | `data/qwen_adv/weibo21_train_adv_deepseek.pkl` |
| `gen_socialdebias_adv.py` | SocialDebias-Adv 测试集生成（Qwen+DeepSeek 双源） | `data/socialdebias_adv/*.pkl` |

## 数据过滤（2 个）

| 脚本 | 适用 | 备注 |
|---|---|---|
| `filter_adversarial_v3.py` | 英文（PolitiFact / GossipCop） | 实体一致性 + NLI + 语义相似度 |
| `filter_adversarial_v4_zh.py` | 中文（Weibo21） | 在 v3 基础上适配中文分词与实体识别 |

> v1 / v2 旧版本归档在 `legacy/`。

## 评估（5 个）

| 脚本 | 用途 |
|---|---|
| `evaluate_adversarial.py` | 通用对抗评估（BERT / SD 在 4 变体上） |
| `evaluate_ablation_adv.py` | 消融实验对抗评估 |
| `evaluate_contrastive_adv.py` | 对比学习消融对抗评估 |
| `evaluate_surface_adv.py` | 表层特征消融对抗评估 |
| `eval_llm_baseline.py` | DeepSeek 零样本基线评估 |

## 工具与辅助（4 个）

| 脚本 | 用途 |
|---|---|
| `compute_nli_p_entail.py` | 计算 NLI 蕴含概率（生成 `*_p_entail.pkl`） |
| `analyze_explanation.py` | 解释一致性分析（IG + Top-K + Spearman） |
| `aggregate_llm_baseline.py` | LLM 基线结果汇总 CSV |
| `sample_train_data.py` | GossipCop 平衡采样 |

## 结果解析（6 个）

| 脚本 | 用途 |
|---|---|
| `parse_main_3seeds.py` | 主表 3 seed 平均（Table 5.1） |
| `parse_ablation_results.py` | 消融实验结果汇总 |
| `parse_ablation_adv.py` | 消融对抗结果汇总 |
| `parse_contrastive_results.py` | 对比学习扫描结果 |
| `parse_surface_results.py` | 表层特征扫描结果 |
| `parse_lstm_results.py` | LSTM 基线结果 |

## 测试 / 调试（1 个）

| 脚本 | 用途 |
|---|---|
| `test_qwen.py` | Qwen API 连通性测试（开发期使用） |

## 归档（legacy/）

`filter_adversarial.py` 与 `filter_adversarial_v2.py` 是过滤管道早期版本，最终未在论文实验中采用。保留以备追溯。

---

## 完整复现路径

参见仓库根目录 `复现指南_v1.1.md`。