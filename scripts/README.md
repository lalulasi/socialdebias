# scripts 目录说明

这里保留复现指南会直接调用，或被这些入口脚本依赖的实验代码。

## 数据准备

| 脚本 | 用途 |
|---|---|
| `gen_adversarial_local.py` | PolitiFact / GossipCop 本地 Qwen 改写 |
| `gen_adversarial_dashscope.py` | 英文改写的 DashScope API 版本 |
| `gen_deepseek_weibo21.py` | Weibo21 中文改写 |
| `filter_adversarial_v3.py` | 英文对抗训练数据过滤 |
| `filter_adversarial_v4_zh.py` | 中文对抗训练数据过滤 |
| `compute_nli_p_entail.py` | 计算 NLI `p_entail` |

## 训练

| 脚本 | 用途 |
|---|---|
| `train_bert_baseline.py` | BERT baseline 主实现 |
| `train_baseline.py` | 兼容旧命令的 BERT baseline 入口 |
| `train_lstm.py` | BiLSTM 基线 |
| `train_socialdebias_surface.py` | SocialDebias 主训练入口 |
| `train_socialdebias_contrastive.py` | 对比学习扫描 |
| `train_ablation.py` | 消融实验 |

`train_ablation.py` 的完整模型需显式传入
`--use_contrastive --use_soft_labels --lambda_fact_soft 0.5`；仅给出
`--orig_pkl/--adv_pkl` 不会隐式改变实验配置。`w/o InfoNCE` 保留软标签参数但
移除 `--use_contrastive`，`w/o AdvAug` 同时移除对抗数据路径和所有对抗损失，
`w/o LabelSmooth` 使用 `--label_smoothing 0 --lambda_fact_soft 0`。

## 评估

| 脚本 | 用途 |
|---|---|
| `evaluate_bert_adv.py` | BERT baseline 在 SheepDog adv_A/B/C/D 上评估 |
| `evaluate_surface_adv.py` | SocialDebias 主模型对抗评估 |
| `evaluate_ablation_adv.py` | 消融模型对抗评估 |
| `evaluate_contrastive_adv.py` | 对比学习模型对抗评估 |
| `evaluate_adversarial.py` | 共享的对抗评估函数 |
| `run_explanation_metrics.py` | 解释一致性评估 |
| `eval_llm_baseline.py` | DeepSeek 零样本评估 |
| `aggregate_llm_baseline.py` | LLM 评估结果汇总 |

## SocialDebias-Adv

| 脚本 | 用途 |
|---|---|
| `gen_socialdebias_adv.py` | 生成自建对抗测试集 |
| `filter_socialdebias_adv.py` | 过滤自建对抗测试集 |
| `evaluate_socialdebias_adv.py` | BERT / DeepSeek / SocialDebias 三方评估 |
| `package_socialdebias_adv.py` | 打包数据集与数据卡片 |

## 结果解析和人工评估

| 脚本 | 用途 |
|---|---|
| `parse_main_3seeds.py` | 主表 3 seed 汇总 |
| `parse_ablation_results.py` | 消融训练结果汇总 |
| `parse_ablation_adv.py` | 消融对抗结果汇总 |
| `parse_contrastive_results.py` | 对比学习结果汇总 |
| `parse_surface_results.py` | 表层特征结果汇总 |
| `parse_lstm_results.py` | LSTM 结果汇总 |
| `sample_human_eval.py` | 人工评估采样 |
| `analyze_human_eval.py` | 人工标注结果分析 |
| `run_human_eval_sample.sh` | 人工评估采样主控 |
| `run_human_eval_analysis.sh` | 人工评估分析主控 |
