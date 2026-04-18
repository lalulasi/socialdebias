# 实验记录：[实验简短描述]

## 实验身份
- **实验 ID**：expXXX_[数据集]_[方法]_[环境]
- **日期**：2026-04-18
- **开始时间** / **结束时间**：
- **硬件**：AutoDL RTX 4090 (24GB) / Mac M5
- **数据集**：PolitiFact / GossipCop / LUN
- **数据划分**：[train]/[val]/[test] 条
- **Git commit**：`git rev-parse HEAD` 的输出
- **耗时总计**：

## 超参数
- model_name: bert-base-uncased
- max_length: 512
- batch_size: 4
- num_epochs: 3
- learning_rate: 2e-5
- weight_decay: 0.01
- label_smoothing: 0.1
- seed: 42

### 仅 SocialDebias
- lambda_fact: 1.0
- lambda_bias: 0.5
- lambda_consist: 0.3
- use_frozen_bert: True

## 训练过程

### 基线 BERT
| Epoch | Train Loss | Val Loss | Val Acc | Val F1  | Val AUC | 耗时 |
|-------|-----------|----------|---------|---------|---------|------|
| 1     | 0.4546    | 0.2919   | 0.9074  | 0.9071  | 0.9503  |      |
| 2     | 0.1832    | 0.3617   | 0.8704  | 0.8682  | 0.9393  |      |
| 3     | 0.0572    | 0.4380   | 0.8519  | 0.8485  | 0.9490  |      |

最佳 checkpoint: **Epoch 1**（val F1=0.9071）

最佳 checkpoint: Epoch 3

### SocialDebias
| Epoch | L_fact | L_bias | L_consist | Val Acc | Val F1 | Val AUC | 偏置分支 Acc | 耗时 |
|-------|--------|--------|-----------|---------|--------|---------|-------------|------|
| 1     | 0.5540 | 0.8406 | 0.2102    | 0.8148  | 0.8107 | 0.8966  | 0.1667      |      |
| 2     | 0.3857 | 1.4958 | 0.6024    | 0.7963  | 0.7905 | 0.9131  | 0.2037      |      |
| 3     | 0.3609 | 1.1867 | 0.7769    | 0.8333  | 0.8333 | 0.9269  | 0.1852      |      |

最佳 checkpoint: Epoch 3（val F1=0.8333）
Val Loss 未记录（训练脚本暂未输出，已加入 TODO）

最佳 checkpoint: Epoch 3

## 最终结果

### 测试集表现
**F1 指标**
| 方法 | Clean F1 | 对抗A F1 | 对抗B F1 | 对抗C F1 | 对抗D F1 | 平均F1降幅 |
|------|----------|----------|----------|----------|----------|-----------|
| BERT | 0.8776   | 0.7205   | 0.7098   | 0.7777   | 0.7666   | 0.1340    |
| SD   | 0.8774   | 0.8438   | 0.7857   | 0.8438   | 0.7857   | 0.0627    |

**AUC 指标**
| 方法 | Clean AUC | 对抗A AUC | 对抗B AUC | 对抗C AUC | 对抗D AUC | 平均AUC降幅 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| BERT | 0.9417    | 0.8054    | 0.7654    | 0.8598    | 0.8262    | 0.1275    |
| SD   | 0.9654    | 0.9274    | 0.9136    | 0.9200    | 0.8953    | 0.0514    |

**ASR 指标**
| 方法 | 对抗A ASR | 对抗B ASR | 对抗C ASR | 对抗D ASR | 平均 ASR |
|------|-----------|-----------|-----------|-----------|---------|
| BERT | 0.1899    | 0.2025    | 0.1266    | 0.1392    | 0.1646  |
| SD   | 0.1013    | 0.1646    | 0.0886    | 0.1519    | 0.1266  |
  
### 鲁棒性改进
- **F1 下降减少**：+7.13pp（基线 13.40 → SD 6.27）
- **AUC 下降减少**：+7.62pp（基线 12.75 → SD 5.14）
- **ASR 降低**：+3.80pp（基线 16.46% → SD 12.66%）

## 诊断与观察
- 训练是否稳定：是（基线和 SD 的训练曲线都平滑下降）
- 是否过拟合：**基线出现轻微过拟合**（Val Loss 从 0.29 上升到 0.44，val F1 逐步下降，因此 checkpoint 来自 Epoch 1）
- 有无异常现象：基线的最佳 checkpoint 出现在 Epoch 1，表明 3 个 epoch 可能过多，可考虑减少到 2 个 epoch 或加早停机制
- 对抗 D 上 ASR 比基线高？：**是**（SD 0.1519 vs 基线 0.1392，一个值得在论文里讨论的局限点）

## 运行命令（可复现）
```bash
python scripts/train_baseline.py --mode dev_real --dataset politifact --language en
python scripts/train_socialdebias.py --mode dev_real --dataset politifact --language en \
    --lambda_fact 1.0 --lambda_bias 0.5 --lambda_consist 0.3
python scripts/evaluate_adversarial.py --dataset politifact --language en
```

## 结论
- 这次实验主要说明了什么：
- 是否值得写进论文：
- 下次实验需要改进的地方：

## 关联文件
- 模型 checkpoint：`./results/models/baseline_politifact_en.pt`、`./results/models/socialdebias_politifact_en.pt`
- 输出截图：`./experiments/exp001_cloud_politifact.png`
