# 实验记录：云端 GossipCop 完整复现

## 实验身份
- 实验 ID: exp002_cloud_gossipcop
- 日期: 2026-04-18
- 硬件: AutoDL RTX 4090 (24GB)
- 数据集: GossipCop
- 数据划分: 5383/949/1584
...

## 训练过程

### 基线 BERT
| Epoch | Train Loss | Val Loss | Val Acc | Val F1 | Val AUC | 耗时 |
| 1     | 0.4925    | 0.4247   | 0.7956  | 0.7942 | 0.8858  | 103.2s |
| 2     | 0.3526    | 0.4826   | 0.7597  | 0.7581 | 0.8821  | 102.0s |
| 3     | 0.2281    | 0.4639   | 0.8051  | 0.8050 | 0.8925  | 103.0s |

最佳 checkpoint: Epoch 3（val F1=0.8050）

Test: Acc=0.7538 | F1=0.7536 | AUC=0.8409