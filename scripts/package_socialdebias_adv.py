"""
SocialDebias-Adv 数据集打包（H.4）

生成开源数据集压缩包：
  socialdebias_adv/
    ├── README.md
    ├── DATA_CARD.md
    ├── filter_report.csv
    ├── eval_report.csv (如有)
    └── data/
        └── *.pkl (24 个过滤后的对抗测试集)

最终: socialdebias_adv_v1.0.tar.gz
"""
import argparse
import csv
import json
import pickle
import shutil
import tarfile
from collections import defaultdict
from datetime import date
from pathlib import Path


README_TEMPLATE = """# SocialDebias-Adv: 多源 LLM 改写攻击的虚假新闻对抗测试集

**版本**: v1.0
**发布日期**: {release_date}
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
{{
    "pkl":    "原始测试集路径",
    "lang":   "en" | "zh",
    "tone":   "neutral" | "objective" | "sensational" | "emotionally_triggering",
    "source": "qwen" | "deepseek",
    "sample": 90 | 200,                    # 抽样数（PolitiFact 全量, GossipCop/Weibo21 200）
    "seed":   42,                          # 抽样随机种子
    "filter_thresholds": {{                # 三道过滤阈值
        "entity_recall": 0.6,
        "semantic_sim":  0.65,
        "nli_exclude":   "contradiction"
    }},
    "records": [
        {{
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
        }},
        ...
    ]
}}
```

## 数据规模

| 数据集 | 原测试集 | 抽样规模 | 改写总数 | 过滤后保留 | 平均保留率 |
|---|---|---|---|---|---|
| PolitiFact | 90    | 90 全量  | 720    | {pf_kept}    | {pf_rate}% |
| GossipCop  | 1584  | 200 抽样 | 1600   | {gc_kept}    | {gc_rate}% |
| Weibo21    | 1923  | 200 抽样 | 1600   | {wb_kept}    | {wb_rate}% |
| **总计**   |       |          | **3920** | **{total_kept}** | **{total_rate}%** |

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
print(f"保留 {{len(records)}} 条")
for r in records[:3]:
    print(f"  label={{r['label']}} p_entail={{r['p_entail']:.3f}} text={{r['rewritten'][:80]}}")
```

## 引用

如使用 SocialDebias-Adv，请引用论文（详见 `DATA_CARD.md` 引用章节）。

## 许可

数据集本身遵循 MIT 协议。原始数据集（SheepDog / Weibo21）保持各自原协议。
"""


DATA_CARD_TEMPLATE = """# SocialDebias-Adv 数据卡片

## 1. 基本信息

| 字段 | 值 |
|---|---|
| 数据集名称 | SocialDebias-Adv |
| 版本 | v1.0 |
| 发布日期 | {release_date} |
| 协议 | MIT |
| 体量 | {total_kept} 条 (3 dataset × 4 tone × 2 source = 24 个 pkl) |
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
@misc{{socialdebias_adv_2026,
  title  = {{SocialDebias-Adv: A Multi-Source LLM Rewriting Adversarial Test Set for Fake News Detection}},
  author = {{作者信息待补充}},
  year   = {{2026}},
  note   = {{Companion dataset to SocialDebias paper.}},
}}
```

## 7. 复现

完整生成流程见仓库 `复现指南_v1.x.md` H.1-H.3 阶段：

```bash
# H.1 生成（需 DASHSCOPE_API_KEY + DEEPSEEK_API_KEY）
bash run_socialdebias_adv.sh

# H.2 过滤
python scripts/filter_socialdebias_adv.py \\
    --input_dir data/socialdebias_adv \\
    --output_dir data/socialdebias_adv/filtered

# H.3 三方评估（可选）
python scripts/evaluate_socialdebias_adv.py \\
    --filtered_dir data/socialdebias_adv/filtered \\
    --output results/socialdebias_adv_eval.csv
```

## 8. 联系

数据集相关问题：请联系论文作者。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered_dir", default="data/socialdebias_adv/filtered")
    parser.add_argument("--filter_report", default="data/socialdebias_adv/filtered/filter_report.csv")
    parser.add_argument("--eval_report", default="results/socialdebias_adv_eval.csv",
                        help="可选，存在则一并打包")
    parser.add_argument("--output", default="socialdebias_adv_v1.0.tar.gz")
    parser.add_argument("--staging_dir", default="dist/socialdebias_adv")
    args = parser.parse_args()

    src_dir = Path(args.filtered_dir)
    if not src_dir.exists():
        raise RuntimeError(f"过滤目录不存在: {src_dir}")

    pkl_files = sorted(src_dir.glob("*.pkl"))
    if len(pkl_files) != 24:
        print(f"[WARN] 期望 24 个 pkl，实际 {len(pkl_files)}")

    # 统计保留率（用于填 README）
    stats = defaultdict(lambda: {"kept": 0})
    total_kept = 0
    expected_in = {"politifact": 720, "gossipcop": 1600, "weibo21": 1600}
    for pkl in pkl_files:
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        n = len(d["records"])
        total_kept += n
        ds = "politifact" if "politifact" in pkl.name else \
             "gossipcop" if "gossipcop" in pkl.name else "weibo21"
        stats[ds]["kept"] += n

    fmt_args = {
        "release_date": date.today().isoformat(),
        "pf_kept": stats["politifact"]["kept"],
        "gc_kept": stats["gossipcop"]["kept"],
        "wb_kept": stats["weibo21"]["kept"],
        "total_kept": total_kept,
        "pf_rate": round(stats["politifact"]["kept"] / expected_in["politifact"] * 100, 1),
        "gc_rate": round(stats["gossipcop"]["kept"] / expected_in["gossipcop"] * 100, 1),
        "wb_rate": round(stats["weibo21"]["kept"] / expected_in["weibo21"] * 100, 1),
        "total_rate": round(total_kept / 3920 * 100, 1),
    }

    # 准备 staging
    staging = Path(args.staging_dir)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / "data").mkdir()

    # 复制 pkl
    print(f"[1/4] 复制 {len(pkl_files)} 个 pkl 到 {staging / 'data'}")
    for pkl in pkl_files:
        shutil.copy2(pkl, staging / "data" / pkl.name)

    # 复制 filter_report
    if Path(args.filter_report).exists():
        shutil.copy2(args.filter_report, staging / "filter_report.csv")
        print(f"[2/4] 复制 filter_report.csv")

    # 复制 eval_report (可选)
    if Path(args.eval_report).exists():
        shutil.copy2(args.eval_report, staging / "eval_report.csv")
        print(f"[3/4] 复制 eval_report.csv")
    else:
        print(f"[3/4] 跳过 eval_report (未找到 {args.eval_report})")

    # 写 README + DATA_CARD
    (staging / "README.md").write_text(README_TEMPLATE.format(**fmt_args), encoding="utf-8")
    (staging / "DATA_CARD.md").write_text(DATA_CARD_TEMPLATE.format(**fmt_args), encoding="utf-8")
    print(f"[4/4] 写 README.md + DATA_CARD.md")

    # 打包
    out_tar = Path(args.output)
    print(f"\n打包到 {out_tar}")
    with tarfile.open(out_tar, "w:gz") as tar:
        tar.add(staging, arcname="socialdebias_adv")

    size_mb = out_tar.stat().st_size / (1024 * 1024)
    print(f"\n========== 完成 ==========")
    print(f"压缩包: {out_tar} ({size_mb:.1f} MB)")
    print(f"  ├── data/    {len(pkl_files)} 个 pkl")
    print(f"  ├── filter_report.csv")
    print(f"  ├── eval_report.csv (如有)")
    print(f"  ├── README.md")
    print(f"  └── DATA_CARD.md")
    print(f"\n保留率：")
    print(f"  PolitiFact: {fmt_args['pf_kept']:>4}/{expected_in['politifact']} ({fmt_args['pf_rate']}%)")
    print(f"  GossipCop:  {fmt_args['gc_kept']:>4}/{expected_in['gossipcop']} ({fmt_args['gc_rate']}%)")
    print(f"  Weibo21:    {fmt_args['wb_kept']:>4}/{expected_in['weibo21']} ({fmt_args['wb_rate']}%)")
    print(f"  TOTAL:      {fmt_args['total_kept']:>4}/3920 ({fmt_args['total_rate']}%)")


if __name__ == "__main__":
    main()
