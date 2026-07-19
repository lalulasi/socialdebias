# SocialDebias: 面向 LLM 改写攻击的社会化媒体虚假新闻鲁棒检测
## 环境搭建

### 本地开发（Mac M5 / Apple Silicon）
```bash
python3.11 -m venv socialvenv
source socialvenv/bin/activate
pip install --upgrade pip
pip install "setuptools<82" wheel
pip install -r requirements.txt
```

### 云端训练（AutoDL / CUDA）
```bash
# 镜像选 PyTorch 2.1.0 + Python 3.10/3.11 + CUDA 11.8
pip install -r requirements.txt
```

### 文件说明
- `requirements.txt`：核心依赖，跨平台兼容
- `requirements-lock.txt`：本地环境完整快照（仅供参考）

## 论文口径复现注意事项

- SocialDebias 新训练默认使用论文所述 `768→384→128` 双瓶颈；旧版
  `768→384` checkpoint 仍可由评估脚本自动识别，但不能与新结果混报。
- 8 维情绪特征必须来自官方 NRC-EmoLex。请把长表 TSV（
  `word<TAB>emotion<TAB>association`）放到
  `data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`，或设置
  `NRC_EMOLEX_PATH`。中文实验需指定对应的中文 NRC 翻译 TSV。
- 安装后还需执行 `python -m spacy download en_core_web_sm`。
- 已有的 BERT checkpoint 来自旧版“仅微调最后一层”实现时，必须重新训练；
  `run_bert_baseline.sh` 现在会按论文口径执行全参数微调，不再跳过旧产物。

代码与论文逐项核对结果见 `论文代码核对报告.md`。

修正后重新获取论文数据的正式顺序、命令、验收门禁和论文回填清单见
`新一轮论文实验执行步骤.md`。正式复现不要直接运行开发期的 `test_shs/`。

实验数据准备可先运行 `python scripts/audit_experiment_data.py` 查看缺项；取得 NRC
官方研究许可包并用 `scripts/prepare_nrc_emolex.py` 本地转换后，再执行
`bash prepare_paper_v2_data.sh` 重建三级过滤及 NLI `p_entail` 文件。数据准备脚本不会
自动调用付费生成 API。
