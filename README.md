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