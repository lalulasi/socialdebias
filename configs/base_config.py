"""
全局配置：本地用 dev 配置，云端用 prod 配置。
新增：支持真实数据集切换。
"""
from dataclasses import dataclass, field
from typing import Optional

MODEL_MAP = {
    "en": "bert-base-uncased",
    "zh": "bert-base-chinese",
}


@dataclass
class BaseConfig:
    """基础配置"""
    # 语言与模型
    language: str = "en"
    model_name: str = ""

    # 数据源配置（新增）
    use_dummy_data: bool = True  # True=假数据, False=真实数据
    dataset_name: str = "politifact"  # 真实数据集名（politifact/gossipcop/lun）

    # 序列长度（真实数据改成 512）
    max_length: int = 512
    num_classes: int = 2

    # 训练相关
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 8

    # 路径
    data_dir: str = "./data"
    output_dir: str = "./results"
    log_dir: str = "./logs"

    # 随机种子
    seed: int = 42

    # 最大样本数（调试用）
    max_train_samples: Optional[int] = None
    log_every_n_steps: int = 10

    def __post_init__(self):
        if not self.model_name:
            self.model_name = MODEL_MAP[self.language]


@dataclass
class DevDummyConfig(BaseConfig):
    """本地开发（假数据）配置"""
    use_dummy_data: bool = True
    max_length: int = 256  # 假数据短
    batch_size: int = 4
    num_epochs: int = 1
    max_train_samples: Optional[int] = 100
    log_every_n_steps: int = 5


@dataclass
class DevRealConfig(BaseConfig):
    """本地开发（真实数据）配置"""
    use_dummy_data: bool = False
    max_length: int = 512  # 真实新闻长
    batch_size: int = 4  # Mac 上 512 长度 + batch 4 是安全的
    num_epochs: int = 3  # 真实数据至少 3 个 epoch
    max_train_samples: Optional[int] = None  # 用全部（PolitiFact 才 306 条）
    log_every_n_steps: int = 10


@dataclass
class ProdConfig(BaseConfig):
    """云端训练配置"""
    use_dummy_data: bool = False
    max_length: int = 512
    batch_size: int = 16
    num_epochs: int = 5
    max_train_samples: Optional[int] = None
    log_every_n_steps: int = 50


def get_config(mode: str = "dev_real", language: str = "en", dataset: str = "politifact") -> BaseConfig:
    """
    根据模式返回配置。

    mode 可选:
        'dev_dummy' - 本地假数据（用于代码调试）
        'dev_real'  - 本地真实数据（日常开发用）
        'prod'      - 云端完整训练
    """
    if mode == "dev_dummy":
        cfg = DevDummyConfig(language=language)
    elif mode == "dev_real":
        cfg = DevRealConfig(language=language, dataset_name=dataset)
    elif mode == "prod":
        cfg = ProdConfig(language=language, dataset_name=dataset)
    else:
        raise ValueError(f"未知模式: {mode}")
    return cfg