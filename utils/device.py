"""
设备管理工具：自动选择最优设备，支持 Mac MPS / CUDA / CPU 三端兼容

设计原则：
1. 自动检测环境，无需手动指定
2. 提供 fallback 机制，遇到 MPS 不支持的算子时降级到 CPU
3. 提供环境变量开关，方便强制指定设备做调试
"""
import os
import torch
import warnings


def get_device(prefer: str = "auto") -> torch.device:
    """
    获取最优计算设备。

    Args:
        prefer: 'auto' / 'cuda' / 'mps' / 'cpu'
                'auto' 时按 CUDA > MPS > CPU 顺序选择
                也可以通过环境变量 FORCE_DEVICE 强制指定

    Returns:
        torch.device 对象
    """
    # 环境变量优先级最高（方便云端部署时强制指定）
    env_device = os.environ.get("FORCE_DEVICE", "").lower()
    if env_device in ("cuda", "mps", "cpu"):
        prefer = env_device

    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif prefer == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("请求 CUDA 但不可用，回退到 CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    elif prefer == "mps":
        if not torch.backends.mps.is_available():
            warnings.warn("请求 MPS 但不可用，回退到 CPU")
            return torch.device("cpu")
        return torch.device("mps")
    else:
        return torch.device("cpu")


def device_info() -> dict:
    """返回当前可用设备的详细信息（用于日志和调试）"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def safe_to_device(tensor_or_module, device: torch.device, fallback_to_cpu: bool = True):
    """
    安全地把 tensor 或 module 移到目标设备。
    遇到 MPS 不支持的情况时，可选择降级到 CPU。

    这个函数主要是为了应对 M5 上某些算子尚未实现的情况。
    """
    try:
        return tensor_or_module.to(device)
    except (RuntimeError, NotImplementedError) as e:
        if fallback_to_cpu and device.type == "mps":
            warnings.warn(f"MPS 上失败，降级到 CPU: {e}")
            return tensor_or_module.to("cpu")
        raise


def get_recommended_batch_size(device: torch.device) -> int:
    """
    根据设备返回推荐的 batch_size。
    Mac 本地用小 batch 快速调试，云端用大 batch 充分利用 GPU。
    """
    if device.type == "cuda":
        return 32  # 24GB 3090 通常能撑住
    elif device.type == "mps":
        return 8  # Mac 本地保守一点，主要是测试代码逻辑
    else:
        return 4  # CPU 模式只为了跑通流程


if __name__ == "__main__":
    # 直接运行此文件可以查看设备信息
    device = get_device()
    print(f"推荐设备: {device}")
    print(f"推荐 batch_size: {get_recommended_batch_size(device)}")
    print(f"详细信息: {device_info()}")