"""
环境验证脚本：确认 PyTorch 在 M5 上能用 MPS 加速
"""
import torch
import sys
import platform

print("=" * 60)
print("环境信息")
print("=" * 60)
print(f"Python 版本: {sys.version.split()[0]}")
print(f"系统: {platform.system()} {platform.machine()}")
print(f"PyTorch 版本: {torch.__version__}")

print("\n" + "=" * 60)
print("设备检测")
print("=" * 60)
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"MPS 可用: {torch.backends.mps.is_available()}")
print(f"MPS 已构建: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"\n✅ 使用 MPS 加速")

    # 简单测试：在 MPS 上做一次矩阵运算
    print("\n" + "=" * 60)
    print("MPS 计算测试")
    print("=" * 60)
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y
    print(f"矩阵乘法成功: {z.shape}, 设备: {z.device}")

    # 测试更复杂的操作（神经网络相关）
    import torch.nn as nn

    model = nn.Linear(768, 2).to(device)
    x = torch.randn(32, 768, device=device)
    y = model(x)
    print(f"Linear 层成功: {y.shape}, 设备: {y.device}")

    print("\n✅ 所有测试通过，环境就绪！")
elif torch.cuda.is_available():
    print(f"\n✅ 使用 CUDA: {torch.cuda.get_device_name(0)}")
else:
    print(f"\n⚠️ 仅 CPU 可用，会很慢")