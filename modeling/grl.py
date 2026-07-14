"""梯度反转层：前向保持输入不变，反向将梯度乘以 -lambda。"""
import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """梯度反转的 autograd 实现。"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(torch.nn.Module):
    """梯度反转层的模块封装。"""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """更新梯度反转系数。"""
        self.lambda_ = lambda_


if __name__ == "__main__":
    # 验证 GRL 的正确性：前向不变，反向取反
    import torch.nn as nn

    # 构造一个最小测试：一个全连接层 + GRL + 另一个全连接层
    torch.manual_seed(42)

    x = torch.randn(4, 10, requires_grad=True)
    linear1 = nn.Linear(10, 10)
    grl = GradientReversalLayer(lambda_=1.0)
    linear2 = nn.Linear(10, 1)

    # 前向：x -> linear1 -> GRL -> linear2
    h = linear1(x)
    h_grl = grl(h)
    y = linear2(h_grl)

    # 反向
    loss = y.sum()
    loss.backward()

    print("=" * 60)
    print("GRL 正确性测试")
    print("=" * 60)
    print(f"前向：h 和 h_grl 应该相等: {torch.allclose(h, h_grl)}")
    print(f"  h[0, :3]    = {h[0, :3].detach().numpy()}")
    print(f"  h_grl[0, :3]= {h_grl[0, :3].detach().numpy()}")

    print(f"\nlinear1 权重梯度的符号（应该和直接计算的符号相反）:")
    print(f"  grad sign: {linear1.weight.grad[0, :3].sign().tolist()}")

    # 对比：没有 GRL 时的梯度符号
    linear1_ref = nn.Linear(10, 10)
    linear1_ref.load_state_dict(linear1.state_dict())
    linear1_ref.weight.grad = None
    linear2_ref = nn.Linear(10, 1)
    linear2_ref.load_state_dict(linear2.state_dict())

    h_ref = linear1_ref(x)
    y_ref = linear2_ref(h_ref)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    print(f"  无 GRL 时 grad sign: {linear1_ref.weight.grad[0, :3].sign().tolist()}")
    print(f"  （应该和上面相反）")

    print("\nGRL 测试完成")
