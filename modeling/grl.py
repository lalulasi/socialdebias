"""
梯度反转层（Gradient Reversal Layer）

核心思想：前向传播时等同于恒等映射，反向传播时将梯度乘以 -λ。
这样上游模块会收到"反向"梯度，从而学到对下游任务"无用"的表示。

来源：Ganin & Lempitsky (2015), Unsupervised Domain Adaptation by Backpropagation
"""
import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    自定义 autograd Function，实现梯度反转的核心逻辑。
    继承 torch.autograd.Function 需要实现 forward 和 backward 两个静态方法。
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        # 前向传播：恒等映射，但把 lambda_ 保存下来供反向传播使用
        ctx.lambda_ = lambda_
        return x.view_as(x)  # 返回 x 的一个视图，保持计算图连接

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：梯度取反并乘以 lambda_
        # 注意这里要返回和 forward 输入数量相同的梯度（x 和 lambda_ 各一个）
        # lambda_ 不需要梯度，所以返回 None
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(torch.nn.Module):
    """
    梯度反转层的 Module 封装，方便嵌入到 nn.Sequential 或 nn.Module 中。

    Args:
        lambda_: 反转系数，控制对抗强度。通常训练初期取较小值（如 0.1），
                逐步增大到 1.0。
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """动态调整反转系数（用于课程学习式训练）"""
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

    print("\n✅ GRL 测试完成")