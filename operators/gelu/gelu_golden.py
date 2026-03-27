"""
GELU 算子 Golden 参考实现

提供与 PyTorch 对比的基准实现。
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def gelu_golden(x: torch.Tensor) -> torch.Tensor:
    """GELU 激活函数 golden 实现

    使用 PyTorch 内置的 tanh 近似版本，与 PyPTO 保持一致。

    Args:
        x: 输入张量，任意 shape

    Returns:
        输出张量，shape 与输入相同
    """
    return F.gelu(x, approximate='tanh')


def generate_test_cases() -> Tuple[torch.Tensor, torch.Tensor]:
    """生成测试用例

    Returns:
        (input_tensor, expected_output)
    """
    torch.manual_seed(42)

    # 测试用例 1: 小规模 [1, 128, 768] - BERT-base
    x1 = torch.randn(1, 128, 768, dtype=torch.float32)
    y1 = gelu_golden(x1)

    return x1, y1


if __name__ == "__main__":
    x, y = generate_test_cases()
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    print(f"Output shape: {y.shape}, dtype: {y.dtype}")
    print(f"Input sample: {x[0, 0, :5]}")
    print(f"Output sample: {y[0, 0, :5]}")
