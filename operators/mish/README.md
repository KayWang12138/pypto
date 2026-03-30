# Mish 算子

## 概述

Mish 是一种平滑的非单调激活函数，公式为 `mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`。

相比 ReLU， Mish 提供了更平滑的梯度流，在深度网络中表现更好。

## 公式

$$mish(x) = x \cdot \tanh(\ln(1 + e^x))$$

## 实现

- **文件**: `mish_impl.py`
- **核心函数**: `mish_kernel`, `mish_wrapper`
- **数据类型**: float32
- **支持维度**: 1D-4D

## 特殊处理

- 4D 输入自动 reshape 为 2D 避免编译问题
- 1D 输入自动 reshape 为 2D 满足 API 约束

## 使用方法

```python
import torch
from mish_impl import mish_wrapper

# 创建输入
x = torch.randn(128, 1024, dtype=torch.float32)

# 调用算子
y = mish_wrapper(x)

# y 的 shape 与 x 相同
assert y.shape == x.shape
```

## 测试
运行测试文件:
```bash
python test_mish.py
```

## 磾度要求
- **atol**: 0.001
- **rtol**: 0.001

## 参考
- PyTorch: `torch.nn.functional.mish`
- 论文: Mish: A Self Regularized Non-Monotonic Activation Function (Diganta Misra, 2019)
