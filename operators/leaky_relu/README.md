# LeakyReLU 算子

## 概述

LeakyReLU (带泄露的线性整流单元) 是一种激活函数，在正区间保持线性，在负区间保持小的梯度。

**数学公式**: `y = x if x ≥ 0 else α·x`

其中 `α` 是负区间斜率，默认为 0.01。

## 目录结构

```
operators/leaky_relu/
├── spec.md              # 算子需求规范
├── leaky_relu_golden.py # PyTorch 参考实现
├── leaky_relu_impl.py   # PyPTO kernel 实现
├── test_leaky_relu.py   # 测试文件
└── README.md            # 本文档
```

## 使用方法

```python
from leaky_relu_impl import leaky_relu_wrapper
import torch

x = torch.randn(128, 1024, dtype=torch.float32, device="npu:0")
y = leaky_relu_wrapper(x, alpha=0.01)
```

## 运行测试

```bash
export TILE_FWK_DEVICE_ID=0
python test_leaky_relu.py
```

## 支持的输入

- **维度**: 1D, 2D, 3D, 4D
- **数据类型**: FP32, FP16 (自动转换为 FP32 计算)

## 已知限制

1. 3D+ 输入会被 reshape 为 2D 处理
2. 仅支持 NPU 设备

---
*生成时间: 2026-03-30T01:35:00Z*
