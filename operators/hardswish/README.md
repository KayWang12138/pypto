# Hardswish 算子

## 概述

Hardswish (硬 Swish) 是一种移动端友好的激活函数，是 Swish 的分段线性近似，常用于 MobileNetV3 等网络。

**数学公式**: `hardswish(x) = x * relu6(x + 3) / 6`

等价于:
- 当 `x >= 3` 时，`hardswish(x) = x`
- 当 `x <= -3` 时，`hardswish(x) = 0`
- 当 `-3 < x < 3` 时，`hardswish(x) = x * (x + 3) / 6`

## 目录结构

```
operators/hardswish/
├── spec.md                # 算子需求规范
├── hardswish_golden.py    # PyTorch 参考实现
├── hardswish_impl.py      # PyPTO kernel 实现
├── test_hardswish.py      # 测试文件
└── README.md              # 本文档
```

## 使用方法

```python
from hardswish_impl import hardswish_wrapper
import torch

x = torch.randn(128, 1024, dtype=torch.float32, device="npu:0")
y = hardswish_wrapper(x)
```

## 运行测试

```bash
export TILE_FWK_DEVICE_ID=0
python test_hardswish.py
```

## 支持的输入

- **维度**: 1D, 2D, 3D, 4D
- **数据类型**: FP32, FP16 (自动转换为 FP32 计算)

## 已知限制

1. 3D+ 输入会被 reshape 为 2D 处理
2. 仅支持 NPU 设备

---
*生成时间: 2026-03-30T02:35:00Z*
