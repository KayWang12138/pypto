# Add 算子

## 概述

Add 是最基本的算术运算，对两个张量进行逐元素加法，支持广播。

**数学公式**: `y = x1 + x2`

## 目录结构

```
operators/add/
├── add_golden.py    # PyTorch 参考实现
├── add_impl.py      # PyPTO kernel 实现
├── test_add.py      # 测试文件
└── README.md        # 本文档
```

## 使用方法

```python
from add_impl import add_wrapper
import torch

x1 = torch.randn(128, 1024, dtype=torch.float32, device="npu:0")
x2 = torch.randn(128, 1024, dtype=torch.float32, device="npu:0")
y = add_wrapper(x1, x2)
```

## 运行测试

```bash
export TILE_FWK_DEVICE_ID=0
python test_add.py
```

## 支持的输入

- **维度**: 1D, 2D, 3D, 4D
- **数据类型**: FP32, FP16
- **广播**: 支持广播

## 动态轴支持

使用隐式 shape 推断: `pypto.Tensor([], pypto.DT_FP32)`

---
*生成时间: 2026-03-30T04:10:00Z*
