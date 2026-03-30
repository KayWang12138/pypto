# GroupNorm 算子

## 概述

GroupNorm (组归一化) 是一种归一化方法，将通道分成若干组，每组独立归一化。常用于小批量训练和目标检测等场景。

**数学公式**: `y = γ * (x - μ) / sqrt(σ² + ε) + β`

其中 μ 和 σ² 沿 (C//G, H, W) 维度计算，G 是分组数。

## 目录结构

```
operators/groupnorm/
├── spec.md              # 算子需求规范
├── groupnorm_golden.py  # PyTorch 参考实现
├── groupnorm_impl.py    # PyPTO kernel 实现
├── test_groupnorm.py    # 测试文件
└── README.md            # 本文档
```

## 使用方法

```python
from groupnorm_impl import groupnorm_wrapper
import torch

x = torch.randn(2, 64, 14, 14, dtype=torch.float32, device="npu:0")
weight = torch.randn(64, dtype=torch.float32, device="npu:0")
bias = torch.randn(64, dtype=torch.float32, device="npu:0")
y = groupnorm_wrapper(x, num_groups=32, weight=weight, bias=bias)
```

## 运行测试

```bash
export TILE_FWK_DEVICE_ID=0
python test_groupnorm.py
```

## 支持的输入

- **维度**: 4D [N, C, H, W]
- **数据类型**: FP32, FP16
- **分组数**: C % num_groups == 0

## 已知限制

1. 仅支持 4D 输入
2. 仅支持 NPU 设备

---
*生成时间: 2026-03-30T03:15:00Z*
