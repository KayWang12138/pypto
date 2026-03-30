# Sigmoid 算子实现

## 概述

Sigmoid 激活函数实现，将输入逐元素映射到 (0, 1) 区间。

### 数学公式

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### 数据流

```
    输入 x                    输出 y
┌──────────────────┐         ┌──────────────────┐
│  [m, n] 或       │         │  [m, n] 或       │
│  [b, m, n] 或    │ ──────▶ │  [b, m, n] 或    │
│  [b, s, m, n]    │ sigmoid │  [b, s, m, n]    │
│  float16/float32 │         │  float16/float32 │
│  /bfloat16       │         │  /bfloat16       │
└──────────────────┘         └──────────────────┘
```

## 目录结构

```
operators/sigmoid/
├── spec.md              # 需求规范
├── api_report.md        # API 探索报告
├── design.md            # 设计文档
├── sigmoid_golden.py    # Golden 参考实现
├── sigmoid_impl.py      # PyPTO kernel 实现
├── test_sigmoid.py      # 测试代码
└── README.md            # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 进入算子目录
cd operators/sigmoid
```

### 运行测试

```bash
# 运行所有测试用例
python test_sigmoid.py

# 运行特定测试用例
python test_sigmoid.py sigmoid::test_sigmoid_perf_p0
python test_sigmoid.py sigmoid::test_sigmoid_func_p0

# 查看可用测试用例
python test_sigmoid.py --list
```

### 测试用例

| 配置名称 | 类型 | 优先级 | 输入 Shape | 说明 |
|----------|------|--------|------------|------|
| test_sigmoid_perf_p0 | 性能 | P0 | [1024, 1024] | 核心性能场景 |
| test_sigmoid_func_p0 | 功能 | P0 | [32, 64] | 核心功能验证 |
| test_sigmoid_func_p1 | 功能 | P1 | [2, 128, 256] | 3维输入验证 |
| test_sigmoid_func_p2 | 功能 | P2 | [1, 1, 64, 64] | 4维输入验证 |

## 精度要求

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 3e-3 | 3e-3 |
| float16 | 3e-3 | 3e-3 |
| bfloat16 | 3e-3 | 3e-3 |

## API 映射

使用 PyPTO 直接 API：
```python
out[:] = pypto.sigmoid(x)
```

## Tiling 策略

- **算子类型**: Vector
- **Tiling 配置**:
  - 2D: `pypto.set_vec_tile_shapes(32, 128)`
  - 3D: `pypto.set_vec_tile_shapes(1, 32, 128)`
  - 4D: `pypto.set_vec_tile_shapes(1, 1, 32, 128)`

## 已知限制

1. **输入要求**:
   - 必须是连续 Tensor (`is_contiguous() == True`)
   - 不支持空 Tensor
   - Shape 仅支持 2-4 维

2. **数据类型**:
   - 支持 FP16/FP32/BF16
   - pypto.sigmoid 文档声明仅支持 FP32，但实际测试可能支持更多类型

3. **边界处理**:
   - 大正值 (x > 20): sigmoid(x) -> 1
   - 大负值 (x < -20): sigmoid(x) -> 0
   - 零值: sigmoid(0) = 0.5

## 验证入口

```python
from sigmoid_golden import sigmoid_golden
from sigmoid_impl import sigmoid_wrapper

# 生成测试数据
import torch
x = torch.randn(1024, 1024, dtype=torch.float32)

# 执行实现
result = sigmoid_wrapper(x)

# 执行 golden
golden = sigmoid_golden(x)

# 精度对比
import numpy as np
np.testing.assert_allclose(result.numpy(), golden.numpy(), rtol=3e-3, atol=3e-3)
```

---
*生成时间: 2026-03-29*
*生成工具: pypto-op-orchestrator*
