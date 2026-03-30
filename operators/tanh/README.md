# TanH 算子实现

## 概述

本目录包含 PyPTO 框架下的 TanH（双曲正切）激活函数算子实现。

### 数学公式

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

### 特性

- **输出范围**: [-1, 1]
- **应用场景**: RNN/LSTM 门控机制、注意力机制、生成器输出层
- **支持数据类型**: float16, float32, bfloat16
- **支持维度**: 2-4D

## 文件结构

```
operators/tanh/
├── spec.md              # 算子需求规范
├── api_report.md        # API 探索报告
├── design.md            # 设计文档
├── tanh_golden.py       # Golden 参考实现
├── tanh_impl.py         # PyPTO kernel 实现
├── test_tanh.py         # 测试入口
├── README.md            # 本文件
└── .orchestrator_state.json  # 状态文件
```

## 实现说明

### API 映射

由于 PyPTO 没有直接的 `tanh` API，本实现使用组合方式：

| 步骤 | 操作 | PyPTO API |
|------|------|-----------|
| 1 | e^x | `pypto.exp(x)` |
| 2 | -x | `pypto.mul(x, -1.0)` |
| 3 | e^(-x) | `pypto.exp(neg_x)` |
| 4 | e^x - e^(-x) | `pypto.sub(exp_x, exp_neg_x)` |
| 5 | e^x + e^(-x) | `pypto.add(exp_x, exp_neg_x)` |
| 6 | 分子/分母 | `pypto.div(numerator, denominator)` |

### Tiling 策略

- **算子类型**: Vector
- **TileShape 配置**:
  - 2D: `[32, 128]`
  - 3D: `[1, 32, 128]`
  - 4D: `[1, 1, 32, 128]`

## 使用方法

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 运行测试

```bash
cd operators/tanh

# 运行所有测试
python test_tanh.py

# 运行单个测试
python test_tanh.py tanh::test_tanh_perf_p0
python test_tanh.py tanh::test_tanh_func_p0
python test_tanh.py tanh::test_tanh_func_p1
python test_tanh.py tanh::test_tanh_func_p2

# 查看可用测试
python test_tanh.py --list
```

### 代码调用

```python
import torch
from tanh_impl import tanh_wrapper

# 创建输入
x = torch.randn(1024, 1024, dtype=torch.float16, device="npu:0")

# 调用算子
y = tanh_wrapper(x)

# 输出范围 [-1, 1]
print(f"Output range: [{y.min()}, {y.max()}]")
```

## 精度要求

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 3e-3 | 3e-3 |
| float16 | 3e-3 | 3e-3 |
| bfloat16 | 3e-3 | 3e-3 |

## 测试用例

| 配置名称 | 类型 | Shape | 说明 |
|----------|------|-------|------|
| 性能_P0 | 性能 | [1024, 1024] | 核心性能场景 |
| 功能_P0 | 功能 | [32, 64] | 核心功能验证 |
| 功能_P1 | 功能 | [2, 128, 256] | 3维输入验证 |
| 功能_P2 | 功能 | [1, 1, 64, 64] | 4维输入验证 |

## 约束

- 输入 tensor 必须连续（`is_contiguous() == True`）
- 不支持空 tensor（shape size >= 1）
- Shape 仅支持 2-4 维
- Shape Size 不超过 INT32_MAX
- 不支持 nan/inf 输入

## 参考

- PyTorch `torch.tanh`
- PyPTO API 文档: `docs/api/operation/pypto-exp.md`, `pypto-mul.md`, `pypto-sub.md`, `pypto-add.md`, `pypto-div.md`
