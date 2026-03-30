# abs 算子设计方案

## 1. 概述

- **算子名称**: abs
- **功能**: 逐元素计算输入张量的绝对值
- **数学公式**: y = |x|
- **分类**: element-wise (Vector 类型)

## 2. API 映射设计

### 2.1 公式分解

| 步骤 | 数学表达 | PyPTO API | 说明 |
|------|----------|-----------|------|
| 1 | y = \|x\| | `pypto.abs(x)` | 直接映射 |

### 2.2 计算步骤

1. 输入 tensor x
2. 调用 `pypto.abs(x)` 计算绝对值
3. 写回输出 tensor y

## 3. 数据规格设计

### 3.1 Input/Output

```python
@dataclass
class AbsInput:
    x: pypto.Tensor  # [*, D], float32, 支持动态轴

@dataclass
class AbsOutput:
    y: pypto.Tensor  # [*, D], float32, 与输入 shape 相同
```

### 3.2 中间 Tensor

无中间 tensor，直接计算。

### 3.3 数据格式

- **Dtype**: float32 (主要), float16 (可选)
- **Layout**: contiguous

## 4. Tiling 策略

### 4.1 算子类型

- **类型**: Vector (逐元素运算)
- **Tiling API**: `pypto.set_vec_tile_shapes()`

### 4.2 TileShape 配置

根据输入维度选择 tiling:

| 维度 | TileShape 配置 | 说明 |
|------|----------------|------|
| 1D | (tile_size,) | 单维度 |
| 2D | (64, 128) | 标准配置 |
| 3D | (64, 64, 128) | 标准配置 |
| 4D | (4, 16, 64, 128) | 标准配置 |

### 4.3 设置依据

- abs 是逐元素运算，不涉及跨维度计算
- TileShape 维度与输出 tensor 一致
- 使用隐式 shape 推断，无需显式 DYNAMIC 标记

## 5. Loop 结构设计

### 5.1 是否需要 Loop

**不需要 Loop**

- abs 是逐元素运算，PyPTO 内部自动并行处理
- 无需手动循环

### 5.2 场景 A: 无 Loop 实现

```python
@pypto.frontend.jit
def abs_kernel(x, out):
    pypto.set_vec_tile_shapes(64, 128)
    out[:] = pypto.abs(x)
```

## 6. 验证方案

### 6.1 Golden 函数

```python
def abs_golden(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)
```

### 6.2 测试用例

| 配置名称 | 类型 | 优先级 | 输入 Shape | 说明 |
|----------|------|--------|------------|------|
| 功能_P0 | 功能 | P0 | [128, 1024] | 基础功能验证 |
| 动态轴_P0 | 功能 | P0 | [64, 1024] | 动态轴验证 |
| 性能_P0 | 性能 | P0 | [4096, 4096] | 性能测试 |
| 4D_shape | 功能 | P1 | [2, 4, 8, 16] | 多维度验证 |
| FP16 | 功能 | P1 | [128, 1024] | FP16 dtype |

### 6.3 精度标准

- **atol**: 0.001
- **rtol**: 0.001
- **验证方式**: `np.testing.assert_allclose()`

## 7. 性能指标与开箱配置

### 7.1 性能目标

- 首跑精度成功性能的 2 倍

### 7.2 pass_options

无特殊配置。

### 7.3 runtime_options

```python
runtime_options={"run_mode": "npu"}
```

## 8. 风险点与注意事项

### 8.1 已知约束

1. **Shape 维度限制**: API 仅支持 2-4 维
2. **contiguous 要求**: 输入 tensor 必须连续
3. **动态轴声明**: 需在 `from_torch` 时声明

### 8.2 常见错误规避

1. 不要使用 `return pypto.abs(x)`，必须用 `out[:] = pypto.abs(x)`
2. 确保输入 tensor 已调用 `.contiguous()`
3. TileShape 维度必须与 tensor 维度一致

### 8.3 特殊场景处理

- 零值: 正常计算，|0| = 0
- 正负无穷: 正常计算，|±Inf| = +Inf
- NaN: 正常计算，|NaN| = NaN

## 9. 交付件清单

### 9.1 目录结构

```
operators/abs/
├── spec.md           # 需求规格
├── api_report.md     # API 探索报告
├── abs_golden.py     # Golden 参考实现
├── design.md         # 设计方案
├── abs_impl.py       # PyPTO 实现
├── test_abs.py       # 测试脚本
└── README.md         # 使用说明
```

### 9.2 生成顺序

1. spec.md ✓
2. api_report.md ✓
3. abs_golden.py ✓
4. design.md ✓ (本文档)
5. abs_impl.py
6. test_abs.py
7. README.md

---
*生成时间: 2026-03-30T04:20:00Z*
