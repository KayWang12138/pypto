# reshape 算子设计文档

> **算子名称**: reshape
> **算子分类**: tensor_manipulation / shape
> **生成时间**: 2026-03-29T07:40:00Z
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

reshape 算子将输入张量变换为指定的形状，保持数据不变，仅改变维度视图。支持动态轴（batch、seq_len）和负维度自动推断。

### 1.2 数学公式

$$y = \text{reshape}(x, \text{shape})$$

### 1.3 算法描述

reshape 是原子操作，无需分解为子步骤。核心逻辑是元数据变换，不涉及数据搬运或计算。

```
Algorithm: reshape
────────────────────────────────────
1. 验证元素总数匹配: prod(input.shape) == prod(target_shape)
2. 处理 -1 维度（如存在）: 自动推断该维度大小
3. 更新 Tensor 元数据（shape, stride）
4. 返回新视图（数据指针不变）
```

### 1.4 数据流图

```
        输入 x                  目标 shape            输出 y
    ┌──────────────┐       ┌──────────────┐      ┌──────────────┐
    │  [b, s1, d]  │       │   [s1,s2,s3]  │      │ [b, s2, s3]  │
    │   float32    │ ────▶ │    int32      │ ───▶ │   float32    │
    └──────────────┘       └──────────────┘      └──────────────┘

    动态轴: b, s1             (支持 -1)           动态轴: b, s2, s3

    约束: b x s1 x d = b x s2 x s3
```

---

## 2. API 映射设计

### 2.1 数学公式分解

reshape 是原子操作，无需分解：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `y = reshape(x, shape)` | 直接调用 PyPTO reshape API |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | `y = reshape(x, shape)` | `pypto.reshape` | input, shape, valid_shape, inplace | `docs/api/operation/pypto-reshape.md` |

### 2.3 计算步骤序列

```python
# 单步调用，无中间计算
output = pypto.reshape(input, target_shape, valid_shape=valid_shape, inplace=False)
```

### 2.4 设计依据

- **来源**: `api_report.md` §3 API 映射 + `docs/api/operation/pypto-reshape.md`
- **说明**: PyPTO 提供原生 `pypto.reshape` API，直接映射 PyTorch `torch.reshape`，支持 -1 维度自动推断和动态 shape

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
from dataclasses import dataclass
from pypto import Tensor, SymbolicScalar
from typing import List, Optional, Union

@dataclass
class ReshapeInput:
    x: Tensor                    # 输入张量, shape: [d1, d2, ..., dn], dtype: FP32/BF16/FP16
    target_shape: List[int]      # 目标形状, 支持一个维度为 -1
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None  # 动态场景有效 shape
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class ReshapeOutput:
    y: Tensor                    # 输出张量, shape: target_shape, dtype: 与输入相同
```

### 3.3 中间 Tensor 定义

reshape 是 view 操作，无中间 Tensor。

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| - | - | - | 无中间 Tensor |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| x, y | ND | reshape 不改变数据排列，保持原始 ND 格式 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| batch | 批次大小 | [1, INT32_MAX] |
| seq_len | 序列长度 | [1, INT32_MAX] |

动态轴处理方式：
```python
from pypto import SymbolicScalar

b = SymbolicScalar("batch")
s = SymbolicScalar("seq_len")
```

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def reshape_kernel(inputs: ReshapeInput) -> ReshapeOutput:
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: 无需 Tiling
- **判断依据**: reshape 是纯 shape 变换操作，不涉及向量计算（elementwise）或矩阵乘法（matmul），无需设置 TileShape

### 4.2 TileShape 初值设置

```python
# 不需要调用 set_vec_tile_shapes 或 set_cube_tile_shapes
# reshape 是元数据操作，无数据搬运或计算
```

### 4.3 设置依据

reshape 操作只改变 Tensor 的 shape 元数据，不涉及实际数据搬运或计算，因此不需要 Tiling 配置。

### 4.4 注意事项

- reshape 前需确保输入 Tensor 是 contiguous 的（通过 `from_torch` 约束）
- 非连续 Tensor 可能导致隐式拷贝

### 4.5 判断依据与适用条件

- **判断依据**: reshape 不涉及任何逐元素或矩阵计算，属于 shape 操作类别
- **适用条件**: 所有 shape 变换场景
- **不适用场景**: 无（reshape 本身不需要 tiling）

---

## 5. Loop 结构设计

### 场景 A：不需要 Loop

> reshape 操作本身不需要 Loop。但在实际使用中，如果需要对包含动态轴的 batch 进行迭代处理，由外层调用者负责 loop。

- **结论**: 不需要 pypto.loop（reshape 操作本身）
- **原因**: reshape 是单步元数据操作，编译期可确定所有信息，不需要分块迭代
- **适用条件**: 所有 reshape 调用
- **限制**: 无
- **处理方式**: 直接调用 `pypto.reshape`，无需循环包装

**说明**: 虽然输入 Tensor 可能有动态轴（batch、seq_len），但 reshape 只改变 shape 元数据，不需要遍历数据。如果外层需要对 batch 维度做循环，那是调用者的职责，不是 reshape 算子内部需要处理的。

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
import torch
from typing import List

def reshape_golden(x: torch.Tensor, shape: List[int]) -> torch.Tensor:
    """
    reshape 参考实现 (PyTorch)

    将输入张量变换为指定的形状。支持负维度自动推断。

    Args:
        x: 输入张量，支持任意维度和 dtype (float32/bfloat16/float16)
        shape: 目标形状，支持一个维度为 -1 表示自动推断

    Returns:
        输出张量，形状为目标 shape，dtype 与输入相同
    """
    return torch.reshape(x, shape)
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 基础_P0 | 功能 | P0 | shape=[2, 12] | [2, 3, 4] | [2, 12] | 基础 reshape |
| 动态轴_P0 | 功能 | P0 | shape=[b, s2, 8] | [b, s1, 64] | [b, s2, 8] | 动态 batch 和 seq |
| 负维度_P0 | 功能 | P0 | shape=[2, -1] | [2, 3, 4] | [2, 12] | 负维度自动推断 |
| 展开_P1 | 功能 | P1 | shape=[-1] | [2, 3, 4] | [24] | 展平为一维 |
| 增维_P1 | 功能 | P1 | shape=[2, 3, 4] | [2, 12] | [2, 3, 4] | 增加维度 |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 最小 batch | b=1, s=1 | 边界值测试 |
| 大 batch | b=128, s=1024 | 大规模测试 |
| 全 -1 推断 | shape=[-1] | flatten 场景 |
| NaN/Inf 保持 | 含特殊值输入 | 数值稳定性 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| bfloat16 | 0.01 | 0.01 |
| float16 | 0.01 | 0.01 |

**说明**: reshape 是 view 操作，不涉及数值计算，理论上输出应与输入完全一致（bitwise equal）。精度容差主要用于处理 dtype 转换场景。

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

reshape 是 view 操作，性能开销极小（仅元数据操作）。

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| - | - | - | - | - | - | ~0 (view 操作) |

### 7.2 开箱性能配置

```python
# reshape 不需要 TileShape 配置
# 无需调用 set_vec_tile_shapes 或 set_cube_tile_shapes
```

### 7.3 pass_options 配置

```python
# reshape 无特殊 pass_options 需求
pass_options = {}
```

### 7.4 runtime_options 配置

```python
runtime_options = {
    "run_mode": pypto.RunMode.NPU  # 0=NPU, 1=模拟器
}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- **输入必须 contiguous**: `from_torch` 要求输入 Tensor 必须是连续的（`is_contiguous() == True`）
- **元素总数必须匹配**: `prod(input.shape) == prod(target_shape)`
- **-1 维度最多一个**: shape 中最多一个维度为 -1
- **Shape Size 上限**: 不超过 INT32_MAX

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| 非连续 Tensor 错误 | 输入经过 transpose/permute 后未 contiguous | `from_torch` 会失败 | 调用前执行 `tensor.contiguous()` |
| inplace 约束违规 | inplace=True 但输出作为 Function 最终输出 | 运行时错误 | 仅在 loop 内使用 inplace=True |
| 元素总数不匹配 | reshape 前后元素数不等 | 运行时错误 | 调用前验证 shape 约束 |
| 动态 shape 推断失败 | 动态轴场景未提供 valid_shape | 编译期无法确定 shape | 使用 valid_shape 参数显式指定 |

### 8.3 特殊场景处理

- **非连续 Tensor**: 在调用 reshape 前需要调用者确保 contiguous
- **动态 shape**: 使用 `valid_shape` 参数显式指定运行时实际 shape
- **inplace 模式**: 仅在 loop 内使用，输出不能作为 Function 最终输出

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 优先使用非 inplace 模式 | 除非在 loop 内且有明确性能需求 |
| 确保 contiguous | 调用 reshape 前检查输入是否 contiguous |
| 使用 valid_shape | 动态 shape 场景显式指定，避免编译期推断失败 |
| 直接使用 -1 | PyPTO 自动处理负维度推断，无需手动计算 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/reshape/
├── spec.md                          # 需求规范
├── api_report.md                    # API 探索报告
├── design.md                        # 设计文档（本文件）
├── reshape_golden.py                # Golden 参考实现
├── reshape_impl.py                  # 算子实现代码
├── test_reshape.py                  # 测试代码
├── README.md                        # 实现说明
└── .orchestrator_state.json         # 状态文件
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 分析 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| reshape_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| reshape_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_reshape.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `reshape` |
| 目录名 | 与算子名称一致 | `operators/reshape/` |
| Golden 文件 | `{op}_golden.py` | `reshape_golden.py` |
| 实现文件 | `{op}_impl.py` | `reshape_impl.py` |
| 测试文件 | `test_{op}.py` | `test_reshape.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → reshape_golden.py → design.md → reshape_impl.py → test_reshape.py
```

---

*设计文档生成完成*
*质量检查: 通过*
