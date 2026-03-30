# mean_reduction 算子设计文档

> **算子名称**: mean_reduction
> **算子分类**: reduction
> **生成时间**: 2026-03-28T22:08:00Z
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

沿指定轴计算张量均值的归约操作，类似于 PyTorch 的 `torch.mean()` 或 `tensor.mean()`。支持动态轴（batch、seq 等维度）和 keepdim 参数。

### 1.2 数学公式

$$y = \text{mean}(x, \text{dim}) = \frac{\sum_{i} x_i}{N}$$

其中 N 为 dim 轴上的元素数量。

### 1.3 算法描述

<!-- 简单算子，公式足以描述计算逻辑，无需算法描述 -->

### 1.4 数据流图

```
    输入 x                         输出 y
┌──────────────────┐         ┌──────────────────┐
│  [b, s, n, d]    │         │  [b, n, d]       │  (dim=1, keepdim=False)
│   float32        │ ──────▶ │   float32        │
└──────────────────┘         └──────────────────┘
                                   或
                             ┌──────────────────┐
                             │  [b, 1, n, d]    │  (dim=1, keepdim=True)
                             │   float32        │
                             └──────────────────┘

公式: y = mean(x, dim) = sum(x_i) / N  (沿指定轴dim归约)
动态轴: b (batch), s (seq)
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | sum_result = sum(x, dim, keepdim) | 沿 dim 轴求和 |
| 2 | y = sum_result / N | 除以归约轴元素数量得到均值 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | sum(x, dim, keepdim) | `pypto.sum(input, dim, keepdim)` | input: x, dim: int, keepdim: bool | `docs/api/operation/pypto-sum.md` |
| 2 | sum_result / N | `pypto.div(input, other)` | input: sum_result, other: float | `docs/api/operation/pypto-div.md` |

**说明**: PyPTO 无直接的 `mean` API，使用 `pypto.sum` + `pypto.div` 的 substitute 方案实现。

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
# 步骤 1: 沿 dim 轴求和
sum_result = pypto.sum(x, dim=dim, keepdim=keepdim)

# 步骤 2: 获取归约轴元素数量 N（支持动态轴）
N = x.shape[dim]

# 步骤 3: 除以 N 得到均值
y = pypto.div(sum_result, N)
```

### 2.4 设计依据

- 来源：api_report.md + docs/api/operation/pypto-sum.md + docs/api/operation/pypto-div.md
- 说明：PyPTO 无直接 mean API，但 sum 和 div 均为直接支持的 API，组合使用可完整实现 mean 功能。x.shape[dim] 在动态轴场景下返回 SymbolicScalar，可直接用于 div 操作。

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
from dataclasses import dataclass
from pypto import Tensor

@dataclass
class MeanReductionInput:
    x: Tensor      # 输入张量, shape: [b, s, n, d] 或任意 shape, dtype: DT_FP32
    dim: int       # 归约轴索引
    keepdim: bool  # 是否保持归约后的维度, 默认 False
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class MeanReductionOutput:
    y: Tensor      # 归约后的均值张量, shape: 去掉 dim 维度的 shape 或保持维度 (keepdim=True), dtype: DT_FP32
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| sum_result | 去掉 dim 维度的 shape (keepdim=False) 或保持维度 (keepdim=True) | DT_FP32 | 沿 dim 轴求和的结果 |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| x | ND | 默认格式，from_torch 自动推导 |
| sum_result | ND | 与输入格式一致 |
| y | ND | 与输入格式一致 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| b | batch 维度，表示批次大小 | [1, INT32_MAX] |
| s | sequence 维度，表示序列长度 | [1, INT32_MAX] |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def mean_reduction_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    dim: int,
    keepdim: bool,
    y: pypto.Tensor([], pypto.DT_FP32),
) -> None:
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: 仅涉及 reduction (sum) 和 element-wise (div) 操作，无 matmul

### 4.2 TileShape 初值设置

```python
# 对于 3D 输入 [b, s, d]，TileShape 设置为 [8, 8, 8]
# 尾轴 8 满足 float32 的 32B 对齐要求（8 * 4 = 32 bytes）
pypto.set_vec_tile_shapes(8, 8, 8)
```

### 4.3 设置依据

1. **尾轴 32B 对齐**: float32 需要 8 元素对齐（8 * 4 = 32 bytes），设置 tile_d = 8 满足要求
2. **次尾轴限制**: sum API 要求次尾轴 ≤ 255，设置 tile_s = 8 满足要求
3. **TileShape 总大小**: 8 * 8 * 8 * 4 = 2KB << 64KB，满足 TileShape ≤ 64KB 约束
4. **与输入维度一致**: TileShape 维度数与输入一致（3D）

### 4.4 注意事项

- 尾轴需 32 bytes 对齐，float32 时尾轴需为 8 的倍数
- 次尾轴 ≤ 255，大 shape 需注意
- keepdim=False 后需重设 TileShape 再调用其他 operation

### 4.5 判断依据与适用条件

- 判断依据：本算子为 Vector 类型，仅使用 `set_vec_tile_shapes`
- 适用条件：2-4 维输入，尾轴可对齐到 32B，次尾轴 ≤ 255
- 不适用场景：输入超过 4 维时需先 reshape；尾轴无法对齐时需 padding

---

## 5. Loop 结构设计

### 5.1 Loop 判断结论

- **结论**: 不需要 pypto.loop
- **原因**: 虽然存在动态轴（b, s），但 PyPTO 的 sum 和 div API 内部会自动处理动态轴的数据切分和迭代，编译器会生成相应的运行时代码
- **适用条件**: 单轴归约，使用 PyPTO 内置 sum/div API
- **限制**: 多轴归约需要显式循环或多次调用

### 5.2 静态轴 vs 动态轴处理

| 轴 | 类型 | 处理方式 |
|----|------|----------|
| b (dim=0 时) | 动态 | PyPTO sum API 内部自动处理 |
| s (dim=1 时) | 动态 | PyPTO sum API 内部自动处理 |
| n, d | 静态 | TileShape 直接覆盖 |

### 5.3 数据依赖处理

无复杂数据依赖。sum → div 为顺序执行，无跨迭代依赖。

### 5.4 尾块处理策略

由 PyPTO 编译器自动处理，无需手动处理尾块。

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def mean_reduction_golden(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """mean_reduction 参考实现"""
    return torch.mean(x, dim=dim, keepdim=keepdim)
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | dim=1, keepdim=False | [b, s, 4096] | [b, 4096] | 核心性能场景，沿 seq 维度归约 |
| 功能_P0 | 功能 | P0 | dim=1, keepdim=True | [b, s, 4096] | [b, 1, 4096] | keepdim 功能验证 |
| 功能_P1 | 功能 | P1 | dim=0, keepdim=False | [b, s, 4096] | [s, 4096] | 沿 batch 维度归约 |
| 动态_P0 | 功能 | P0 | dim=1, keepdim=False | [b, s, d] (动态) | [b, d] | 动态 shape 测试 |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 零值输入 | x = zeros | 验证 mean of zeros = 0 |
| 常量输入 | x = full(5.0) | 验证 mean of constant = constant |
| 小 shape | [2, 4, 8] | 边界小 shape 测试 |
| 大 shape | [64, 1024, 4096] | 大 shape 性能测试 |
| 负索引 dim | dim=-1 | 负索引支持测试 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.01 | 0.01 |
| bfloat16 | 0.01 | 0.01 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 性能_P0 | 性能 | P0 | dim=1, keepdim=False | [2, 512, 4096] | [2, 4096] | 首跑精度成功即可 |

### 7.2 开箱性能配置

```python
# Tiling 配置（基于 3D 输入 [b, s, d]）
# 尾轴 8 满足 float32 32B 对齐，次尾轴 8 < 255
pypto.set_vec_tile_shapes(8, 8, 8)
```

### 7.3 pass_options 配置

无需特殊 pass_options 配置。

### 7.4 runtime_options 配置

```python
runtime_options = {
    "run_mode": pypto.RunMode.NPU,
}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- pypto.sum 要求输入 shape 为 2-4 维，Shape Size ≤ INT32_MAX
- pypto.sum 要求 TileShape ≤ 64KB，尾轴 32 bytes 对齐，次尾轴 ≤ 255
- pypto.div 的 other 参数不支持 nan、inf
- 输入张量必须 contiguous（is_contiguous() == True）

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| TileShape 未设置 | 调用 sum 前未调用 set_vec_tile_shapes | 编译失败 | 确保在 sum 调用前设置 TileShape |
| 尾轴未对齐 | tile_d 不是 8 的倍数（float32） | 编译失败或性能劣化 | 确保 tile_d 为 8 的倍数 |
| 次尾轴超限 | tile_s > 255 | 编译失败 | 确保 tile_s ≤ 255 |
| 输入不连续 | x.is_contiguous() == False | 运行时错误 | 使用 x.contiguous() 确保连续 |
| 动态轴未标记 | from_torch 未指定 dynamic_axis | 静态 shape，无法处理动态输入 | 在 from_torch 中指定 dynamic_axis |

### 8.3 特殊场景处理

- **动态轴**: 通过 `pypto.from_torch(x, dynamic_axis=[0, 1])` 标记 b, s 为动态轴
- **keepdim=False 后续操作**: 需要重新设置 TileShape 再调用其他 operation
- **多轴归约**: 当前设计不支持，如需支持需多次调用或扩展设计

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 优先验证静态 shape | 先用固定 shape 验证功能正确性，再测试动态 shape |
| 封装 wrapper 函数 | 提供 mean_reduction_wrapper 函数封装 from_torch、kernel 调用、输出转换 |
| 复用 golden 验证 | 直接使用 mean_reduction_golden.py 中的验证逻辑 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/mean_reduction/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── mean_reduction_golden.py         # Golden 参考实现（已有）
├── mean_reduction_impl.py           # 算子实现代码
├── test_mean_reduction.py           # 测试代码
└── .orchestrator_state.json         # 状态文件
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 探索 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| mean_reduction_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| mean_reduction_impl.py | 代码 | 算子核心实现 | pypto-op-develop |
| test_mean_reduction.py | 代码 | 测试用例 | pypto-op-develop |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `mean_reduction` |
| 目录名 | 与算子名称一致 | `operators/mean_reduction/` |
| Golden 文件 | `{op}_golden.py` | `mean_reduction_golden.py` |
| 实现文件 | `{op}_impl.py` | `mean_reduction_impl.py` |
| 测试文件 | `test_{op}.py` | `test_mean_reduction.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → mean_reduction_golden.py → mean_reduction_impl.py → test_mean_reduction.py
```
