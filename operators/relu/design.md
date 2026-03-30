# ReLU 算子设计文档

> **算子名称**: relu
> **算子分类**: element-wise
> **生成时间**: 2026-03-28
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

ReLU（Rectified Linear Unit）是深度学习中常用的激活函数。对输入 tensor 的每个元素执行整流线性单元运算，只保留正数部分，负数变为 0。

### 1.2 数学公式

$$res_i = \max(0, input_i)$$

### 1.3 算法描述

简单 element-wise 算子，公式已完整描述计算逻辑，无需复杂算法伪代码。

### 1.4 数据流图

```
    输入 input                    输出 output
┌──────────────────┐         ┌──────────────────┐
│  [m, n] 或       │         │  [m, n] 或       │
│  [b, m, n] 或    │ ──────▶ │  [b, m, n] 或    │
│  [b, s, m, n]    │  relu   │  [b, s, m, n]    │
│  float16/float32 │         │  float16/float32 │
│  /bfloat16       │         │  /bfloat16       │
└──────────────────┘         └──────────────────┘

公式: output_i = max(0, input_i)
约束: Shape仅支持2-4维，Shape Size <= INT32_MAX
      不支持空Tensor，不支持nan/inf
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | res_i = max(0, input_i) | 逐元素取最大值，负数置零 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | max(0, x) | `pypto.relu(input)` | input: Tensor | `docs/api/operation/pypto-relu.md` |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
out[:] = pypto.relu(input)
```

### 2.4 设计依据

- **来源**: api_report.md §3.1 API 映射结果
- **说明**: PyPTO 提供直接的 `pypto.relu()` API，完全满足需求规格。无需 substitute，API 级别为 direct。

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class ReluInput:
    input: Tensor  # 源操作数, shape: [m, n] 或 [b, m, n] 或 [b, s, m, n], dtype: DT_FP16/DT_FP32/DT_BF16
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class ReluOutput:
    output: Tensor  # 输出Tensor, shape: 与input相同, dtype: 与input相同
```

### 3.3 中间 Tensor 定义

无中间 Tensor。ReLU 是单步 element-wise 操作，输入直接映射到输出。

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| input | ND | 默认格式，由 from_torch 自动推导 |
| output | ND | 与输入格式一致 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| m | 特征维度 | [1, INT32_MAX] |
| n | 特征维度 | [1, INT32_MAX] |
| b | batch维度 | [1, INT32_MAX] |
| s | sequence维度 | [1, INT32_MAX] |

**约束**: Shape Size 不大于 INT32_MAX

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def relu_wrapper(input: pypto.Tensor(), output: pypto.Tensor()):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: ReLU 是纯逐元素操作（element-wise），不涉及矩阵乘法，仅需要对每个输入元素执行 max(0, x) 计算，因此使用 Vector 类型的 tiling 配置。

### 4.2 TileShape 初值设置

```python
# 2D 输入
pypto.set_vec_tile_shapes(32, 128)

# 3D 输入
pypto.set_vec_tile_shapes(1, 32, 128)

# 4D 输入
pypto.set_vec_tile_shapes(1, 1, 32, 128)

# 动态配置（根据输入维度自动适配）
def configure_tiling(x):
    if len(x.shape) == 2:
        pypto.set_vec_tile_shapes(32, 128)
    elif len(x.shape) == 3:
        pypto.set_vec_tile_shapes(1, 32, 128)
    elif len(x.shape) == 4:
        pypto.set_vec_tile_shapes(1, 1, 32, 128)
```

### 4.3 设置依据

1. **TileShape 维度匹配**: TileShape 维度数应与输入 shape 维度数一致（pypto.relu 文档要求）
2. **尾轴 32B 对齐**: fp16/bf16 尾轴需为 16 的倍数，fp32 需为 8 的倍数。选择 128 满足所有 dtype 对齐要求
3. **非尾轴选择**: 32 或 1，便于处理各种 shape 大小
4. **来源**: api_report.md §5 Tiling 需求 + `docs/api/config/pypto-set_vec_tile_shapes.md`

### 4.4 注意事项

- TileShape 维度数必须与输入 shape 维度数一致
- 每个维度必须 > 0
- 最多 4 维

### 4.5 判断依据与适用条件

- **判断依据**: ReLU 是简单 element-wise 操作，框架可自动处理数据切分
- **适用条件**: 2-4 维输入，所有 dtype（FP16/FP32/BF16）
- **不适用场景**: 输入维度超过 4 维时不支持（spec 约束）

---

## 5. Loop 结构设计

### 场景 A：不需要 Loop

> 适用于所有轴编译期已知、单次 Tile 可处理的算子（如逐元素运算）。

- **结论**: 不需要 pypto.loop
- **原因**: ReLU 是简单 element-wise 操作，框架自动处理数据切分，无需手动循环
- **适用条件**: 所有 2-4 维输入，满足 spec 约束的任意 shape
- **限制**: 无
- **处理方式**: 编译器自动处理数据切分，无需手动循环

**依据**: quick_ref.md §2.1 条件 4 - "所有轴编译期已知 & 单次 Tile 可处理 → 不需要 Loop"

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def relu_golden(x: torch.Tensor) -> torch.Tensor:
    """ReLU 参考实现"""
    return torch.nn.functional.relu(x)
```

Golden 实现已生成于 `relu_golden.py`。

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | - | [1024, 1024] | [1024, 1024] | 核心性能场景 |
| 功能_P0 | 功能 | P0 | - | [32, 64] | [32, 64] | 核心功能验证 |
| 功能_P1 | 功能 | P1 | - | [2, 128, 256] | [2, 128, 256] | 3维输入验证 |
| 功能_P2 | 功能 | P2 | - | [1, 1, 64, 64] | [1, 1, 64, 64] | 4维输入验证 |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 全正数输入 | 全部为正值 | 输出应与输入相同 |
| 全负数输入 | 全部为负值 | 输出应全为 0 |
| 混合输入 | 正负混合 | 负数变 0，正数保留 |
| 零值输入 | 包含 0 | 输出 0（边界正常处理） |
| 极小 shape | [1, 1] | 最小有效 shape |
| 各 dtype | FP16/FP32/BF16 | 覆盖所有支持的数据类型 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 3e-3 | 3e-3 |
| float16 | 3e-3 | 3e-3 |
| bfloat16 | 3e-3 | 3e-3 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期目标 |
|----------|------|--------|------|------------|------------|----------|
| 性能_P0 | 性能 | P0 | - | [1024, 1024] | [1024, 1024] | 首跑精度成功性能的 2 倍 |

### 7.2 开箱性能配置

```python
# 推荐初始配置
pypto.set_vec_tile_shapes(32, 128)
```

### 7.3 pass_options 配置

无需特殊 pass_options 配置。

### 7.4 runtime_options 配置

```python
runtime_options={"run_mode": pypto.RunMode.NPU}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- 输入必须 contiguous（调用 from_torch 前需确保 tensor.is_contiguous() == True）
- 不支持 nan/inf（输入不应包含 nan/inf 特殊值）
- Shape 仅支持 2-4 维
- Shape Size 不超过 INT32_MAX
- 不支持空 Tensor

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| 非连续 Tensor | from_torch 输入不连续 | 运行时错误 | 调用前检查 is_contiguous() 或调用 .contiguous() |
| nan/inf 输入 | 输入包含特殊值 | 未定义行为 | 测试时避免生成 nan/inf |
| TileShape 维度不匹配 | TileShape 维度数与输入不同 | 编译错误 | 确保维度数一致 |
| 空 Tensor | shape size 为 0 | API 不支持 | 测试时确保 shape 各维度 > 0 |

### 8.3 特殊场景处理

- **零值处理**: 正常计算，输出 0（spec 明确支持）
- **极值处理**: 正常计算（spec 约束不支持 nan/inf）

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 参考 elementwise_ops.py | 参考 `examples/01_beginner/compute/elementwise_ops.py` 中的 kernel 模式 |
| 参考 activation.py | 参考 `examples/02_intermediate/operators/activation/activation.py` 中的激活函数实现模式 |
| 简单直接 | ReLU 是最简单的激活函数之一，直接调用 pypto.relu 即可 |

---

## 9. 交付件清单

### 9.1 目录结构

```
custom/relu/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── relu_golden.py                   # Golden 参考实现（已有）
├── relu_impl.py                     # 算子实现代码（待实现）
├── test_relu.py                     # 测试代码（待实现）
└── README.md                        # 实现说明（待实现）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 设计 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| relu_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| relu_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_relu.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `relu` |
| 目录名 | 与算子名称一致 | `custom/relu/` |
| Golden 文件 | `{op}_golden.py` | `relu_golden.py` |
| 实现文件 | `{op}_impl.py` | `relu_impl.py` |
| 测试文件 | `test_{op}.py` | `test_relu.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → relu_golden.py → relu_impl.py → test_relu.py
```
