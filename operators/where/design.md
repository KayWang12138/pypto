# where 算子设计文档

> **算子名称**: where
> **算子分类**: comparison (element-wise)
> **生成时间**: 2026-03-28T12:20:00Z
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

where 算子根据条件张量从 x 或 y 中选择元素。当 condition 对应位置为 True 时选择 x 的元素，为 False 时选择 y 的元素。输出张量与 x、y 广播后具有相同的 shape。

### 1.2 数学公式

$out_i = condition_i \ ? \ x_i \ : \ y_i$

即：
$$
result_{i}=
\begin{cases}
input_{i} & \text{if } condition_{i}==True \\
other_{i} & \text{if } condition_{i}==False
\end{cases}
$$

### 1.3 数据流图

```
    condition          输入 x            输入 y
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  [b, s, n, d] │  │  [b, s, n, d] │  │  [b, s, n, d] │
│    bool       │  │   float32     │  │   float32     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                │                 │
                ▼                 ▼
         ┌─────────────────────────────┐
         │      pypto.where            │
         │  output[i] = cond[i] ? x[i] : y[i]
         └──────────────┬──────────────┘
                        │
                        ▼
               ┌──────────────┐
               │    输出 out   │
               │  [b, s, n, d] │
               │   float32     │
               └──────────────┘

公式: out_i = condition_i ? x_i : y_i
动态轴: b (batch), s (seq_len)
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $result_i = condition_i \ ? \ input_i \ : \ other_i$ | 逐元素条件选择，基于布尔掩码选择 input 或 other |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | $result_i = condition_i \ ? \ input_i \ : \ other_i$ | `pypto.where(condition, input, other)` | condition: DT_BOOL Tensor; input/other: DT_FP32/DT_FP16/DT_BF16 Tensor 或 float/Element | `docs/api/operation/pypto-where.md` |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
# 1. 设置 TileShape（与输出 shape 维度一致）
pypto.set_vec_tile_shapes(b_tile, s_tile, n_tile, d_tile)

# 2. 调用 where API
out = pypto.where(condition, x, y)

# 3. 返回结果
return out
```

### 2.4 设计依据

- **来源**: api_report.md (API 映射结果为 direct)、docs/api/operation/pypto-where.md
- **说明**: PyPTO 提供直接对应的 `pypto.where` API，无需 Substitute，可直接映射实现

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class WhereInput:
    condition: Tensor  # 条件张量，shape: [b, s, n, d]，dtype: DT_BOOL
    x: Tensor          # condition 为 True 时选择的值，shape: [b, s, n, d]，dtype: DT_FP32/DT_FP16/DT_BF16
    y: Union[Tensor, float, Element]  # condition 为 False 时选择的值，支持张量或标量
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class WhereOutput:
    out: Tensor  # 根据 condition 从 x 和 y 选择的元素，shape: [b, s, n, d]，dtype: 与 x/y 一致
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| out | [b, s, n, d] | 与 x/y 一致 | 输出张量，无需中间 Tensor |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| condition/x/y/out | ND | 逐元素操作，默认 ND 格式即可 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| b | batch 维度，批次大小 | [1, INT32_MAX] |
| s | seq_len 维度，序列长度 | [1, INT32_MAX] |
| n | head 维度，注意力头数 | 固定或动态 |
| d | hidden_dim 维度，隐藏层维度 | 固定或动态 |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def where(inputs: WhereInput) -> WhereOutput:
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: where 是逐元素条件选择操作，不涉及矩阵乘法（matmul），仅需要 Vector 类型的 tiling 配置，使用 `set_vec_tile_shapes`

### 4.2 TileShape 初值设置

```python
# 输出 shape 为 [b, s, n, d]，TileShape 设置为 4 维
# 考虑 L0 容量和 32B 对齐要求
# fp32: 尾轴需为 8 的倍数；fp16/bf16: 尾轴需为 16 的倍数
pypto.set_vec_tile_shapes(1, 128, 8, 64)
```

### 4.3 设置依据

1. **维度匹配**: TileShape 维度应与输出一致，输出为 4D [b, s, n, d]，故 TileShape 也为 4D
2. **尾轴对齐**: 尾轴 64 满足 32B 对齐（fp32: 64/8=8，fp16: 64/16=4）
3. **广播场景**: 广播场景下 TileShape 仍按输出 shape 设置，编译器自动处理广播轴
4. **L0 容量**: (1, 128, 8, 64) * 4 bytes * 3 (condition+x+y) = 96KB，在 L0 容量范围内

### 4.4 注意事项

- TileShape 每个维度必须大于 0
- 最多支持 4 维 TileShape
- 广播场景下 TileShape 按输出 shape 设置，不按输入 shape

### 4.5 判断依据与适用条件

- **判断依据**: where 是简单逐元素操作，使用 Vector 类型 tiling，TileShape 与输出 shape 维度一致
- **适用条件**: 适用于 2-4D shape 的 where 操作，支持广播场景
- **不适用场景**: 超过 4D 的 shape 需要先 reshape；空 Tensor 不支持

---

## 5. Loop 结构设计

### 场景 A：不需要 Loop

> 适用于所有轴编译期已知、单次 Tile 可处理的算子（如逐元素运算）。

- **结论**: 不需要 pypto.loop
- **原因**: where 是简单逐元素操作，编译器自动处理数据切分，无需手动循环。当存在动态轴时，编译器会生成动态 shape 的 kernel
- **适用条件**: 适用于 2-4D shape 的 where 操作
- **限制**: 如果动态轴范围跨度极大（如 1~64k），可能需要考虑 `loop_unroll` 优化
- **处理方式**: 编译器自动处理数据切分，无需手动循环

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def where_golden(
    condition: torch.Tensor,
    x: torch.Tensor,
    y: Union[torch.Tensor, float, int],
) -> torch.Tensor:
    """where 参考实现"""
    return torch.where(condition, x, y)
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | 无特殊参数 | condition:[b,s,1,1], x:[b,s,n,d], y:[b,s,n,d] | [b,s,n,d] | 广播场景性能测试 |
| 功能_P0 | 功能 | P0 | 无特殊参数 | condition:[b,s,n,d], x:[b,s,n,d], y:[b,s,n,d] | [b,s,n,d] | 基础功能验证 |
| 功能_P1 | 功能 | P1 | y为标量 | condition:[b,s,n,d], x:[b,s,n,d], y:scalar | [b,s,n,d] | 标量 y 支持 |
| 动态轴 | 功能 | P1 | 动态 b,s | condition:[b,s,n,d], x:[b,s,n,d], y:[b,s,n,d] | [b,s,n,d] | 动态轴验证 |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 最小 shape | b=1, s=1, n=1, d=1 | 边界最小值 |
| 大 shape | b=128, s=2048, n=16, d=128 | 边界最大值 |
| 不同 dtype | fp32/fp16/bf16 | dtype 支持验证 |
| 标量 y | y=0.0 或 y=float | 标量输入验证 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.001 | 0.001 |
| bfloat16 | 0.01 | 0.01 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 性能_P0 | 性能 | P0 | 无特殊参数 | condition:[2,128,1,1], x:[2,128,8,64], y:[2,128,8,64] | [2,128,8,64] | 待实测 |

### 7.2 开箱性能配置

```python
# 开箱 TileShape 配置
# 输出 shape: [b, s, n, d] = [batch, seq_len, num_heads, head_dim]
# 典型配置: b=2, s=128, n=8, d=64
pypto.set_vec_tile_shapes(1, 128, 8, 64)
```

### 7.3 pass_options 配置

无需特殊 pass_options 配置。

### 7.4 runtime_options 配置

```python
runtime_options = {
    "run_mode": pypto.RunMode.NPU  # 0=NPU, 1=模拟器
}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- condition 必须是 DT_BOOL 类型的 Tensor
- input/other 支持 DT_FP32、DT_FP16、DT_BF16，不支持空 Tensor
- Shape 仅支持 2-4 维，Shape Size 不大于 INT32_MAX
- 广播规则只支持单轴广播

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| fp16 标量精度问题 | 直接传入 float 标量给 fp16 场景 | 不保证正确性 | 使用 Element 类型传入标量 |
| 多轴广播失败 | condition/x/y 存在多轴同时广播 | PyPTO where 只支持单轴广播 | 预处理输入，先 broadcast 到统一 shape |
| 空 Tensor 错误 | 任一输入为空 Tensor | 不支持空 Tensor | 确保输入非空 |
| 维度超限 | shape 维度 > 4 | 编译错误 | 先 reshape 到 4D 以内 |

### 8.3 特殊场景处理

1. **广播场景**: condition shape 与 x/y 不同但可广播时，确保符合单轴广播规则
2. **标量 other**: 使用 `pypto.where(condition, x, 0.0)` 时，建议用 Element 类型确保 dtype 一致
3. **动态轴**: 存在动态轴 b、s 时，需在 from_torch 时正确标记 dynamic_axis

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 使用 Element 传标量 | 对于 fp16/bf16 场景，优先使用 `pypto.Element(pypto.DT_FP16, value)` 而非直接传 float |
| TileShape 与输出一致 | 无论是否有广播，TileShape 都按输出 shape 设置 |
| from_torch 标记动态轴 | 使用 `dynamic_axis` 参数标记 batch 和 seq_len 为动态 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/where/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── where_golden.py                  # Golden 参考实现（已有）
├── where_impl.py                    # 算子实现代码
├── test_where.py                    # 测试代码
└── output/                          # 运行输出（自动生成）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 探索 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| where_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| where_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_where.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `where` |
| 目录名 | 与算子名称一致 | `operators/where/` |
| Golden 文件 | `{op}_golden.py` | `where_golden.py` |
| 实现文件 | `{op}_impl.py` | `where_impl.py` |
| 测试文件 | `test_{op}.py` | `test_where.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → where_golden.py → where_impl.py → test_where.py
```
