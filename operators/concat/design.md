# concat 算子设计文档

> **算子名称**: concat
> **算子分类**: tensor_manipulation
> **生成时间**: 2026-03-28
> **基于**: spec.md

---

## 1. 概述

### 1.1 功能描述

将多个张量沿指定维度拼接成一个张量。所有输入张量在非拼接维度上的 shape 必须相同，dtype 必须一致。

### 1.2 数学公式

沿 dim 维度拼接多个张量，保持其他维度不变：

$$\text{output}[i_0, ..., i_{\text{dim}}, ..., i_{n-1}] = \text{tensor}_k[i_0, ..., i'_{\text{dim}}, ..., i_{n-1}]$$

其中 $k$ 是第 $i_{\text{dim}}$ 所属的输入张量索引，$i'_{\text{dim}}$ 是在该张量内的偏移。

输出 shape 计算：
$$\text{output.shape}[\text{dim}] = \sum_{k=0}^{N-1} \text{tensor}_k.\text{shape}[\text{dim}]$$

### 1.3 算法描述

```
Algorithm: Tensor Concatenation
────────────────────────────────────
输入: tensors = [T₀, T₁, ..., T_{N-1}], dim
输出: output Tensor

1. 检查所有张量的数量 N >= 2
2. 检查所有张量的 dtype 一致
3. 标准化 dim 参数（处理负数索引）
4. 检查所有张量的维度数相同
5. 检查所有张量在非 dim 维度的 shape 相同
6. 计算输出 shape：
   output_shape[dim] = sum(T_i.shape[dim] for all i)
   output_shape[other] = T_0.shape[other]
7. 分配输出张量内存
8. 沿 dim 维度依次复制各输入张量数据到输出张量
9. return output
```

### 1.4 数据流图

```
     输入 tensors (可变数量 N)             输出 output
┌─────────────────────────┐         ┌──────────────────┐
│ tensor_0 [b, s, d₀]     │         │                  │
│ tensor_1 [b, s, d₁]     │ ──────▶ │ output [b,s,Σdᵢ] │
│ ...                     │ concat  │                  │
│ tensor_{N-1} [b,s,dₙ₋₁} │         │                  │
└─────────────────────────┘         └──────────────────┘

约束条件：
  • 非拼接维度必须相同：所有 tensor 的 b、s 必须一致
  • dtype 必须相同：所有 tensor 的 dtype 一致
  • 拼接轴可变：dim 参数可指定任意轴

动态轴：b (batch), s (seq_len)
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | output = concat(tensors, dim) | 沿 dim 维度拼接多个张量 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | concat(tensors, dim) | `pypto.concat(tensors, dim)` | tensors: List[Tensor], dim: int | `docs/api/operation/pypto-concat.md` |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
pypto.set_vec_tile_shapes(*tile_shapes)
output[:] = pypto.concat(tensors, dim=dim)
```

### 2.4 设计依据

- 来源：api_report.md + docs/api/operation/pypto-concat.md
- 说明：PyPTO 原生支持 `pypto.concat` API，可直接调用完成张量拼接，无需 substitute

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
from dataclasses import dataclass
from typing import List
import pypto

@dataclass
class ConcatInput:
    tensors: List[pypto.Tensor]  # 待拼接的张量列表，每个 shape: [b, s, d_i], dtype: float32/float16/bfloat16
    dim: int                      # 拼接维度，支持负数索引，默认 0
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class ConcatOutput:
    output: pypto.Tensor  # 拼接后的张量，shape: [b, s, Σd_i], dtype 与输入一致
```

### 3.3 中间 Tensor 定义

concat 是基础操作，无中间 Tensor。

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| - | - | - | 无中间 Tensor |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| 所有 tensor | ND | 默认格式，concat 操作不要求特殊内存布局 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| batch (b) | 批次大小 | [1, 65536] |
| seq_len (s) | 序列长度 | [1, 32768] |
| d_i | 特征维度 | [1, 16384] |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def concat_kernel(
    tensor_0: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, d0], pypto.DT_FP32),
    tensor_1: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, d1], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, d0 + d1], pypto.DT_FP32),
    dim: int
):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: concat 是形状操作算子，不涉及矩阵乘法（matmul），仅涉及内存复制和拼接，属于 Vector 类型算子

### 4.2 TileShape 初值设置

```python
# 根据输入维度动态设置 tile shapes
tile_shapes = [8 for _ in range(len(tensors[0].shape))]
pypto.set_vec_tile_shapes(*tile_shapes)
```

**典型配置示例**（3D 输入）：
```python
# 输入 [b, s, d_i]，输出 [b, s, Σd_i]
pypto.set_vec_tile_shapes(8, 128, 1024)  # 适配 [b, s, d] 3D 输入
```

### 4.3 设置依据

1. **维度一致性**：TileShape 维度应和输出一致（来自 `docs/api/operation/pypto-concat.md`）
2. **切分规则**：如输入 tensors 维度为 [m, c1, p], [m, c2, p]，输出为 [m, c1+c2, p]，TileShape 设置为 [m1, n1, p1]，则 m1, p1 分别用于切分 m, p 轴，n1 用于切分 c1 和 c2 轴
3. **尾轴对齐**：fp32 尾轴需为 8 的倍数，fp16/bf16 尾轴需为 16 的倍数（32B 对齐）

### 4.4 注意事项

- **viewshape 约束**：设置 viewshape 时，dim 对应维度不切块（即 viewshape 对应值 >= tensors 任一 tensor 的对应值）
- **TileShape 维度数**：最多 4 维，与输入维度数一致

### 4.5 判断依据与适用条件

- **判断依据**：concat 是内存拷贝密集型操作，TileShape 主要影响数据搬运效率
- **适用条件**：适用于 2-4 维张量拼接场景
- **不适用场景**：超过 4 维的张量（PyPTO concat API 限制）

---

## 5. Loop 结构设计

### 场景 A：不需要 Loop

> 适用于所有轴编译期已知、单次 Tile 可处理的算子（如逐元素运算）。

- **结论**：不需要 pypto.loop
- **原因**：`pypto.concat` API 内部已处理数据搬运和拼接逻辑，编译器自动处理动态轴的数据切分。concat 是基础形状操作，不涉及多步骤分块计算或状态累积。
- **适用条件**：2-4 维张量拼接，张量数量 2-128 个
- **限制**：动态轴由编译器自动处理，无需手动 loop
- **处理方式**：编译器自动处理数据切分，无需手动循环

**注意**：虽然 spec 中声明了动态轴（batch 和 seq_len），但 `pypto.concat` 是原生支持的 API，编译器会自动处理动态轴的数据搬运，不需要在 kernel 代码中显式使用 `pypto.loop`。

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def concat_golden(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """concat 参考实现"""
    # 边界条件处理：单张量输入，返回副本
    if len(tensors) == 1:
        return tensors[0].clone()

    # 过滤空张量
    non_empty = [t for t in tensors if t.shape[dim] > 0]
    if len(non_empty) == 0:
        return tensors[0].clone()
    if len(non_empty) == 1:
        return non_empty[0].clone()

    # 使用 PyTorch 内置 API
    return torch.cat(non_empty, dim=dim)
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_2tensor_P0 | 功能 | P0 | dim=-1, N=2 | [b,s,1024], [b,s,512] | [b,s,1536] | 基础功能验证，拼接两个张量 |
| 功能_3tensor_P1 | 功能 | P1 | dim=0, N=3 | [b1,s,d], [b2,s,d], [b3,s,d] | [b1+b2+b3,s,d] | 拼接三个张量，batch 维度 |
| 性能_large_P0 | 性能 | P0 | dim=-1, N=2 | [4096,1024], [4096,1024] | [4096,2048] | 大规模张量拼接性能 |
| 动态轴_seq_P0 | 功能 | P0 | dim=1, N=2 | [b,s1,d], [b,s2,d] | [b,s1+s2,d] | 动态 seq_len 维度拼接 |
| 动态轴_batch_P0 | 功能 | P0 | dim=0, N=2 | [b1,s,d], [b2,s,d] | [b1+b2,s,d] | 动态 batch 维度拼接 |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 负数索引 | dim=-1, dim=-2 | 验证负数索引正确性 |
| 零值输入 | tensors 含零值 | 零值作为有效数据参与拼接 |
| NaN/Inf | tensors 含 NaN/Inf | NaN/Inf 保持不变 |
| 单张量输入 | N=1 | 返回该张量的副本 |
| 不同 dtype | float32, float16, bfloat16 | 验证 dtype 支持范围 |

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
| 性能_large_P0 | 性能 | P0 | dim=-1, N=2 | [4096,1024], [4096,1024] | [4096,2048] | 待实测（目标：首跑性能的 2 倍） |

### 7.2 开箱性能配置

```python
# 3D 输入 [b, s, d] 的推荐 TileShape
# 尾轴 1024 满足 fp32 的 8 倍对齐要求
pypto.set_vec_tile_shapes(8, 128, 1024)

# 2D 输入 [m, n] 的推荐 TileShape
pypto.set_vec_tile_shapes(8, 1024)
```

### 7.3 pass_options 配置

concat 基础算子无需特殊 pass_options 配置。

### 7.4 runtime_options 配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def concat_kernel(...):
    ...
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- **张量数量限制**：2 <= len(tensors) <= 128（来自 `docs/api/operation/pypto-concat.md`）
- **维度限制**：仅支持 2-4 维张量
- **Shape Size 限制**：<= INT32_MAX (2147483647)
- **非拼接维度约束**：所有张量在非 dim 维度的 shape 必须相同
- **dtype 约束**：所有张量 dtype 必须相同

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| viewshape 不切块 | 设置 viewshape 时 dim 维度切块 | 编译失败或结果错误 | 确保 viewshape[dim] >= 任一输入张量的 shape[dim] |
| 单张量精度问题 | len(tensors) == 1 | 精度暂时不保证 | 建议至少 2 个张量输入 |
| 空 Tensor | 某个输入张量 shape[dim] == 0 | 不支持空 Tensor | 在 golden 中过滤空张量 |
| 维度不匹配 | 非拼接维度 shape 不同 | 运行时错误 | 调用前检查 shape 一致性 |

### 8.3 特殊场景处理

1. **负数索引**：dim 参数支持负数索引（如 -1 表示最后一维），API 内部自动处理
2. **边界情况**：单张量输入时返回副本，golden 函数已处理
3. **动态轴**：使用 `pypto.DYNAMIC` 标记动态轴，编译器自动处理

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 使用 from_torch 确保连续 | 输入张量必须是 contiguous 的，使用 `pypto.from_torch` 转换时自动处理 |
| 动态轴标记 | 使用 `pypto.DYNAMIC` 或 `dynamic_axis` 参数标记动态轴 |
| Tiling 配置 | TileShape 维度与输出一致，尾轴满足对齐要求 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/concat/
├── spec.md                    # 需求规范（已有）
├── api_report.md              # API 探索报告（已有）
├── design.md                  # 设计文档（本文件）
├── concat_golden.py           # Golden 参考实现（已有）
├── concat_impl.py             # 算子实现代码
├── test_concat.py             # 测试代码
└── README.md                  # 实现说明
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 探索 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| concat_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| concat_impl.py | 代码 | 算子核心实现 | pypto-op-develop |
| test_concat.py | 代码 | 测试用例 | pypto-op-develop |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `concat` |
| 目录名 | 与算子名称一致 | `operators/concat/` |
| Golden 文件 | `{op}_golden.py` | `concat_golden.py` |
| 实现文件 | `{op}_impl.py` | `concat_impl.py` |
| 测试文件 | `test_{op}.py` | `test_concat.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → concat_golden.py → concat_impl.py → test_concat.py
```
