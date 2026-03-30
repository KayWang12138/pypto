# PyPTO 算子设计文档： embedding

## 基本信息

| 项目 | 内容 |
|------|------|
| **算子名称**: embedding |
| **算子分类**: embedding |
| **生成时间**: 2026-03-30 |
| **基于**: spec.md, api_report.md |

---

## 1. 概述

### 1.1 功能描述

Embedding 算子实现查表操作，将索引张量映射到对应的嵌入向量。这是自然语言处理和推荐系统中常用的基础算子，支持动态轴（batch, seq）和可选的 padding_idx 功能。

### 1.2 数学公式

```
output = embedding_lookup(weight, indices)

核心公式:
  output[i, j, :] = weight[indices[i, j], :]

可选 padding_idx 处理:
  if padding_idx is not None:
    mask = (indices == padding_idx).unsqueeze(-1)  # [batch, seq, 1]
    output = output.masked_fill(mask, 0.0)
```

### 1.3 数据流图

```
    输入 indices              权重 weight (embedding table)
+------------------+      +----------------------------+
|  [batch, seq]    |      |  [vocab_size, embed_dim]   |
|     int64        |      |       float32              |
+--------+---------+      +-------------+--------------+
         |                              |
         |    +-------------------------+
         |    |
         v    v
    +-------------------------------------+
    |         embedding 查表操作          |
    |   output[i,j,:] = weight[indices[i,j], :]   |
    |   若 padding_idx 不为 None 且      |
    |   indices[i,j] == padding_idx，    |
    |   则 output[i,j,:] = 0             |
    +-----------------+-------------------+
                      |
                      v
              +------------------+
              |  输出 output      |
              |[batch, seq, embed_dim] |
              |     float32      |
              +------------------+
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将 embedding 公式拆解为基本操作步骤:

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `output = weight[indices]` | 核心 embedding 查表 |
| 2 | `mask = (indices == padding_idx)` | padding 索引检测 (可选) |
| 3 | `mask_3d = mask.unsqueeze(-1)` | mask 维度扩展 (可选) |
| 4 | `output = where(mask_3d, 0.0, output)` | 条件填充 (可选) |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | `output = weight[indices]` | `pypto.gather` | input=weight, dim=0, index=indices | docs/api/operation/pypto-gather.md |
| 2 | `mask = (indices == padding_idx)` | `pypto.eq` | input=indices_fp32, other=float(padding_idx) | docs/api/operation/pypto-eq.md |
| 3 | `mask_3d = mask.unsqueeze(-1)` | `pypto.unsqueeze` | input=mask, dim=-1 | docs/api/operation/pypto-unsqueeze.md |
| 4 | `output = where(mask_3d, 0.0, output)` | `pypto.where` | condition=mask_3d, input=0.0, other=output | docs/api/operation/pypto-where.md |
| 5 | `indices_fp32 = cast(indices, FP32)` | `pypto.cast` | input=indices, dtype=DT_FP32 | docs/api/operation/pypto-cast.md |

### 2.3 计算步骤序列

```python
# 1. 核心查表操作
output = pypto.gather(weight, dim=0, index=indices)

# 2. padding_idx 处理（可选）
if padding_idx is not None:
    # 2.1 indices 转换为 FP32（eq 不支持 INT64）
    indices_fp32 = pypto.cast(indices, pypto.DT_FP32)
    # 2.2 检测 padding 位置
    mask = pypto.eq(indices_fp32, float(padding_idx))
    # 2.3 扩展 mask 维度 [batch, seq] -> [batch, seq, 1]
    mask_3d = pypto.unsqueeze(mask, dim=-1)
    # 2.4 条件填充
    output = pypto.where(mask_3d, 0.0, output)
```

### 2.4 设计依据

- 来源: spec.md（输入输出规格、动态轴定义）, api_report.md（API 映射、约束分析）
- 说明: 选择 pypto.gather 作为核心 API，因为它直接支持索引查表操作，满足 embedding 的核心需求

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class EmbeddingInput:
    indices: Tensor  # 索引张量，shape [batch, seq], dtype int64, 动态轴 [0, 1]
    weight: Tensor  # Embedding 表, shape [vocab_size, embed_dim], dtype float32, 騾态
    padding_idx: Optional[int] = None  # 填充索引
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class EmbeddingOutput:
    output: Tensor  # 嵌入向量输出, shape [batch, seq, embed_dim], dtype float32, 动态轴 [0, 1]
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| indices_fp32 | [batch, seq] | float32 | indices 转换后的张量（用于 eq 比较） |
| mask | [batch, seq] | bool | padding 位置掩码 |
| mask_3d | [batch, seq, 1] | bool | 扩展后的掩码，用于 where |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| indices | ND | 一维索引张量，使用 ND 格式 |
| weight | ND | 二维权重矩阵，使用 ND 格式 |
| output | ND | 输出张量，使用 ND 格式 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| batch | 批次维度 | [1, 1024] |
| seq | 序列长度 | [1, 4096] |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128}
)
def embedding_kernel(
    indices: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_INT64),
    weight: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    padding_idx: int = None
):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: embedding 核心操作是 pypto.gather（索引查表)，不涉及 matmul

### 4.2 TileShape 初值设置

```python
pypto.set_vec_tile_shapes(tile_batch, tile_seq, tile_embed_dim)
```

### 4.3 设置依据

- gather 的 dim=0 轴 (vocab_size) 不可切，需要全载
- batch 和 seq 是动态轴，需要分块处理
- embed_dim 可以分块以优化性能

- 典型配置: `(1, 128, 256)` 或 `(1, 64, 512)`

### 4.4 注意事项

- vocab_size 轴不可切，viewshape[0] 需要大于等于 vocab_size
- TileShape 的选择需要考虑 UB 内存限制
- 大 vocab_size 场景可能需要更小的 tile_batch

- 建议根据实际 shape 动态调整 TileShape

### 4.5 判断依据与适用条件

- 判断依据: pypto.gather 要求 dim=0 轴不可切
- 适用条件: 适用于 batch/seq 维度需要动态变化的场景
- 不适用场景: vocab_size 鼚特别大（>100k）时可能需要更细粒度的分块策略

---

## 5. Loop 结构设计

### 5.1 Loop 判断结论

- **结论**: 需要 Loop
- **原因**: batch 和 seq 是动态轴，且 gather 的 vocab_size 轴不可切，需要分批处理以适应不同 shape
- **Loop 类型**: pypto.loop
- **适用条件**: batch 和 seq 维度动态变化范围大（1~1024 for batch, 1~4096 for seq）
- **限制**: vocab_size 轴需要全载，不能分块

- **Loop 策略**: 按 batch 维度循环，每次处理一个 batch 的所有 seq
 **实现简单，性能可控**

### 5.2 静态轴 vs 动态轴处理

| 轴 | 类型 | 处理方式 |
|----|------|----------|
| batch | 动态 | pypto.loop 循环处理 |
| seq | 动态 | 单次循环内全量处理（vocab_size 不可切，seq 维度影响相对较小） |
| embed_dim | 静态 | 可分块处理（通过 TileShape 控制） |
| vocab_size | 静态 | 全载，不可分块 |

### 5.3 Loop 合并策略

- 策略: 按 batch 维度循环
- 每次循环处理一个完整的 batch
- seq 维度在单次循环内全量处理（因为 vocab_size 不可切）
- embed_dim 通过 TileShape 控制分块大小

### 5.4 数据依赖处理

- 无跨循环数据依赖
- 每个循环独立处理一个 batch 的数据

### 5.5 尾块处理策略

- batch 维度的尾块： 最后一次循环处理剩余的 batch 数量
- seq 维度: 无尾块（vocab_size 不可切，seq 维度在单次循环内全量处理）
- embed_dim 维度: 无尾块（由 TileShape 控制）

### 5.6 loop_unroll 配置

```python
# 根据 batch 范围动态调整 loop_unroll
# batch 范围 [1, 1024]，按区间配置 loop_unroll
loop_unroll = {
    (1, 128): 1,    # 小 batch 不需要 unroll
    (129, 512): 2,   # 中等 batch 2x unroll
    (513, 1024): 4,  # 大 batch 4x unroll
}
```

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def embedding_golden(
    indices: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: Optional[int] = None
) -> torch.Tensor:
    """embedding 参考实现"""
    output = weight[indices]
    if padding_idx is not None:
        mask = (indices == padding_idx).unsqueeze(-1)
        output = output.masked_fill(mask, 0.0)
    return output
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 小模型 | 性能 | P0 | padding_idx=None | indices: [1, 128], weight: [32000, 4096] | [1, 128, 4096] | 小 batch 测试 |
| 中模型 | 性能 | P0 | padding_idx=None | indices: [8, 512], weight: [32000, 4096] | [8, 512, 4096] | 典型推理场景 |
| 大模型 | 性能 | P0 | padding_idx=None | indices: [32, 2048], weight: [128000, 4096] | [32, 2048, 4096] | 大模型推理 |
| padding 功能 | 功能 | P1 | padding_idx=0 | indices: [4, 256], weight: [10000, 256] | [4, 256, 256] | 验证 padding 功能 |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 空 batch | batch=0, seq=128 | 验证空输入处理 |
| 空 seq | batch=4, seq=0 | 验证空序列处理 |
| 最小 batch | batch=1, seq=1 | 验证最小输入 |
| 大 vocab_size | vocab_size=200000 | 验证大词表场景 |

### 6.3 精度验证标准
| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.001 | 0.001 |
| bfloat16 | 0.001 | 0.001 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 小模型 | 性能 | P0 | padding_idx=None | indices: [1, 128], weight: [32000, 4096] | [1, 128, 4096] | 待测试 |
| 中模型 | 性能 | P0 | padding_idx=None | indices: [8, 512], weight: [32000, 4096] | [8, 512, 4096] | 待测试 |
| 大模型 | 性能 | P0 | padding_idx=None | indices: [32, 2048], weight: [128000, 4096] | [32, 2048, 4096] | 待测试 |

### 7.2 开箱性能配置

```python
# 小模型配置
pypto.set_vec_tile_shapes(1, 128, 256)

# 中模型配置
pypto.set_vec_tile_shapes(1, 256, 512)

# 大模型配置
pypto.set_vec_tile_shapes(1, 512, 1024)
```

### 7.3 runtime_options 配置
```python
@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "stitch_cfgcache_size": 2500000
    }
)
```

---

## 8. 风险点与注意事项

### 8.1 已知约束
- pypto.gather 要求 dim=0 轴 (vocab_size) 不可切，需要全载
- pypto.eq 不直接支持 INT64 类型，需要先 cast 为 FP32
- indices 的值必须保证在 [0, vocab_size) 范围内，否则行为未定义

- padding_idx 处理需要额外的 cast 和比较操作,可能影响性能

### 8.2 常见错误规避
| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| gather shape 不匹配 | indices 不是 2D | 运行时错误 | 确保 indices 维度为 [batch, seq] |
| eq dtype 不支持 | indices 为 INT64 | 编译/运行错误 | 使用 cast 转换为 FP32 |
| vocab_size 过大 | vocab_size > UB 容量 | 内存溢出 | 减小 tile_batch，分多次处理 |
| padding_idx 越界 | padding_idx >= vocab_size | 逻辑错误 | 参数校验 |

### 8.3 特殊场景处理
- **空输入处理**: batch=0 或 seq=0 时，直接返回空张量
- **大 vocab_size**: vocab_size > 100k 时，建议减小 tile_batch 以避免内存压力
- **动态轴边界**: batch 或 seq 接近范围边界时，确保 TileShape 正确设置

### 8.4 实现建议
| 建议项 | 说明 |
|--------|------|
| 优先保证 vocab_size 全载 | vocab_size 轴不可切，必须全载到 UB |
| 合理设置 TileShape | 根据实际 shape 动态调整 tile_batch/tile_seq/tile_embed_dim |
| padding_idx 可选实现 | padding_idx 是 P1 功能，根据需求决定是否实现 |
| 动态轴边界测试 | 重点测试 batch/seq 的最小值和最大值 |

---

## 9. 交付件清单

### 9.1 目录结构
```
operators/embedding/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── embedding_golden.py              # Golden 参考实现（已有）
├── embedding_impl.py              # 算子实现代码（待实现）
└── test_embedding.py              # 测试代码（待实现）
```

### 9.2 文件清单
| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 设计 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| embedding_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| embedding_impl.py | 代码 | 算子核心实现 | pypto-op-develop（待调用） |
| test_embedding.py | 代码 | 测试用例 | pypto-op-develop（待调用） |

### 9.3 命名规范
| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | embedding |
| 目录名 | 与算子名称一致 | operators/embedding/ |
| Golden 文件 | {op}_golden.py | embedding_golden.py |
| 实现文件 | {op}_impl.py | embedding_impl.py |
| 测试文件 | test_{op}.py | test_embedding.py |

### 9.4 生成顺序
```
spec.md -> api_report.md -> design.md -> embedding_golden.py -> embedding_impl.py -> test_embedding.py
```
