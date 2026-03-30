# flash_attention 算子设计文档

> **算子名称**: flash_attention
> **算子分类**: attention
> **生成时间**: 2026-03-29T00:00:00Z
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

Flash Attention 是一种内存高效的注意力计算方法，通过分块计算和在线 Softmax 算法，将 HBM 访问复杂度从 O(N^2) 降低到 O(N)，显著减少内存读写次数，提升大序列长度场景下的性能。

核心特性:
- **分块计算**: 将 Q/K/V 分成小块 (tiles)，每次只加载一个 tile 到 SRAM
- **在线 Softmax**: 使用在线 Softmax 算法，增量更新 max 值和累加值，避免存储完整 N^2 的注意力矩阵
- **内存优化**: 只需 O(N) 而非 O(N^2) 的 HBM 访问

### 1.2 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中:
- $Q \in \mathbb{R}^{N \times H \times L \times d}$: Query 张量
- $K \in \mathbb{R}^{N \times H \times S \times d}$: Key 张量
- $V \in \mathbb{R}^{N \times H \times S \times d}$: Value 张量
- $d$: 头维度 (head dimension)
- 缩放因子: $\frac{1}{\sqrt{d}}$

### 1.3 算法描述

```
Algorithm: Flash Attention (Forward)
────────────────────────────────────
输入: Q, K, V in R^{N x H x L x d}, 分块大小 Br, Bc, scale
输出: O in R^{N x H x L x d}

1. 初始化:
   - 将 Q 分为 Tr = ceil(L / Br) 块 (Q_1, ..., Q_Tr)
   - 将 K, V 分为 Tc = ceil(S / Bc) 块 (K_1, ..., K_Tc), (V_1, ..., V_Tc)
   - 初始化 O = zeros(N, H, L, d), l = zeros(N, H, L, 1), m = -inf(N, H, L, 1)

2. 外层循环: 遍历 K/V 块
   for j = 1 to Tc:
     2.1 加载 K_j, V_j 到 SRAM (shape: [N, H, Bc, d])

3. 内层循环: 遍历 Q 块
   for i = 1 to Tr:
     3.1 加载 Q_i, O_i, l_i, m_i 到 SRAM

     3.2 计算当前块的注意力分数:
         S_ij = Q_i @ K_j^T * scale  (shape: [Br, Bc])

     3.3 应用掩码 (如果需要):
         if causal_mask:
             S_ij = apply_causal_mask(S_ij, i, j)
         if attn_mask:
             S_ij = S_ij + attn_mask_ij

     3.4 在线 Softmax 更新:
         - m_ij_new = max(m_i, rowmax(S_ij))
         - P_ij = exp(S_ij - m_ij_new)
         - l_ij_new = exp(m_i - m_ij_new) * l_i + rowsum(P_ij)

     3.5 更新输出:
         O_i = (exp(m_i - m_ij_new) * l_i / l_ij_new) * O_i
             + (P_ij / l_ij_new) @ V_j

     3.6 更新状态:
         m_i = m_ij_new
         l_i = l_ij_new

     3.7 写回 O_i, l_i, m_i 到 HBM

4. 返回 O
```

### 1.4 数据流图

```
        Q                    K                    V
   [N, H, L, d]        [N, H, S, d]        [N, H, S, d]
        │                    │                    │
        │              ┌─────┴─────┐              │
        │              │ 分块加载   │              │
        │              │ K_j, V_j  │              │
        │              └─────┬─────┘              │
        │                    │                    │
        │    ┌───────────────┼────────────────────┤
        │    │               │                    │
        ▼    ▼               ▼                    │
   ┌─────────────┐     ┌─────────────┐           │
   │ 分块 Q_i    │     │ K_j^T      │           │
   └──────┬──────┘     └──────┬──────┘           │
          │                   │                  │
          └────────┬──────────┘                  │
                   │                             │
                   ▼                             │
            ┌────────────┐                       │
            │  Q_i @ K_j │                       │
            │  * scale   │                       │
            │ [Br, Bc]   │                       │
            └─────┬──────┘                       │
                  │                              │
                  │    causal_mask / attn_mask   │
                  │         │                    │
                  ▼         ▼                    │
            ┌───────────────────┐                │
            │   apply_mask      │                │
            └─────────┬─────────┘                │
                      │                          │
                      ▼                          │
            ┌───────────────────┐                │
            │  Online Softmax   │                │
            │  m_new, l_new     │                │
            └─────────┬─────────┘                │
                      │                          │
                      └──────────────┬───────────┘
                                     │
                                     ▼
                              ┌────────────┐
                              │  P_ij @ V_j│
                              │  更新 O_i  │
                              │ [Br, d]    │
                              └─────┬──────┘
                                    │
                                    ▼
                             输出 O_i
                            [Br, d]
                                    │
                      ┌─────────────┴─────────────┐
                      │  循环直到所有块处理完毕    │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                             最终输出 O
                            [N, H, L, d]
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤:

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $K^T$ | K 转置: [N, H, S, d] -> [N, H, d, S] |
| 2 | $Q @ K^T$ | 计算注意力分数: [N, H, L, d] @ [N, H, d, S] -> [N, H, L, S] |
| 3 | $\times \frac{1}{\sqrt{d}}$ | 缩放注意力分数 |
| 4 | $+ \text{mask}$ | 应用注意力掩码 (可选) |
| 5 | $\text{rowmax}(S)$ | 在线 Softmax: 计算 max 值 |
| 6 | $S - m$ | 数值稳定减法 |
| 7 | $\exp(S - m)$ | 计算指数 |
| 8 | $\text{rowsum}(P)$ | 在线 Softmax: 计算累加值 |
| 9 | $P / l$ | 归一化 |
| 10 | $P @ V$ | 计算输出: [N, H, L, S] @ [N, H, S, d] -> [N, H, L, d] |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | $K^T$ | `pypto.transpose(input, 2, 3)` | input=K, dim0=2, dim1=3 | `docs/api/operation/pypto-transpose.md` |
| 2 | $Q @ K^T$ | `pypto.matmul(q, k_t, out_dtype, b_trans=True)` | input=Q, mat2=K, out_dtype=FP32, b_trans=True | `docs/api/operation/pypto-matmul.md` |
| 3 | $\times scale$ | `pypto.mul(scores, scale)` | input=scores, other=scale | `docs/api/operation/pypto-mul.md` |
| 4 | $+ mask$ | `pypto.add(scores, mask)` | input=scores, other=mask | `docs/api/operation/pypto-add.md` |
| 5 | rowmax | `pypto.amax(scores, dim=-1, keepdim=True)` | input=scores, dim=-1 | `docs/api/operation/pypto-amax.md` |
| 6 | $- m$ | `pypto.sub(scores, max_val)` | input=scores, other=max | `docs/api/operation/pypto-sub.md` |
| 7 | exp | `pypto.exp(scores_sub)` | input=scores_sub | `docs/api/operation/pypto-exp.md` |
| 8 | rowsum | `pypto.sum(exp_scores, dim=-1, keepdim=True)` | input=exp_scores, dim=-1 | `docs/api/operation/pypto-sum.md` |
| 9 | $/ l$ | `pypto.div(output, sum_val)` | input=output, other=sum | `docs/api/operation/pypto-div.md` |
| 10 | $P @ V$ | `pypto.matmul(p, v, out_dtype)` | input=p, mat2=v | `docs/api/operation/pypto-matmul.md` |
| - | max update | `pypto.maximum(m_old, m_new)` | input=m_old, other=m_new | `docs/api/operation/pypto-maximum.md` |
| - | dtype cast | `pypto.cast(input, dtype)` | dtype=DT_FP32/DT_BF16 | `docs/api/operation/pypto-cast.md` |
| - | causal mask | `pypto.triu(input, diagonal=1)` | diagonal=1 | `docs/api/operation/pypto-triu.md` |

### 2.3 计算步骤序列

```python
# Flash Attention 在线 Softmax 核心逻辑
# 1. K 转置
k_t = pypto.transpose(k, 2, 3)

# 2. 计算注意力分数 (使用 b_trans=True 避免显式转置)
scores = pypto.matmul(q, k_t, out_dtype=pypto.DT_FP32, b_trans=True)

# 3. 缩放
scores_scaled = pypto.mul(scores, scale)

# 4. 应用掩码 (可选)
if mask is not None:
    scores_scaled = pypto.add(scores_scaled, mask)

# 5-9. 在线 Softmax (分块计算中迭代更新)
# m_new = rowmax(scores)
m_new = pypto.amax(scores_scaled, dim=-1, keepdim=True)

# scores - m_new
scores_sub = pypto.sub(scores_scaled, m_new)

# exp(scores - m_new)
p = pypto.exp(scores_sub)

# sum(exp_scores)
l_new = pypto.sum(p, dim=-1, keepdim=True)

# 10. 计算输出
output = pypto.matmul(p, v, out_dtype=pypto.DT_FP32)

# 归一化
output = pypto.div(output, l_new)
```

### 2.4 设计依据

- **来源**: api_report.md, GLM-4.5 Flash Attention 实现 (`models/glm_v4_5/glm_attention.py`), PyPTO API 文档
- **说明**:
  - matmul 使用 `b_trans=True` 参数优化 K 转置，避免显式 transpose
  - softmax 计算必须在 FP32 下进行，保证数值稳定性
  - 在线 Softmax 使用 amax + exp + sum 组合实现
  - 参考 GLM-4.5 的分块策略和循环结构

---

## 3. 数据规格设计

### 3.1 FlashAttentionInput dataclass

```python
@dataclass
class FlashAttentionInput:
    query: Tensor      # Query tensor [N, H, L, d], float32/bfloat16
    key: Tensor        # Key tensor [N, H, S, d], float32/bfloat16
    value: Tensor      # Value tensor [N, H, S, d], float32/bfloat16
    attn_mask: Optional[Tensor] = None  # Attention mask [N, H, L, S] or [L, S], float/bool
    is_causal: bool = False              # Whether to apply causal mask
    scale: Optional[float] = None        # Scale factor (default: 1/sqrt(d))
```

### 3.2 FlashAttentionOutput dataclass

```python
@dataclass
class FlashAttentionOutput:
    output: Tensor     # Output tensor [N, H, L, d], float32/bfloat16
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| k_t | [N, H, d, S] | FP32/BF16 | K 转置后 |
| scores | [N, H, L, S] | FP32 | 注意力分数 (matmul 输出) |
| scores_scaled | [N, H, L, S] | FP32 | 缩放后的注意力分数 |
| m | [N, H, L, 1] | FP32 | 在线 Softmax max 值 |
| m_new | [N, H, Br, 1] | FP32 | 当前块 max 值 |
| p | [N, H, Br, Bc] | FP32 | exp(scores - m_new) |
| l | [N, H, L, 1] | FP32 | 在线 Softmax 累加值 |
| l_new | [N, H, Br, 1] | FP32 | 当前块累加值 |
| o_acc | [N, H, L, d] | FP32 | 累加的输出 |
| causal_mask | [L, S] | FP32 | 因果掩码 (预计算) |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| Q, K, V | ND | 标准 ND 格式，4D tensor |
| scores | ND | matmul 输出使用 ND 格式 |
| output | ND | 输出使用 ND 格式 |

**格式选择理由**:
- PyPTO transpose 4D 只支持 (2,3) 交换，ND 格式更通用
- 参考 GLM-4.5 和 examples 中的 attention 实现都使用 ND 格式
- NZ 格式需要额外的对齐处理，增加复杂度

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| N | Batch size | [1, 1024] |
| H | Number of heads | [1, 128] |
| L | Query sequence length | [1, 8192] |
| S | Key/Value sequence length | [1, 8192] |
| d | Head dimension | [64, 256] |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def flash_attention_kernel(
    query: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    key: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    value: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    scale: float,
):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: 混合 (Cube + Vector)
- **判断依据**:
  - 含 matmul 操作 (Q @ K^T, P @ V) → 需要 Cube Tiling
  - 含逐元素操作 (mul, add, sub, exp) 和归约操作 (amax, sum) → 需要 Vector Tiling
  - Flash Attention 核心是分块计算，两者都需要

### 4.2 TileShape 初值设置

```python
# Cube tiling for matmul operations
# 参考 GLM-4.5 Flash Attention 实现
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# Vector tiling for elementwise operations
# 4D tensor [batch, heads, seq, dim]
pypto.set_vec_tile_shapes(1, 8, 16, 128)
```

### 4.3 设置依据

- **来源**: GLM-4.5 Flash Attention (`models/glm_v4_5/glm_attention.py`)
- **判断依据**:
  - Cube tile shapes 设置 M, K, N 轴的切分大小
  - Vector tile shapes 用于逐元素和归约操作
  - 分块大小需要平衡并行度和内存使用
- **适用条件**: 适用于大多数标准 attention 场景 (L, S <= 4096)
- **不适用场景**: 超长序列 (>4096) 可能需要更小的分块大小

### 4.4 注意事项

- **matmul 前必须调用 set_cube_tile_shapes**: 调用 matmul 前必须设置 cube tiling
- **softmax 精度问题**: PyPTO softmax 仅支持 FP32，需要进行类型转换
- **transpose 4D 约束**: 只支持 (2,3) 交换，其他交换需要通过组合实现
- **尾轴 32B 对齐**: Vector tiling 尾轴需要 32 字节对齐

### 4.5 分块参数设计

| 参数 | 值 | 说明 |
|------|-----|------|
| Br (Q 分块) | 128 | Query 块大小 |
| Bc (K/V 分块) | 512 | Key/Value 块大小 |
| g_tile | 8 | 并行 head 数 |

---

## 5. Loop 结构设计

### 5.1 Loop 判断结论

- **结论**: 需要 Loop
- **原因**: Flash Attention 需要分块计算，遍历 K/V 块和 Q 块进行在线 Softmax 更新
- **Loop 类型**: `pypto.loop` + Python for
- **适用条件**: 所有 Flash Attention 场景
- **限制**: 需要正确处理尾块和边界条件

### 5.2 静态轴 vs 动态轴处理

| 轴 | 类型 | 处理方式 |
|----|------|----------|
| N (batch) | 动态 | pypto.loop |
| H (num_heads) | 静态/动态 | Python for / pypto.loop |
| L (seq_len_q) | 动态 | pypto.loop |
| S (seq_len_kv) | 动态 | pypto.loop |

### 5.3 循环结构设计

```python
# 外层循环: 遍历 batch
for b_idx in pypto.loop(batch_size, name="LOOP_batch"):
    # 中层循环: 遍历 heads (静态展开)
    for h_idx in range(num_heads):
        # 内层循环: 遍历 K/V 块
        for s_idx in pypto.loop(s_loop, name="LOOP_s", unroll_list=[8, 4, 2, 1]):
            # 最内层循环: 遍历 Q 块
            for q_idx in pypto.loop(q_loop, name="LOOP_q"):
                # 在线 Softmax 更新逻辑
                ...
```

### 5.4 数据依赖处理

- **m (max) 更新**: 当前块的 max 值需要与之前的 max 值比较
- **l (sum) 更新**: 需要根据 max 值变化调整之前的累加值
- **O (output) 更新**: 需要根据 max 和 sum 的变化调整之前的输出

```python
# 在线 Softmax 更新公式
m_new = pypto.maximum(m_old, m_cur)
l_scale = pypto.exp(pypto.sub(m_old, m_new))
l_new = pypto.add(pypto.mul(l_old, l_scale), l_cur)
o_scale = pypto.exp(pypto.sub(m_old, m_new))
o_new = pypto.add(pypto.mul(o_old, o_scale), o_cur)
```

### 5.5 尾块处理策略

```python
# 计算实际块大小 (处理尾块)
actual_block_size = pypto.min(block_size, remaining_size)

# 使用 valid_shape 指定有效区域
block_view = pypto.view(tensor, [block_size, dim], [offset, 0], valid_shape=[actual_block_size, dim])
```

### 5.6 循环边界控制

```python
# 使用 is_loop_begin 和 is_loop_end 判断边界
if pypto.is_loop_begin(s_idx):
    # 第一个块: 初始化
    m[:] = m_cur
    l[:] = l_cur
    o[:] = o_cur
else:
    # 后续块: 更新
    m_new = pypto.maximum(m, m_cur)
    ...

if pypto.is_loop_end(s_idx):
    # 最后一个块: 归一化输出
    o_final = pypto.div(o, l)
```

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def flash_attention_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Flash Attention 参考实现"""
    # 使用 PyTorch F.scaled_dot_product_attention
    # 或手动实现标准 attention
    ...
```

### 6.2 测试用例设计

#### 基于 spec.md 典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | scale=None, causal=False | Q:[1,8,1024,128], K:[1,8,1024,128], V:[1,8,1024,128] | [1,8,1024,128] | 性能核心场景 |
| 功能_P0 | 功能 | P0 | scale=None, causal=False | Q:[2,4,512,64], K:[2,4,512,64], V:[2,4,512,64] | [2,4,512,64] | 基础功能验证 |
| 因果注意力_P1 | 功能 | P1 | causal=True | Q:[1,8,512,128], K:[1,8,512,128], V:[1,8,512,128] | [1,8,512,128] | 因果掩码验证 |
| 长序列_P1 | 功能 | P1 | scale=None | Q:[1,4,4096,64], K:[1,4,4096,64], V:[1,4,4096,64] | [1,4,4096,64] | 长序列测试 |
| 掩码注意力_P1 | 功能 | P1 | attn_mask=[1,1,512,512] | Q:[1,8,512,128], K:[1,8,512,128], V:[1,8,512,128] | [1,8,512,128] | 自定义掩码验证 |
| 超长序列_P2 | 功能 | P2 | scale=None | Q:[1,2,8192,64], K:[1,2,8192,64], V:[1,2,8192,64] | [1,2,8192,64] | 超长序列测试 |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 单头注意力 | num_heads=1 | 测试 H=1 的场景 |
| 不等序列长度 | L != S | 测试 query 和 key/value 序列长度不同 |
| 空掩码 | attn_mask=None | 测试不提供 mask 的情况 |
| dtype 转换 | float16/bfloat16 输入 | 测试 dtype 转换 |

### 6.3 精度验证标准

| Dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 0.001 | 0.001 | 高精度 |
| float16 | 0.01 | 0.01 | 混合精度 |
| bfloat16 | 0.01 | 0.01 | 混合精度 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置 (性能类) 的预期性能:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 性能_P0 | 性能 | P0 | scale=None, causal=False | Q:[1,8,1024,128], K:[1,8,1024,128], V:[1,8,1024,128] | [1,8,1024,128] | < 100us |
| 长序列_P1 | 功能 | P1 | scale=None | Q:[1,4,4096,64], K:[1,4,4096,64], V:[1,4,4096,64] | [1,4,4096,64] | < 500us |

### 7.2 开箱性能配置

```python
# Tiling configuration
tile_config = {
    'cube_tile_shapes': [[128, 128], [128, 128], [128, 128]],
    'vec_tile_shapes': [1, 8, 16, 128],
}

# Block sizes for Flash Attention
block_config = {
    'Br': 128,  # Query block size
    'Bc': 512,  # Key/Value block size
    'g_tile': 8,  # Parallel heads
}
```

### 7.3 pass_options 配置

```python
# 参考 GLM-4.5 实现
pass_options = {
    "pg_upper_bound": 1536,
    "cube_l1_reuse_setting": {0: 4}
}
```

### 7.4 runtime_options 配置

```python
runtime_options = {
    "run_mode": pypto.RunMode.NPU,
    "stitch_function_max_num": 128,
}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- **Softmax 仅支持 DT_FP32**: PyPTO softmax 仅支持 FP32，需要在 softmax 前后进行类型转换
- **Transpose 4D 限制**: 4D transpose 只支持 (2,3) 交换，不支持 (0,1) 和 (0,3)
- **动态轴处理**: 动态轴需要使用 pypto.loop，可能影响性能
- **Cube Tiling 必须设置**: 调用 matmul 前必须调用 set_cube_tile_shapes

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| softmax 精度溢出 | FP16/BF16 输入直接 softmax | 精度丢失 | matmul 输出 FP32, softmax 使用 FP32 |
| transpose 轴错误 | 使用不支持的轴交换 | 程序崩溃 | 使用支持的 (2,3) 交换 |
| 动态 shape 处理 | 动态轴超出范围 | 循环边界错误 | 在循环内添加边界检查 |
| mask 类型错误 | bool mask 未转换 | 掩码不生效 | 使用 pypto.where 或 add 转换 |
| 在线 Softmax 数值问题 | max 值更新不正确 | 输出 NaN/Inf | 使用数值稳定的更新公式 |

### 8.3 特殊场景处理

- **is_causal=True 时**: 确保只应用一次因果掩码，避免重复应用
- **attn_mask 与 is_causal 互斥**: 两者不能同时使用，is_causal 优先级更高
- **长序列场景**: 需要调整 tiling 配置，使用更小的分块大小
- **尾块处理**: 需要正确处理最后一个块的大小

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 精度优先 | 推荐使用 matmul FP32 + softmax FP32 以获得更好的精度 |
| 分阶段实现 | 先实现基础版本，再添加 causal mask 和 attn mask 支持 |
| 参考 GLM-4.5 | GLM-4.5 的 Flash Attention 实现是最佳参考 |
| 动态轴处理 | 使用 pypto.loop + pypto.view 组合处理动态轴 |
| 测试覆盖 | 添加多种 shape 和 dtype 的测试用例 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/flash_attention/
├── spec.md                          # 需求规范 (已有)
├── api_report.md                    # API 探索报告 (已有)
├── design.md                        # 设计文档 (本文件)
├── flash_attention_golden.py        # Golden 参考实现 (已有)
├── flash_attention_impl.py          # 算子实现代码 (待生成)
├── test_flash_attention.py          # 测试代码 (待生成)
├── README.md                        # 实现说明 (待生成)
└── .orchestrator_state.json         # 状态文件
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 设计 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design (本 skill) |
| flash_attention_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| flash_attention_impl.py | 代码 | 算子核心实现 | pypto-op-develop (待调用) |
| test_flash_attention.py | 代码 | 测试用例 | pypto-op-develop (待调用) |
| README.md | 文档 | 实现说明 | pypto-op-develop (待调用) |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `flash_attention` |
| 目录名 | 与算子名称一致 | `operators/flash_attention/` |
| Golden 文件 | `{op}_golden.py` | `flash_attention_golden.py` |
| 实现文件 | `{op}_impl.py` | `flash_attention_impl.py` |
| 测试文件 | `test_{op}.py` | `test_flash_attention.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → flash_attention_golden.py → flash_attention_impl.py → test_flash_attention.py → README.md
```

---

*生成时间: 2026-03-29T00:00:00Z*
*状态: Stage 4 完成*
