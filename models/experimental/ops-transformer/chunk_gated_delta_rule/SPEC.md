---
schema_version: 1
op_name: chunk_gated_delta_rule
supported_dtypes: [float16]
p0_shapes:
  - [[1, 2048, 4, 128], [1, 2048, 8, 128], [1, 2048, 8, 128], [1, 2048, 8], null]
  - [[1, 2048, 4, 128], [1, 2048, 8, 128], [1, 2048, 8, 128], [1, 2048, 8], [1, 8, 128, 128]]
tolerance: {rtol: 0.05, atol: 0.05}
dynamic_axes: ['B', 'T', 'N']
dynamic_axes_ranges: {B: [1, 16], T: [64, 4096], N: [1, 32]}
shape_constraints: {H: [1, 64], Hg: 'H/Hg must be integer ratio', K: 128, V: 128, BT: 64}
default_params: {'USE_G': true, 'USE_INITIAL_STATE': true, 'STORE_FINAL_STATE': true, 'SAVE_NEW_VALUE': true, 'chunk_size': 64}
perf_target: '首跑精度成功后进行性能调优'
---

## 算子需求规范

### 1. 基础信息
- **算子名称**: chunk_gated_delta_rule
- **算子分类**: custom (线性注意力机制算子)

### 1.1 功能描述

基于 chunk 的门控 Delta Rule 前向传播算子，用于线性注意力机制（Linear Attention）中的隐藏状态递推计算。支持定长序列和变长序列两种模式。

核心计算流程：
1. **残差计算**: v_new = v - w @ h (GEMM)
2. **门控缩放**: v_new = v_new * exp(g_last - g) (可选)
3. **状态衰减**: h = h * exp(g_last) (可选)
4. **状态更新**: h = h + k.T @ v_new (GEMM)

### 1.2 算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| chunk_size (BT) | 64 | chunk 分块大小，固定 |
| USE_G | true | 是否启用门控机制 |
| USE_INITIAL_STATE | true | 是否使用初始状态 h0 |
| STORE_FINAL_STATE | true | 是否输出最终状态 ht |
| SAVE_NEW_VALUE | true | 是否输出 v_new |

### 1.3 数学公式

对于每个 chunk t (t = 0, 1, ..., NT-1):
$$
\begin{align}
h[t] &= h[t-1] \text{ (累积状态，初始为 h0 或零)} \\
v_{new}[t] &= v[t] - w[t] \cdot h[t] \text{ (残差计算)} \\
\text{若 USE\_G:} \\
&\quad g_{last} = g[t_{valid} - 1] \text{ (chunk 内最后一个有效 token 的 gate)} \\
&\quad v_{new}[t] = v_{new}[t] \cdot \exp(g_{last} - g[t]) \text{ (门控缩放)} \\
&\quad h[t] = h[t] \cdot \exp(g_{last}) \text{ (状态衰减)} \\
h[t+1] &= h[t] + k[t]^T \cdot v_{new}[t] \text{ (状态更新)}
\end{align}
$$

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| 分块策略 (chunk_size=64) | ✓ 需要 | ✓ 高 | 固定 BT=64，chunk 间串行计算 | P0 |
| 动态 Shape / 变长序列 | ✓ 需要 | ✓ 高 | 支持 cu_seqlens 变长序列，动态计算 NT | P0 |
| 门控机制 (USE_G) | ✓ 需要 | ✓ 高 | exp(g_last - g) 门控缩放 + h 状态衰减 | P1 |
| 状态累积 | ✓ 需要 | ✓ 高 | h_state 在 chunk 间累积传递 | P0 |
| 精度控制 (float32 计算) | ✓ 需要 | ✓ 高 | v_new 计算使用 float32 精度避免累积误差 | P0 |
| 初始状态支持 (h0) | ✓ 需要 | ✓ 高 | 支持 USE_INITIAL_STATE 参数 | P1 |
| 最终状态输出 (ht) | ✓ 需要 | ✓ 高 | 支持 STORE_FINAL_STATE 参数 | P1 |
| v_new 输出 | ✓ 需要 | ✓ 高 | 支持 SAVE_NEW_VALUE 参数 | P1 |

### 3. 算法描述

```
Algorithm: chunk_gated_delta_rule Forward
────────────────────────────────────────────────────────────
输入: k, w, v, g (可选), h0 (可选), cu_seqlens (变长模式)
输出: h (每个 chunk 的状态), v_new, ht (可选)

参数: BT=64 (chunk size), USE_G, USE_INITIAL_STATE, STORE_FINAL_STATE

1. Grid 划分: N * H blocks (每个 batch-head 组合一个 block)
   - 定长模式: N = B
   - 变长模式: N = 序列数 (len(cu_seqlens) - 1)

2. V 分块: V // 2 = 64 (由 vid 控制，两个 Vector sub-block 并行)

3. 初始化状态:
   if USE_INITIAL_STATE:
       h_state = h0[i_n, i_h, K//2*vid : K//2*vid + K//2, :]
   else:
       h_state = zeros([K//2, V])

4. 主循环: for i in range(NT_max):
   if i < NT_i:  # 动态长度掩码
       g_start = bos + i * BT

       4.1 w @ h (GEMM on Cube):
           load w[g_start:g_start+BT] -> w_chunk_l1
           load h_state -> h_state_l1
           wh_frag = gemm_v0(w_chunk_l1, h_state_l1)
           store wh_frag -> ws_wh (float32)

       4.2 v_new = v - w @ h (Vector, float32):
           load v[g_start + BT//2*vid : ...] -> v_chunk_ub
           convert v_chunk_ub -> v_chunk_ub_float (float32)
           load ws_wh -> wh_ub_float (float32)
           v_new_ub_float = v_chunk_ub_float - wh_ub_float

       4.3 门控计算 (可选, USE_G=True):
           load g[g_start:g_start+BT] -> g_chunk_ub_all
           g_last = g_chunk_ub_all[BT-1] or g_chunk_ub_all[T_len - i*BT - 1]
           g_exp = exp(g_last - g_chunk)
           v_new_ub_float = v_new_ub_float * broadcast(g_exp)
           h_state_ub_float = h_state_ub_float * exp(g_last)

       4.4 k @ v_new (GEMM on Cube):
           convert v_new_ub_float -> v_new_ub (float16)
           load k[g_start:g_start+BT] -> k_chunk_l1
           hupd_frag = gemm_v0(k_chunk_l1, v_new_l1, transpose_A=True)
           store hupd_frag -> ws_hupd

       4.5 状态累积 (Vector):
           load ws_hupd -> hupd_ub_float
           h_state_ub_float = h_state_ub_float + hupd_ub_float
           convert h_state_ub_float -> h_state_ub (float16)
           store h_state_ub -> h[i_n, i, i_h, K//2*vid : ...]

5. Epilogue: if STORE_FINAL_STATE:
   store h_state_ub -> ht[i_n, i_h, K//2*vid : ...]
```

### 4. 数据流图

```
输入张量:
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐
│  k [T, Hg, K]    │  │  w [T, H, K]     │  │  v [T, H, V]     │  │ g [T, H]     │
│   float16        │  │   float16        │  │   float16        │  │  float32     │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘
         │                     │                     │                   │
         │                     │                     │                   │
         ▼                     ▼                     ▼                   ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                    Chunk Loop (for each chunk i)                        │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Step 1: w @ h (GEMM on Cube)                                    │   │
    │  │   w_chunk_l1 [BT, K] × h_state_l1 [K, V] -> wh_frag [BT, V]     │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                         │
    │                              ▼ ws_wh (float32)                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Step 2: v_new = v - w @ h (Vector, float32)                     │   │
    │  │   v_chunk_ub_float - wh_ub_float -> v_new_ub_float              │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                         │
    │                              ▼                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Step 3: 门控计算 (可选, USE_G=True)                              │   │
    │  │   v_new *= exp(g_last - g_chunk)                                │   │
    │  │   h_state *= exp(g_last)                                        │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                         │
    │                              ▼                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Step 4: k.T @ v_new (GEMM on Cube)                              │   │
    │  │   k_chunk_l1.T [K, BT] × v_new_l1 [BT, V] -> hupd_frag [K, V]   │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                         │
    │                              ▼                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Step 5: 状态累积 (Vector)                                       │   │
    │  │   h_state += hupd_ub                                           │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                         │
    │                              ▼                                         │
    │                     store h_state -> h[i]                              │
    └────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                         输出张量                                        │
    │  ┌───────────────┐  ┌───────────────┐  ┌────────────────────┐          │
    │  │ h [NT, H, K,V]│  │v_new [T, H, V]│  │ ht [H, K, V] (可选)│          │
    │  │   float16     │  │   float16     │  │    float16         │          │
    │  └───────────────┘  └───────────────┘  └────────────────────┘          │
    └───────────────────────────────────────────────────────────────────────┘
```

### 5. 输入输出规格

**输入规格**:

| 变量 | Shape (定长) | Shape (变长) | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------------|-------------|-------|--------|--------|------|
| k | [B, T, Hg, K] | [1, T_total, Hg, K] | float16 | B, T | ✓ 高 | Key 向量，GQA 模式 |
| w | [B, T, H, K] | [1, T_total, H, K] | float16 | B, T | ✓ 高 | 门控权重 |
| v (u) | [B, T, H, V] | [1, T_total, H, V] | float16 | B, T | ✓ 高 | Value 向量 |
| g | [B, T, H] | [1, T_total, H] | float32 | B, T | ✓ 高 | 门控向量 (可选) |
| h0 | [B, H, K, V] | [1, N, H, K, V] | float16 | B, N | ✓ 高 | 初始状态 (可选) |
| cu_seqlens | - | [N+1] | int32 | N | ✓ 高 | 变长序列边界 (变长模式) |

**输出规格**:

| 变量 | Shape (定长) | Shape (变长) | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------------|-------------|-------|--------|--------|------|
| h | [B, NT, H, K, V] | [1, NT_total, H, K, V] | float16 | B, NT | ✓ 高 | 每个 chunk 的隐藏状态 |
| v_new | [B, T, H, V] | [1, T_total, H, V] | float16 | B, T | ✓ 高 | 更新后的 value |
| ht | [B, H, K, V] | [1, N, H, K, V] | float16 | B | ✓ 高 | 最终隐藏状态 (可选) |

**Shape 约束**:
- K = 128 (固定)
- V = 128 (固定)
- BT = 64 (chunk size 固定)
- Hg ≤ H (GQA 比例必须满足)
- H / Hg 必须为整数
- NT = ceildiv(T, BT)

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float16 | ✓ | 0.05 | 0.05 | 输入/输出 dtype |
| float32 | ✓ | 0.001 | 0.001 | 计算精度 dtype (ws_wh, 门控计算) |

### 7. 精度要求
- **atol**: 0.05 (T ≤ 2048)
- **rtol**: 0.05 (T ≤ 2048)
- **精度容忍度分级**:
  - T ≤ 128: rtol=0.01, atol=0.001
  - 128 < T ≤ 512: rtol=0.01, atol=0.005
  - 512 < T ≤ 2048: rtol=0.05, atol=0.05

### 8. 动态轴说明
- **动态轴**: ['B', 'T', 'N']
- **轴含义**:
  - B: Batch size (定长模式)
  - T: Sequence length (总序列长度)
  - N: 序列数 (变长模式)
- **取值范围**:
  - B: [1, 16]
  - T: [64, 4096]
  - N: [1, 32]
- **静态维度**:
  - H: [1, 64] (head 数)
  - Hg: [1, 32] (key head 数，需满足 H/Hg 为整数)
  - K: 128 (固定)
  - V: 128 (固定)
  - BT: 64 (固定 chunk size)

### 9. 边界条件处理
- **零值**: normal (正常计算)
- **极值**: 数值稳定处理 - exp(g_last - g) 使用 clamp 避免 overflow
- **NaN/Inf**: normal (正常计算)

### 10. 性能要求
- **性能目标**: 首跑精度成功后进行性能调优，目标达到 tilelang 实现的性能水平

### 11. 参考信息
- **参考实现**: /data/x00952168/pypto/chunk_gated_delta_rule/chunk_gated_delta_rule_varlen.py (tilelang)
- **设计文档**: /data/x00952168/pypto/chunk_gated_delta_rule/design.md
- **类似算子**: Flash Attention, Linear Attention, Delta Rule

### 12. 应用场景
- **目标模型**: 线性注意力模型 (Linear Attention, RWKV 等)
- **使用位置**: Transformer 层中的隐藏状态递推计算

**典型配置** (测试规格与 tilelang 脚本一致):

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| Fixed_P0 | 功能 | P0 | use_g=True, use_initial_state=True, B=1, T=2048 | k[1,2048,4,128], w[1,2048,8,128], v[1,2048,8,128], g[1,2048,8], h0[1,8,128,128] | h[1,32,8,128,128], v_new[1,2048,8,128], ht[1,8,128,128] | 核心功能验证 |
| Fixed_P0_no_g | 功能 | P0 | use_g=False, use_initial_state=True, B=1, T=2048 | k[1,2048,4,128], w[1,2048,8,128], v[1,2048,8,128], h0[1,8,128,128] | h[1,32,8,128,128], v_new[1,2048,8,128], ht[1,8,128,128] | 无门控模式 |
| Fixed_P0_no_h0 | 功能 | P0 | use_g=True, use_initial_state=False, B=1, T=2048 | k[1,2048,4,128], w[1,2048,8,128], v[1,2048,8,128], g[1,2048,8] | h[1,32,8,128,128], v_new[1,2048,8,128], ht[1,8,128,128] | 无初始状态 |
| Fixed_P0_no_both | 功能 | P0 | use_g=False, use_initial_state=False, B=1, T=2048 | k[1,2048,4,128], w[1,2048,8,128], v[1,2048,8,128] | h[1,32,8,128,128], v_new[1,2048,8,128], ht[1,8,128,128] | 最简配置 |
| Varlen_P1_4x512 | 功能 | P1 | use_g=True/False, seqlens=[512,512,512,512] | k[1,2048,4,128], w[1,2048,8,128], v[1,2048,8,128] | h[1,32,8,128,128], v_new[1,2048,8,128] | 4 等长序列 |
| Varlen_P1_mixed | 功能 | P1 | use_g=True/False, seqlens=[128,256,512,1024,128] | k[1,2048,4,128], w[1,2048,8,128], v[1,2048,8,128] | h[1,~,8,128,128], v_new[1,2048,8,128] | 变长序列混合 |
| Varlen_P1_single | 功能 | P1 | use_g=True/False, seqlens=[2048] | k[1,2048,4,128], w[1,2048,8,128], v[1,2048,8,128] | h[1,32,8,128,128], v_new[1,2048,8,128] | 单序列变长模式 |
| Varlen_P1_2x1024 | 功能 | P1 | use_g=True/False, seqlens=[1024,1024] | k[1,2048,4,128], w[1,2048,8,128], v[1,2048,8,128] | h[1,32,8,128,128], v_new[1,2048,8,128] | 2 等长序列 |

---
*生成时间: 2026-04-21*
*确认状态: 已确认 (基于 tilelang 设计文档)*
*置信度说明: ✓ 高 (基于完整 tilelang 实现和设计文档)*