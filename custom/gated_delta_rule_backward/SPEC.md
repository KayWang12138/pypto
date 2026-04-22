---
schema_version: 1
op_name: gated_delta_rule_backward
supported_dtypes: [float32]
p0_shapes:
  - [1, 128, 4, 128, 128, 64, 2]
  - [1, 4096, 4, 128, 128, 128, 64]
tolerance: {atol: 0.001, rtol: 0.001}
dynamic_axes: ['B', 'T']
dynamic_axes_ranges: {B: [1, 64], T: [64, 8192]}
shape_constraints:
  - T % BT == 0
  - NT = T / BT
  - K == V (not required, but typical)
default_params: {scale: '1/sqrt(K)', l2_eps: 1.0e-6, use_qk_l2norm_in_kernel: true}
perf_target: 首跑精度成功性能的 2 倍
---

## 算子需求规范

### 1. 基础信息
- **算子名称**: gated_delta_rule_backward
- **算子分类**: custom（反向传播算子，分块循环 + 状态递推 + WY 低秩表示）

### 1.1 功能描述

Gated Delta Rule 的反向传播算子。给定前向计算的输出梯度（`do`, `dht`）和前向缓存（`A`, `w`, `S_before`, `v_new`, `q_norm`, `k_norm`, `q_rstd`, `k_rstd`），反向传播计算所有输入的梯度（`dq`, `dk`, `dv`, `db`, `dg_raw`, `dh0`）。

该算子是 Gated Delta Rule 前向算子的精确反向，采用从最后一个 chunk 到第一个 chunk 的逆序循环（时间步反向传播），在每个 chunk 内部通过 WY 低秩表示分解梯度计算。

### 1.2 算法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scale` | float | `1/sqrt(K)` | 注意力缩放因子 |
| `BT` | int | 用户指定 | 时间维度分块大小 |
| `use_qk_l2norm_in_kernel` | bool | `true` | 是否使用融合 L2 归一化（前向已对 q,k 做 L2 norm） |
| `l2_eps` | float | `1e-6` | L2 归一化的 epsilon |

### 1.3 数学公式

反向传播的核心是对前向计算的每个操作求导。主要计算步骤：

1. **局部注意力梯度**: `dv0 = (A_local^T @ do) * scale`，其中 `A_local = (qk^T * decay) * m_le`
2. **状态递推反向传播**（逆序）: `d_s = d_s_next * exp(g_last) + q_eff^T @ do * scale - w^T @ dv_total`
3. **q/k/g 梯度**: 局部注意力贡献 + 状态贡献
4. **WY 表示更新**: 通过 `A` 矩阵反推 `du`, `dw`，再分解为 `dv`, `dk`, `db`, `dg`
5. **L2 norm 反向**（可选）: 当 `use_qk_l2norm_in_kernel=True` 时，对 dq/dk 施加 L2 norm 的反向传播

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| 分块逆序循环 | ✓ 需要 | ✓ 高 | 从 chunk NT-1 到 0 逆序遍历，维护状态梯度 d_s | P0 |
| WY 低秩表示 | ✓ 需要 | ✓ 高 | 使用前向 A 矩阵分解 du/dw，反推 dv/dk/db/dg | P0 |
| 融合 L2 norm 反向 | ✓ 需要 | ✓ 高 | 当 use_qk_l2norm_in_kernel=True 时，对 dq/dk 施加 L2 norm 反向 | P0 |
| 指数衰减计算 | ✓ 需要 | ✓ 高 | g_cum 和 decay 的指数运算，涉及数值稳定性 | P0 |
| 常量矩阵 | ✓ 需要 | ✓ 高 | i_mat, m_le, m_lt, c_cum, c_rcum 为 host 端常量 | P0 |
| 前向缓存消费 | ✓ 需要 | ✓ 高 | A, w, S_before, v_new, q_norm, k_norm, q_rstd, k_rstd | P0 |
| fp32 全精度 | ✓ 需要 | ✓ 高 | 全部输入输出和中间计算均为 fp32 | P0 |

### 3. 算法描述

```
Algorithm: Gated Delta Rule Backward
────────────────────────────────────
输入: q[B,T,H,K], k[B,T,H,K], v[B,T,H,V], g_raw[B,T,H], beta[B,T,H],
      initial_state[B,H,K,V], do[B,T,H,V], dht[B,H,K,V],
      前向缓存: A[B,H,NT,BT,BT], w[B,H,NT,BT,K], S_before[B,H,NT,K,V],
               v_new[B,H,NT,BT,V], q_norm[B,T,H,K], k_norm[B,T,H,K],
               q_rstd[B,T,H], k_rstd[B,T,H]
      常量矩阵: i_mat[BT,BT], m_le[BT,BT], m_lt[BT,BT], c_cum[BT,BT], c_rcum[BT,BT]
输出: dq, dk[B,T,H,K], dv[B,T,H,V], db, dg_raw[B,T,H], dh0[B,H,K,V]

1. 初始化: dq=0, dk=0, dv=0, db=0, dg_raw=0
2. 对每个 (b, h):
3.   d_s = dht[b, h]
4.   for i = 0 to NT-1:   // 逆序遍历 chunk
5.     c = NT - 1 - i
6.     t0 = c * BT, t1 = t0 + BT
7.     提取 chunk 输入: qc, kc, vc, betac, gc_raw, doc
8.     提取 chunk 缓存: a, w, s_before, v_new
9.     
10.    // 计算 g 和衰减
11.    g_cum = c_cum @ gc_raw
12.    eg = exp(g_cum), gl = g_cum[-1]
13.    decay = exp(g_cum[:,None] - g_cum[None,:])
14.    
15.    // 局部注意力 dv0
16.    qk = qc @ kc^T
17.    a_local = (qk * decay) * m_le
18.    dv0 = (a_local^T @ doc) * scale
19.    
20.    // 状态递推反向传播
21.    s_tok = exp(gl - g_cum)
22.    dv_state = (kc @ d_s) * s_tok
23.    dv_total = dv_state + dv0
24.    q_eff = qc * eg
25.    d_s = d_s * exp(gl) + q_eff^T @ doc * scale - w^T @ dv_total
26.    
27.    // q/k/g 梯度（局部注意力 + 状态贡献）
28.    dq_c += (doc @ s_before^T) * eg * scale
29.    dk_c += v_new * s_tok @ d_s_next^T       // 状态贡献
30.    dg_cum += ... (多种贡献项)
31.    
32.    // WY 低秩表示分解
33.    dw = -(dv_total @ s_before^T)
34.    du = dv_total
35.    dvb = A^T @ du
36.    dkbg = A^T @ dw
37.    // 分解为 dv, dk, db, dg 增量
38.    
39.    // 聚合并反归一化
40.    dg_raw_c = c_rcum @ dg_cum
41.    if use_qk_l2norm_in_kernel:
42.      dq_raw_c = l2norm_bwd(q_norm_c, q_rstd_c, dq_c)
43.      dk_raw_c = l2norm_bwd(k_norm_c, k_rstd_c, dk_c)
44.    
45.    写入 dq, dk, dv, db, dg_raw 的对应 chunk 位置
46.  dh0[b, h] = d_s
```

### 4. 数据流图

```
                           ┌──────────────────────────────────────────────┐
                           │           前向缓存 (来自 forward)              │
                           │  A[B,H,NT,BT,BT], w[B,H,NT,BT,K]           │
                           │  S_before[B,H,NT,K,V], v_new[B,H,NT,BT,V]  │
                           │  q_norm, k_norm, q_rstd, k_rstd             │
                           └────────────┬─────────────────────────────────┘
                                        │
     ┌──────────────────────────────────┼───────────────────────────────────┐
     │                                  │                                   │
     ▼                                  ▼                                   ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│q[B,T,H,K]│  │k[B,T,H,K]│  │v[B,T,H,V]│  │do[B,T,H,V│  │dht[B,H,K,│  │常量矩阵   │
│          │  │          │  │          │  │]         │  │V]        │  │[BT,BT]   │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │              │             │             │
     │    ┌────────┘             │              │             │             │
     │    │                      │              │             │             │
     ▼    ▼                      ▼              ▼             ▼             ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                逆序 Chunk 循环 (c = NT-1 ... 0)                              │
  │                                                                             │
  │  ┌─────────────────────────────────────────────────────────────────────────┐│
  │  │ Step 1: 计算 g_cum, eg, decay                                          ││
  │  │ Step 2: 局部注意力 dv0 = (A_local^T @ doc) * scale                     ││
  │  │ Step 3: 状态递推 d_s 反向传播                                            ││
  │  │ Step 4: dq/dk/dg_cum 累积（局部 + 状态贡献）                             ││
  │  │ Step 5: WY 分解 du/dw → dv/dk/db/dg                                    ││
  │  │ Step 6: dg_raw = c_rcum @ dg_cum                                       ││
  │  │ Step 7: L2 norm 反向（可选）                                             ││
  │  └─────────────────────────────────────────────────────────────────────────┘│
  └──────────────────────────────────┬──────────────────────────────────────────┘
                                     │
           ┌────────────┬────────────┼────────────┬────────────┐
           ▼            ▼            ▼            ▼            ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
     │dq[B,T,H, │ │dk[B,T,H, │ │dv[B,T,H, │ │db[B,T,H] │ │dg_raw[B, │
     │K]        │ │K]        │ │V]        │ │          │ │T,H]      │
     └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
                                                      ┌──────────┐
                                                      │dh0[B,H,K,│
                                                      │V]        │
                                                      └──────────┘
```

### 5. 输入输出规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| `q` | `[B, T, H, K]` | float32 | B, T | ✓ 高 | 原始 query（pre-L2-norm） |
| `k` | `[B, T, H, K]` | float32 | B, T | ✓ 高 | 原始 key（pre-L2-norm） |
| `v` | `[B, T, H, V]` | float32 | B, T | ✓ 高 | value |
| `g_raw` | `[B, T, H]` | float32 | B, T | ✓ 高 | 门控原始值 |
| `beta` | `[B, T, H]` | float32 | B, T | ✓ 高 | beta 参数 |
| `initial_state` | `[B, H, K, V]` | float32 | B | ✓ 高 | 初始隐状态 |
| `do` | `[B, T, H, V]` | float32 | B, T | ✓ 高 | dL/d(out)，输出梯度 |
| `dht` | `[B, H, K, V]` | float32 | B | ✓ 高 | dL/d(final_state)，终态梯度 |
| `A` | `[B, H, NT, BT, BT]` | float32 | B | ✓ 高 | 前向缓存：WY 表示矩阵 |
| `w` | `[B, H, NT, BT, K]` | float32 | B | ✓ 高 | 前向缓存：W 的值 |
| `S_before` | `[B, H, NT, K, V]` | float32 | B | ✓ 高 | 前向缓存：chunk 前状态 |
| `v_new` | `[B, H, NT, BT, V]` | float32 | B | ✓ 高 | 前向缓存：更新后的 v |
| `q_norm` | `[B, T, H, K]` | float32 | B, T | ✓ 高 | L2 归一化后的 q（仅 use_qk_l2norm_in_kernel=True） |
| `k_norm` | `[B, T, H, K]` | float32 | B, T | ✓ 高 | L2 归一化后的 k（仅 use_qk_l2norm_in_kernel=True） |
| `q_rstd` | `[B, T, H]` | float32 | B, T | ✓ 高 | q 的 L2 rstd（仅 use_qk_l2norm_in_kernel=True） |
| `k_rstd` | `[B, T, H]` | float32 | B, T | ✓ 高 | k 的 L2 rstd（仅 use_qk_l2norm_in_kernel=True） |
| `i_mat` | `[BT, BT]` | float32 | - | ✓ 高 | 单位矩阵（host 端常量） |
| `m_le` | `[BT, BT]` | float32 | - | ✓ 高 | 下三角含对角线矩阵（host 端常量） |
| `m_lt` | `[BT, BT]` | float32 | - | ✓ 高 | 严格下三角矩阵（host 端常量） |
| `c_cum` | `[BT, BT]` | float32 | - | ✓ 高 | 下三角累加矩阵（host 端常量） |
| `c_rcum` | `[BT, BT]` | float32 | - | ✓ 高 | 上三角累加矩阵（host 端常量） |
| `scale` | scalar | float32 | - | ✓ 高 | `1/sqrt(K)` |
| `BT` | scalar | int | - | ✓ 高 | 分块大小 |
| `use_qk_l2norm_in_kernel` | scalar | bool | - | ✓ 高 | 是否启用 L2 归一化（当前版本 = True） |
| `l2_eps` | scalar | float32 | - | ✓ 高 | L2 归一化 epsilon（默认 1e-6） |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| `dq` | `[B, T, H, K]` | float32 | B, T | ✓ 高 | dL/d(q) |
| `dk` | `[B, T, H, K]` | float32 | B, T | ✓ 高 | dL/d(k) |
| `dv` | `[B, T, H, V]` | float32 | B, T | ✓ 高 | dL/d(v) |
| `db` | `[B, T, H]` | float32 | B, T | ✓ 高 | dL/d(beta) |
| `dg_raw` | `[B, T, H]` | float32 | B, T | ✓ 高 | dL/d(g_raw) |
| `dh0` | `[B, H, K, V]` | float32 | B | ✓ 高 | dL/d(initial_state) |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 唯一支持的数据类型 |

### 7. 精度要求
- **atol**: 1e-3
- **rtol**: 1e-3
- **累积方式**: fp32 全精度累积
- **验证方式**: `detailed_tensor_compare` 逐张量对比

### 8. 动态轴说明
- **动态轴**: B, T
- **轴含义**: B = batch size, T = sequence length
- **取值范围**: B ∈ [1, 64], T ∈ [64, 8192]
- **约束**: T % BT == 0, NT = T / BT

### 9. 边界条件处理
- **零值**: 正常计算
- **极值**: 正常计算（指数运算可能产生极值，需注意数值稳定性）
- **NaN/Inf**: 正常计算

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的 2 倍

### 11. 参考信息
- **参考实现**: `models/qwen3_next/gated_delta_rule_golden.py` 中的 `torch_golden_gated_delta_rule_backward_ref`
- **论文**: Gated Delta Rule（DeltaNet 变体）
- **类似算子**: DeltaNet backward, RWKV backward

### 12. 应用场景
- **目标模型**: Qwen3-Next
- **使用位置**: Gated Delta Rule attention 层的反向传播

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| Small | 功能 | P0 | B=1,T=128,H=4,K=128,V=128,BT=64 | q/k:[1,128,4,128], v:[1,128,4,128] | dq/dk:[1,128,4,128], dv:[1,128,4,128] | 小规模功能验证 |
| Large | 性能 | P0 | B=1,T=4096,H=4,K=128,V=128,BT=128 | q/k:[1,4096,4,128], v:[1,4096,4,128] | dq/dk:[1,4096,4,128], dv:[1,4096,4,128] | 大规模性能验证 |

---
*生成时间: 2026-04-21*
*确认状态: 已确认（用户提供了完整规格，直接生成）*
*置信度说明: ✓ 高（用户提供了完整的 golden reference 和张量规格）*
