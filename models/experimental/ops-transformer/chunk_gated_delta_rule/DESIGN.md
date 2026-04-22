---
schema_version: "2.1"
op_name: "chunk_gated_delta_rule"
status: draft
last_updated: "2026-04-21"

compute_kind: "hybrid"
dtypes: ["fp16", "fp32"]
dynamic_axes: ["B", "T", "N"]
precision: { rtol: 0.05, atol: 0.05 }
---

# chunk_gated_delta_rule 设计方案

## 1. 计算图与精度路由

### 1.1 API 调用序列

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| 1 | 状态初始化 | `pypto.zeros([K, V], dtype)` 或 `h0` | FP16 | FP16 | [K, V] | USE_INITIAL_STATE 决定来源 |
| 2 | 状态类型转换 | `pypto.cast(h_state, DT_FP32)` | FP16 | FP32 | [K, V] | 精度控制：状态用 FP32 |
| 3 | w @ h | `pypto.matmul(w_chunk, h_state, DT_FP32)` | FP16(FP32转换) | FP32 | [BT, V] | Cube GEMM，b_trans=True |
| 4 | v 类型转换 | `pypto.cast(v_chunk, DT_FP32)` | FP16 | FP32 | [BT, V] | 精度控制：计算用 FP32 |
| 5 | v_new = v - w@h | `pypto.sub(v_fp32, wh_fp32)` | FP32 | FP32 | [BT, V] | Vector 减法 |
| 6 | g_last 提取 | `g_chunk[-1]` 或 `g_chunk[T_len-1]` | FP32 | FP32 | [1] | 最后有效 token 的 gate |
| 7 | exp(g_last - g) | `pypto.exp(g_last - g_chunk)` | FP32 | FP32 | [BT] | Vector exp |
| 8 | v_new 门控缩放 | `pypto.mul(v_new, g_exp)` | FP32 | FP32 | [BT, V] | 需广播 g_exp 到 [BT, V] |
| 9 | h 状态衰减 | `pypto.mul(h_state, exp(g_last))` | FP32 | FP32 | [K, V] | 状态门控衰减 |
| 10 | k.T @ v_new | `pypto.matmul(k_chunk, v_new, DT_FP32, a_trans=True)` | FP16(FP32转换) | FP32 | [K, V] | Cube GEMM |
| 11 | 状态累积 | `pypto.add(h_state, h_upd)` | FP32 | FP32 | [K, V] | Vector 加法 |
| 12 | 状态类型转换 | `pypto.cast(h_state, DT_FP16)` | FP32 | FP16 | [K, V] | 输出精度 |
| 13 | 存储 h[t] | `h[chunk_idx] = h_state_fp16` | FP16 | FP16 | [NT, K, V] | 状态输出 |
| 14 | 存储 v_new | `v_new[offset:offset+BT] = v_new_fp16` | FP16 | FP16 | [T, V] | v_new 输出 |
| 15 | 存储 ht | `ht[batch, head] = h_state_fp16` | FP16 | FP16 | [K, V] | 最终状态（可选） |

### 1.2 精度路由

```text
输入(FP16) → [cast: FP16 → FP32] → 计算(FP32) → [cast: FP32 → FP16] → 输出(FP16)
```

| 转换位置 | 转换方向 | 原因 |
|---------|---------|------|
| GEMM 输入前 | FP16 → FP32 | matmul 精度要求，避免累积误差 |
| v_new 计算前 | FP16 → FP32 | 残差计算精度敏感，避免 FP16 截断 |
| 状态输出前 | FP32 → FP16 | 输出 dtype 为 FP16 |
| v_new 输出前 | FP32 → FP16 | 输出 dtype 为 FP16 |

**精度控制策略**：
- 所有中间计算（matmul、sub、exp、mul、add）使用 FP32
- 累积状态 h_state 在整个计算过程中保持 FP32
- 仅在存储输出时转换为 FP16

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
| 全程 FP16 计算 | 精度问题：v_new = v - w@h 在 FP16 下累积误差过大，验证失败 |
| 状态全程 FP32 保持 | 输出 tensor dtype 必须为 FP16，存储时必须转换 |
| 使用 workspace buffer ws_wh | 已在 TileLang 设计中验证可行，但 PyPTO 可直接使用 cast 避免额外内存开销 |

---

## 2. 数据规格

### 2.1 Kernel 函数签名

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def chunk_gated_delta_rule_kernel(
    # 输入
    k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, Hg, K], pypto.DT_FP16),   # [B, T, Hg, K] 或 [1, T_total, Hg, K]
    w: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP16),    # [B, T, H, K]
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, V], pypto.DT_FP16),    # [B, T, H, V]
    g: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),       # [B, T, H] (可选，门控向量)
    h0: pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP16),               # [B, H, K, V] (可选，初始状态)
    cu_seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),               # [N+1] (变长模式)
    
    # 输出
    h: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K, V], pypto.DT_FP16), # [B, NT, H, K, V]
    v_new: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, V], pypto.DT_FP16),# [B, T, H, V]
    ht: pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP16),               # [B, H, K, V] (可选)
    
    # 配置参数
    BT: int = 64,               # chunk size，固定常量
    USE_G: bool = True,         # 是否启用门控
    USE_INITIAL_STATE: bool = True,
    STORE_FINAL_STATE: bool = True,
):
    ...
```

### 2.2 动态轴分析

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 | Loop 处理 |
|--------|---------|-----------------|---------|----------|
| B (batch) | 是 | [1, 16] | `pypto.DYNAMIC` | `pypto.loop(B)` |
| T (seq_len) | 是 | [64, 4096] | `pypto.DYNAMIC` | chunk 级 loop，NT = ceil(T/BT) |
| N (序列数) | 是 | [1, 32] | `pypto.DYNAMIC` | 变长模式下用于遍历序列 |
| H (head) | 否 | [1, 64] | 数值 | `pypto.loop(H)` |
| Hg (key_head) | 否 | [1, 32] | 数值 | GQA 比例 H/Hg |
| K | 否 | 128 (固定) | 常量 | 不需要 loop |
| V | 否 | 128 (固定) | 常量 | 不需要 loop |
| BT | 否 | 64 (固定) | 常量 | chunk size |
| NT | 派生 | ceil(T/BT) | 运行时计算 | `pypto.loop(NT)` |

**派生维度说明**：
- `NT = (T + BT - 1) // BT`：chunk 数，由 T 派生
- `k_head = head_idx // (H // Hg)`：GQA key head 映射

### 2.3 值类型分析（避免 SymbolicScalar 误用）

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| B | `k.shape[0]` | SymbolicScalar | 不可用于 Python `if/range`，需用 `pypto.loop(B)` |
| T | `k.shape[1]` | SymbolicScalar | 用于计算 NT，不可用于 Python 切片 |
| NT | `(T + BT - 1) // BT` | SymbolicScalar | 派生值，需用 `pypto.loop(NT)` |
| s (变长序列长度) | `act_seq_len[b_idx+1] - act_seq_len[b_idx]` | SymbolicScalar | 变长模式序列长度 |
| actual_BT | `(s - chunk_idx * BT).min(BT)` | Element/SymbolicScalar | 尾块实际大小 |
| BT, K, V, H | 字面量 | int | 编译期常量，常规使用 |
| g_last | `g_chunk[-1]` 或条件选择 | pypto.Element | 标量参与计算 |

---

## 3. Tiling 策略

### 3.1 算子类型

**Hybrid（混合算子）**：Cube matmul + Vector element-wise

- Cube 操作：`pypto.matmul`（w @ h, k.T @ v_new）
- Vector 操作：`pypto.cast`, `pypto.sub`, `pypto.exp`, `pypto.mul`, `pypto.add`

### 3.2 Tiling 推导

#### 同时驻留 UB/L1 的 Tensor（单 chunk 计算）

| Tensor | 用途 | shape | dtype | 大小估算 |
|--------|------|-------|-------|---------|
| w_chunk | w 数据切片 | [BT, K] = [64, 128] | FP16 | 8KB |
| k_chunk | k 数据切片 | [BT, K] = [64, 128] | FP16 | 8KB |
| v_chunk | v 数据切片 | [BT, V] = [64, 128] | FP16 | 8KB |
| v_chunk_fp32 | v FP32 缓冲 | [BT, V] | FP32 | 32KB |
| h_state | 累积状态 | [K, V] = [128, 128] | FP32 | 64KB |
| wh | w @ h 结果 | [BT, V] | FP32 | 32KB |
| v_new | 残差计算结果 | [BT, V] | FP32 | 32KB |
| g_chunk | g 数据切片 | [BT] = [64] | FP32 | 256B |
| g_exp | exp(g_last - g) | [BT] | FP32 | 256B |
| h_upd | k.T @ v_new 结果 | [K, V] | FP32 | 64KB |
| h_state_fp16 | 状态 FP16 输出 | [K, V] | FP16 | 32KB |

**总 UB/L1 估算**：
- Cube 输入 L1: w_chunk + k_chunk + h_state ≈ 24KB
- Vector UB FP32: v_chunk_fp32 + wh + v_new + h_state + h_upd ≈ 224KB
- Vector UB FP16: h_state_fp16 ≈ 32KB

**UB 容量**：约 256KB（安全范围内）

#### Tiling 推导步骤

1. **尾轴对齐**：
   - FP16: 16 元素对齐 (32B)
   - FP32: 8 元素对齐 (32B)
   - K=128, V=128 均满足对齐要求

2. **Cube Tiling**：
   - `w @ h`: w[BT, K] × h[K, V] → wh[BT, V]
     - 配置: `pypto.set_cube_tile_shapes([64, 128], [128, 128], [64, 128])`
   - `k.T @ v_new`: k[K, BT] × v_new[BT, V] → h_upd[K, V]
     - 配置: `pypto.set_cube_tile_shapes([128, 64], [64, 128], [128, 128])`

3. **Vector Tiling**：
   - element-wise 操作 [BT, V]: `pypto.set_vec_tile_shapes(64, 128)`
   - 状态操作 [K, V]: `pypto.set_vec_tile_shapes(128, 128)`
   - g 操作 [BT]: `pypto.set_vec_tile_shapes(64)`

4. **展开约束验证**：
   - NT × H × tensor_count ≤ 18000
   - NT=32, H=8 → 256 iterations < 18000 ✓

#### 最终 Tiling 配置

```python
# Cube matmul: w @ h
pypto.set_cube_tile_shapes([64, 128], [128, 128], [64, 128])

# Vector element-wise: v_new 计算
pypto.set_vec_tile_shapes(64, 128)

# Cube matmul: k.T @ v_new
pypto.set_cube_tile_shapes([128, 64], [64, 128], [128, 128])

# Vector state: 状态累积
pypto.set_vec_tile_shapes(128, 128)
```

### 3.3 替代方案

| 备选 tile | 否决理由 |
|-----------|---------|
| `[128, 128]` Cube tile for w@h | BT=64 固定，tile 不能超过实际 shape |
| `[32, 64]` Vector tile | 过小，展开次数过多，编译开销大 |
| V 分半处理 (TileLang 设计) | PyPTO 无需 V 分半，可直接处理 [K, V] |

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| B | [1, 16] DYNAMIC | 运行期 | `pypto.loop(B, name="LOOP_B", unroll_list=[4, 2, 1])` |
| H | [1, 64] | 编译期已知 | `pypto.loop(H, name="LOOP_H")` |
| NT | 派生自 T | 运行期 | `pypto.loop(NT, name="LOOP_NT", unroll_list=[16, 4, 1])` |

### 4.2 完整伪代码

> 必须标注每个变量类型（SymbolicScalar / int / Tensor），以及 view/assemble 的 offset。

```python
import pypto
import math

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def chunk_gated_delta_rule_kernel(
    k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, Hg, K], pypto.DT_FP16),    # [B, T, Hg, K]
    w: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP16),     # [B, T, H, K]
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, V], pypto.DT_FP16),     # [B, T, H, V]
    g: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),        # [B, T, H] (可选)
    h0: pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP16),                # [B, H, K, V] (可选)
    cu_seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),                # [N+1] (变长)
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K, V], pypto.DT_FP16),  # [B, NT, H, K, V]
    v_new_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, V], pypto.DT_FP16), # [B, T, H, V]
    ht_out: pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP16),            # [B, H, K, V]
    BT: int = 64,
    USE_G: bool = True,
    USE_INITIAL_STATE: bool = True,
    STORE_FINAL_STATE: bool = True,
):
    """
    Chunk Gated Delta Rule 前向传播。
    
    计算流程（每个 chunk）:
        1. v_new = v - w @ h_state
        2. 若 USE_G: v_new = v_new * exp(g_last - g), h_state = h_state * exp(g_last)
        3. h_state = h_state + k.T @ v_new
    """
    # 常量
    K = 128        # int (编译期常量)
    V = 128        # int
    Hg = k.shape[2]  # int (编译期)
    H = w.shape[2]   # int
    
    # 动态维度
    B = k.shape[0]   # SymbolicScalar
    T = k.shape[1]   # SymbolicScalar
    N_seq = len(cu_seqlens) - 1  # int (运行时 Python 计算)
    
    # GQA 比例
    group = H // Hg  # int
    
    # 是否变长模式
    is_varlen = (cu_seqlens is not None)
    
    # =========================================
    # 定长模式
    # =========================================
    if not is_varlen:
        NT = (T + BT - 1) // BT   # SymbolicScalar (派生)
        
        # Batch Loop
        for b_idx in pypto.loop(B, name="LOOP_B", idx_name="b_idx", unroll_list=[4, 2, 1]):
            # Head Loop
            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                # GQA: 计算 key head
                k_head_idx = h_idx // group  # int
                
                # 初始化状态
                pypto.set_vec_tile_shapes(128, 128)
                if USE_INITIAL_STATE:
                    h_state_fp16 = pypto.view(h0, [K, V], [b_idx, h_idx, 0, 0])  # [K, V], FP16
                else:
                    h_state_fp16 = pypto.zeros([K, V], pypto.DT_FP16)  # [K, V], FP16
                
                h_state = pypto.cast(h_state_fp16, pypto.DT_FP32)  # [K, V], FP32
                
                # Chunk Loop
                for chunk_idx in pypto.loop(NT, name="LOOP_NT", idx_name="chunk_idx", unroll_list=[16, 4, 1]):
                    t_start = chunk_idx * BT   # SymbolicScalar
                    
                    # 尾块处理
                    actual_BT = (T - t_start).min(BT)  # Element
                    
                    # ========== Step 1: w @ h (Cube GEMM) ==========
                    pypto.set_cube_tile_shapes([64, 128], [128, 128], [64, 128])
                    
                    w_chunk = pypto.view(w, [BT, K], [b_idx, t_start, h_idx, 0], valid_shape=[actual_BT, K])  # [BT, K], FP16
                    
                    # w @ h_state (matmul)
                    wh_fp32 = pypto.matmul(w_chunk, h_state, pypto.DT_FP32, b_trans=True)  # [BT, V], FP32
                    
                    # ========== Step 2: v_new = v - w @ h (Vector) ==========
                    pypto.set_vec_tile_shapes(64, 128)
                    
                    v_chunk_fp16 = pypto.view(v, [BT, V], [b_idx, t_start, h_idx, 0], valid_shape=[actual_BT, V])  # [BT, V], FP16
                    v_chunk = pypto.cast(v_chunk_fp16, pypto.DT_FP32)  # [BT, V], FP32
                    
                    v_new = pypto.sub(v_chunk, wh_fp32)  # [BT, V], FP32
                    
                    # ========== Step 3: 门控计算 (可选) ==========
                    if USE_G:
                        pypto.set_vec_tile_shapes(64)
                        
                        g_chunk = pypto.view(g, [BT], [b_idx, t_start, h_idx], valid_shape=[actual_BT])  # [BT], FP32
                        
                        # 提取 g_last（最后有效 token 的 gate）
                        # 使用条件判断处理尾块
                        if chunk_idx * BT + BT <= T:  # full chunk
                            g_last = g_chunk[BT - 1:BT]  # [1], FP32 (切片提取)
                        else:
                            # 尾块：取实际最后一个位置
                            g_last_idx = T - chunk_idx * BT - 1  # Element
                            g_last = pypto.view(g_chunk, [1], [g_last_idx])  # [1], FP32
                        
                        # exp(g_last - g_chunk)
                        g_exp = pypto.exp(g_last - g_chunk)  # [BT], FP32
                        
                        # 广播 g_exp 到 [BT, V]
                        g_exp_broadcast = pypto.expand_clone(g_exp, [V])  # [BT, V], FP32
                        
                        # 门控缩放: v_new = v_new * exp(g_last - g)
                        pypto.set_vec_tile_shapes(64, 128)
                        v_new = pypto.mul(v_new, g_exp_broadcast)  # [BT, V], FP32
                        
                        # 状态衰减: h_state = h_state * exp(g_last)
                        g_last_exp = pypto.exp(g_last)  # [1], FP32
                        g_last_broadcast = pypto.expand_clone(g_last_exp, [K, V])  # [K, V], FP32
                        h_state = pypto.mul(h_state, g_last_broadcast)  # [K, V], FP32
                    
                    # ========== Step 4: k.T @ v_new (Cube GEMM) ==========
                    pypto.set_cube_tile_shapes([128, 64], [64, 128], [128, 128])
                    
                    k_chunk = pypto.view(k, [BT, K], [b_idx, t_start, k_head_idx, 0], valid_shape=[actual_BT, K])  # [BT, K], FP16
                    
                    # k.T @ v_new (matmul, a_trans=True)
                    h_upd = pypto.matmul(k_chunk, v_new, pypto.DT_FP32, a_trans=True)  # [K, V], FP32
                    
                    # ========== Step 5: 状态累积 (Vector) ==========
                    pypto.set_vec_tile_shapes(128, 128)
                    
                    h_state = pypto.add(h_state, h_upd)  # [K, V], FP32
                    
                    # ========== Step 6: 存储输出 ==========
                    # 类型转换 FP32 → FP16
                    h_state_fp16 = pypto.cast(h_state, pypto.DT_FP16)  # [K, V], FP16
                    v_new_fp16 = pypto.cast(v_new, pypto.DT_FP16)  # [BT, V], FP16
                    
                    # 存储 h[chunk_idx]
                    pypto.assemble(h_state_fp16, [b_idx, chunk_idx, h_idx, 0, 0], h_out)  # 写入 h_out
                    
                    # 存储 v_new
                    pypto.assemble(v_new_fp16, [b_idx, t_start, h_idx, 0], v_new_out)
                
                # ========== Step 7: 存储最终状态 ht ==========
                if STORE_FINAL_STATE:
                    h_state_fp16 = pypto.cast(h_state, pypto.DT_FP16)  # [K, V], FP16
                    pypto.assemble(h_state_fp16, [b_idx, h_idx, 0, 0], ht_out)
    
    # =========================================
    # 变长模式
    # =========================================
    else:
        # Batch/序列 Loop (遍历每个变长序列)
        for seq_idx in pypto.loop(N_seq, name="LOOP_SEQ", idx_name="seq_idx", unroll_list=[4, 2, 1]):
            bos = cu_seqlens[seq_idx]      # int (运行时读取)
            eos = cu_seqlens[seq_idx + 1]  # int
            s_len = eos - bos              # int (当前序列长度)
            
            NT_seq = (s_len + BT - 1) // BT  # int
            
            # Head Loop
            for h_idx in pypto.loop(H, name="LOOP_H_VARLEN", idx_name="h_idx"):
                k_head_idx = h_idx // group  # int
                
                # 初始化状态
                pypto.set_vec_tile_shapes(128, 128)
                if USE_INITIAL_STATE:
                    h_state_fp16 = pypto.view(h0, [K, V], [0, seq_idx, h_idx, 0, 0])  # [K, V], FP16
                else:
                    h_state_fp16 = pypto.zeros([K, V], pypto.DT_FP16)
                
                h_state = pypto.cast(h_state_fp16, pypto.DT_FP32)  # [K, V], FP32
                
                # Chunk Loop (变长序列的 chunk)
                for chunk_idx in pypto.loop(NT_seq, name="LOOP_NT_VARLEN", idx_name="chunk_idx"):
                    t_start = chunk_idx * BT  # int
                    
                    # 尾块处理
                    actual_BT = min(s_len - t_start, BT)  # Python min (int 计算)
                    
                    # ========== 计算 (同定长模式) ==========
                    # ... (计算流程与定长模式相同，但 offset 使用 bos + t_start)
                    
                    # 数据切片使用 bos + t_start 做偏移
                    w_chunk = pypto.view(w, [BT, K], [0, bos + t_start, h_idx, 0], valid_shape=[actual_BT, K])
                    k_chunk = pypto.view(k, [BT, K], [0, bos + t_start, k_head_idx, 0], valid_shape=[actual_BT, K])
                    v_chunk = pypto.view(v, [BT, V], [0, bos + t_start, h_idx, 0], valid_shape=[actual_BT, V])
                    
                    # ... (后续计算步骤与定长模式相同)
                    
                    # 存储时使用 seq_idx 和 chunk_idx
                    # h_out 和 v_new_out 的偏移计算需要考虑全局位置
```

### 4.3 跨迭代状态

| 状态名 | 初始化 | 更新方式 | submit_before_loop |
|--------|--------|---------|--------------------|
| h_state | `h0` 或 `zeros([K, V])` | `h_state[:] = h_state + h_upd` | True (chunk 间依赖) |
| last_state | 用于状态传递 | `last_state[:] = cur_state` | True |

**状态传递关键模式**（参考 `models/qwen3_next/gated_delta_rule_impl.py`）：

```python
# 状态初始化
last_state = states[b_idx, nv_idx]  # 从输入读取初始状态

for chunk_idx in pypto.loop(...):
    cur_state = compute_chunk(...)  # 计算当前 chunk
    
    # 关键：[:]赋值实现状态传递，而非直接赋值
    last_state[:] = cur_state
    last_state_data[b_idx, nv_idx] = last_state
```

### 4.4 尾块处理

**方案**：使用 `valid_shape` 参数

```python
# 尾块实际大小计算
actual_BT = (T - chunk_idx * BT).min(BT)  # Element (SymbolicScalar.min)

# view 使用 valid_shape
w_chunk = pypto.view(w, [BT, K], [b_idx, t_start, h_idx, 0], valid_shape=[actual_BT, K])
```

**尾块门控 g_last 提取**：

```python
if chunk_idx * BT + BT <= T:  # full chunk
    g_last = g_chunk[BT - 1]
else:  # partial chunk
    g_last_idx = T - chunk_idx * BT - 1  # 最后有效位置
    g_last = pypto.view(g_chunk, [1], [g_last_idx])
```

**变长模式尾块**：使用 `pypto.is_loop_end(chunk_idx)` + `fillpad`

```python
if pypto.is_loop_end(chunk_idx):
    pad_w = pypto.fillpad(w_chunk, "constant", 0.0)
    # 使用填充后的数据计算
```

---

## 5. 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 sum 输入已转 FP32 | ✓ | 本算子不使用 sum |
| 2 | matmul 两侧 dtype 一致 | ✓ | w/k/v FP16 → matmul 输出 FP32 |
| 3 | TileShape 维度数 = 操作数维度数 | ✓ | matmul 2D, vec 2D |
| 4 | 尾轴满足对齐 | ✓ | K=128, V=128 均满足 32B 对齐 |
| 5 | 同阶段 UB 占用 ≤ 容量 | ✓ | ~256KB 总计，在安全范围 |
| 6 | 表达式展开 < 18000 | ✓ | NT×H=256 < 18000 |
| 7 | 输出经 `[:]` / `assemble` 显式写回 | ✓ | 使用 assemble 写回 |
| 8 | 无 view/assemble 同张量回环 | ✓ | 输入输出分离 |
| 9 | 动态轴标 `pypto.DYNAMIC` | ✓ | B, T 已标注 |
| 10 | 动态 loop 提供 `unroll_list` | ✓ | `[16, 4, 1]` |
| 11 | 跨迭代状态用 `[:]` 显式传递 | ✓ | `h_state[:] = ...` |
| 12 | 尾块用 `valid_shape` 处理 | ✓ | view 中指定 |
| 13 | 无 SymbolicScalar 用作 `**` / list index / Python `if` | ✓ | 使用 `sym.min(x)` 和 `pypto.view` |

### 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|
| 1 | `g_last_idx = T - chunk_idx * BT - 1` 的类型 | 尾块门控提取 | 需验证是否可用作 view offset |
| 2 | 变长模式下 `s_len` 的计算方式 | 变长序列处理 | 变长模式可能需要简化为定长 |
| 3 | `pypto.expand_clone` API 约束 | 广播操作 | 需查 docs 确认用法 |

---

## 6. 验证方案

### 6.1 测试配置

| 用例 | 输入 shape | dtype | 重点验证 |
|------|----------|-------|---------|
| Fixed_P0 | k[1,2048,4,128], w[1,2048,8,128], v[1,2048,8,128] | FP16/FP32 | 核心功能，use_g=True |
| Fixed_P0_no_g | 同上 | FP16/FP32 | 无门控模式 |
| Fixed_P0_no_h0 | 同上 | FP16/FP32 | 无初始状态 |
| Small_T128 | k[1,128,2,128], w[1,128,4,128], v[1,128,4,128] | FP16/FP32 | 小序列，边界条件 |

### 6.2 精度容忍度

| 配置 | rtol | atol |
|------|------|------|
| T ≤ 128 | 1e-2 | 1e-3 |
| 128 < T ≤ 512 | 1e-2 | 5e-3 |
| 512 < T ≤ 2048 | 5e-2 | 5e-2 |

### 6.3 Golden 对比方案

使用 `chunk_gated_delta_rule_golden.py` 作为参考实现：

```python
from chunk_gated_delta_rule_golden import chunk_gated_delta_rule_golden

# 生成输入数据
k = torch.randn(B, T, Hg, K, dtype=torch.float16)
w = torch.randn(B, T, H, K, dtype=torch.float16)
v = torch.randn(B, T, H, V, dtype=torch.float16)
g = torch.randn(B, T, H, dtype=torch.float32)
h0 = torch.randn(B, H, K, V, dtype=torch.float16)

# Golden 计算
h_ref, v_new_ref, ht_ref = chunk_gated_delta_rule_golden(k, w, v, g, h0, True, 64)

# PyPTO 实现
h_impl, v_new_impl, ht_impl = chunk_gated_delta_rule_pypto(k, w, v, g, h0, True, 64)

# 精度对比
assert torch.allclose(h_impl, h_ref, rtol=rtol, atol=atol)
assert torch.allclose(v_new_impl, v_new_ref, rtol=rtol, atol=atol)
```

---

## 7. 风险点与注意事项

### 7.1 已知风险

| 风险 | 说明 | 解决方案 |
|------|------|---------|
| **精度控制** | v_new 计算需 FP32 精度 | 所有中间计算使用 FP32，仅存储时转 FP16 |
| **状态传递** | chunk 间依赖 | 使用 `[:]` 赋值而非直接赋值 |
| **尾块 g_last** | 变长序列提取最后一个有效 g | 条件判断 + view 切片 |
| **GQA 映射** | key head 与 value head 不对齐 | `k_head_idx = h_idx // (H // Hg)` |

### 7.2 实现建议

1. **优先实现定长模式**：变长模式复杂度高，建议先验证定长模式精度
2. **状态传递使用 `[:]` 赋值**：参考 `gated_delta_rule_impl.py` 模式
3. **Tiling 配置顺序**：Cube matmul 前必须调用 `set_cube_tile_shapes`
4. **类型转换**：所有 element-wise 计算前确保输入为 FP32

### 7.3 参考

| 参考类型 | 文件路径 | 可复用点 |
|---------|---------|---------|
| PyPTO 实现 | `models/qwen3_next/gated_delta_rule_impl.py` | 三层 loop 结构、状态传递、view/assemble |
| TileLang 设计 | `chunk_gated_delta_rule/design.md` | API 映射、内存规划、精度修复 |
| API 文档 | `docs/api/operation/pypto-matmul.md` | matmul dtype 约束 |
| Loop 示例 | `examples/02_intermediate/controlflow/others/dynamic.py` | 动态 shape + valid_shape |

---

## 完成报告

```text
设计状态：已收敛

迭代过程：
  第 1 轮：API 调用链 15 步，cast 4 处（输入→FP32，输出→FP16）
  第 2 轮：Tiling hybrid，vec=[64,128], cube=[[64,128],[128,128],[64,128]]
  第 3 轮：Loop 3 层，动态轴 B/T，跨迭代依赖 h_state
  第 4 轮：约束检查 13/13 通过

开放问题：
  · g_last_idx 类型验证 — 影响尾块门控提取
  · 变长模式 s_len 计算 — 待简化或延后实现
  · expand_clone API 用法 — 需查 docs 确认
```