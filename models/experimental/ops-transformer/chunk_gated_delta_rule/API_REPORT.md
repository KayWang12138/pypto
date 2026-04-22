---
schema_version: 1
op_name: chunk_gated_delta_rule
supported_dtypes: [DT_FP16, DT_FP32]
dynamic_axes: ['B', 'T', 'N']
shape_constraints: {K: 128, V: 128, BT: 64, Hg: 'H/Hg must be integer ratio'}
tiling_required: true
feasibility: feasible
---

# API 探索报告

> **生成时间**: 2026-04-21

---

## 1. 概述

### 1.1 输入摘要

chunk_gated_delta_rule 是线性注意力机制算子，包含：
- GEMM 计算：`w @ h` (BT×K × K×V → BT×V), `k.T @ v_new` (K×BT × BT×V → K×V)
- element-wise 计算：sub, exp, mul, add
- chunk 分块循环 (BT=64)
- 状态累积传递 (h_state)
- 变长序列支持 (cu_seqlens)

### 1.2 算子分类

- **类型**: 混合 (Cube + Vector)
- **判断依据**: 包含 matmul (Cube 类型) 和 element-wise/exp/sub/mul/add (Vector 类型)

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | matmul (Cube) | `wh = w @ h_state` | w[BT,K] × h[K,V] → wh[BT,V] |
| 2 | sub (Vector) | `v_new = v - wh` | 残差计算 |
| 3 | exp (Vector) | `g_exp = exp(g_last - g)` | 门控指数 (可选) |
| 4 | mul (Vector) | `v_new = v_new * g_exp` | 门控缩放 (可选) |
| 5 | mul (Vector) | `h_state = h_state * exp(g_last)` | 状态衰减 (可选) |
| 6 | matmul (Cube) | `h_upd = k.T @ v_new` | k[K,BT] × v_new[BT,V] → h_upd[K,V] |
| 7 | add (Vector) | `h_state = h_state + h_upd` | 状态累积 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| GEMM (w@h) | `wh = w @ h` | `pypto.matmul(w, h, dtype)` | **direct** | ✓ |
| GEMM (k.T@v) | `h_upd = k.T @ v_new` | `pypto.matmul(k, v_new, dtype, a_trans=True)` | **direct** | ✓ |
| sub | `v_new = v - wh` | `pypto.sub(v, wh)` 或 `v - wh` | **direct** | ✓ |
| exp | `g_exp = exp(g_last - g)` | `pypto.exp(x)` | **direct** | ✓ |
| mul | `v_new * g_exp` | `pypto.mul(v_new, g_exp)` 或 `v_new * g_exp` | **direct** | ✓ |
| add | `h_state + h_upd` | `pypto.add(h_state, h_upd)` 或 `h_state + h_upd` | **direct** | ✓ |
| 填充 | `fill(h_state, 0)` | `pypto.zeros([K, V], dtype)` 或 `pypto.full(size, 0, dtype)` | **direct** | ✓ |
| 广播 | `broadcast(g_exp, [BT//2, V])` | `pypto.expand_clone(g_exp, shape)` | **direct** | ✓ |
| 类型转换 | `FP16 → FP32` | `pypto.cast(tensor, dtype)` | **direct** | ✓ |
| 比较 | `compare(x, 0, "LE")` | `pypto.le(x, 0)` | **direct** | ✓ |
| 条件选择 | `select(mask, a, b)` | `pypto.where(mask, a, b)` | **direct** | ✓ |
| 循环 | `for i in range(NT)` | `pypto.loop(NT)` | **direct** | ✓ |
| 切片 | `h[i, :]` | `pypto.view(h, shape, offset)` | **direct** | ✓ |
| 组装 | `output[i] = result` | `pypto.assemble(result, offset, output)` | **direct** | ✓ |

### 3.2 Substitute 配方

**无需 substitute**：所有操作均有直接 API 支持。

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/FP32/BF16/INT8/INT16/INT32 | FP16 (输入), FP32 (计算) | ✓ |
| contiguous | 必须 | — | ✓ 需确保 |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `pypto.matmul` | dtype | DT_FP16, DT_FP32, DT_BF16, DT_INT8 | ✓ |
| `pypto.matmul` | 维度 | 2-4 维，左右一致 | ✓ [BT,K] × [K,V] |
| `pypto.matmul` | 对齐 | ND: 内轴[1,65535]; NZ: 32字节对齐 | ⚠ 需确保 K=128 对齐 |
| `pypto.add/sub/mul` | dtype | DT_FP16, DT_FP32, DT_BF16, DT_INT16/32 | ✓ |
| `pypto.add/sub/mul` | 维度 | 2-4 维 | ✓ |
| `pypto.exp` | dtype | DT_FP16, DT_FP32, DT_BF16 | ✓ |
| `pypto.exp` | 精度模式 | INTRINSIC/HIGH_PRECISION | ✓ 可用 HIGH_PRECISION |
| `pypto.loop` | 参数 | start, stop, step | ✓ |

---

## 5. Tiling 需求

### 5.1 Cube Tiling (matmul)

| 算子类型 | 需调用 API | 配置示例 |
|----------|-----------|----------|
| Cube (matmul) | `pypto.set_cube_tile_shapes()` | `[128, 128], [128, 128], [128, 128]` |

**典型配置**：
```python
# w @ h: [BT, K] × [K, V] → [BT, V]
pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])

# k.T @ v_new: [K, BT] × [BT, V] → [K, V]
pypto.set_cube_tile_shapes([128, 128], [64, 64], [128, 128])
```

### 5.2 Vector Tiling (element-wise)

| 算子类型 | 需调用 API | 配置示例 |
|----------|-----------|----------|
| Vector (sub/exp/mul/add) | `pypto.set_vec_tile_shapes()` | `[32, 128]` (BT//2 × V) |

**典型配置**：
```python
# element-wise 操作: [BT//2, V]
pypto.set_vec_tile_shapes(32, 128)  # V 分半处理

# 状态累积: [K//2, V]
pypto.set_vec_tile_shapes(64, 128)
```

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `/data/x00952168/pypto/models/qwen3_next/gated_delta_rule_impl.py` | **models/** | **95%** | **高** | 完整 chunk_gated_delta_rule 实现 |
| `/data/x00952168/pypto/models/arctic/sum_lstm.py` | models/ | 75% | 高 | LSTM 状态累积模式 |
| `/data/x00952168/pypto/models/glm_v4_5/glm_attention.py` | models/ | 70% | 高 | Flash Attention + online softmax |
| `/data/x00952168/pypto/examples/02_intermediate/operators/softmax/softmax.py` | examples/ | 80% | 高 | element-wise + reduce 组合 |
| `/data/x00952168/pypto/examples/02_intermediate/controlflow/others/dynamic.py` | examples/ | 85% | 高 | 动态 shape + loop + view/assemble |

### 6.2 可复用模式

**主要参考实现**: `/data/x00952168/pypto/models/qwen3_next/gated_delta_rule_impl.py`

#### 核心可复用点

1. **三层嵌套 Loop 结构**:
```python
for b_idx in pypto.loop(b, name="LOOP_B_TND", idx_name="b_idx"):
    s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]  # 获取序列长度
    b_ofs = act_seq_len[b_idx]  # 获取偏移
    for nv_idx in pypto.loop(nv, name="LOOP_Nv_TND", idx_name="nv_idx"):
        last_state = states[b_idx, nv_idx]  # 初始化状态
        for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx", unroll_list=[16, 1]):
            # Chunk 计算...
```

2. **状态传递模式**:
```python
# 状态初始化
last_state = states[b_idx, nv_idx]

for s_idx in pypto.loop(0, s, l, ...):
    chunk_attn_out, cur_state = recurrent_state_attn_all(..., state=last_state, ...)
    
    # 关键：[:]赋值实现状态传递
    last_state[:] = cur_state
    last_state_data[b_idx, nv_idx] = last_state
```

3. **变长序列处理**:
```python
actual_l = (s - s_idx).min(l)  # 最后一个 chunk 可能不满
valid_shape=[actual_l, d]  # 使用 valid_shape 处理变长

# view 切片
query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])

# 非对齐填充
if pypto.is_loop_end(s_idx):
    pad_q = pypto.fillpad(query_view_2d, "constant", 0.0)
```

4. **GEMM + element-wise 混合**:
```python
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
gate_cum = pypto.matmul(tril, gate_view, pypto.DT_FP32)  # GEMM (Cube)

pypto.set_vec_tile_shapes(128, 128)
decay_mask = ((gate_cum - gate_cum.transpose(0, 1)) * tril).exp()  # element-wise (Vector)
```

5. **Tiling 配置策略**:
```python
# Vector 操作前设置
pypto.set_vec_tile_shapes(128, 128)
pypto.set_vec_tile_shapes(16, 16, 128, 128)  # 4D

# Cube matmul 前设置
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])  # 小块
```

### 6.3 差异分析

| 差异点 | 参考实现做法 | 本算子需求 | 调整建议 |
|--------|--------------|------------|----------|
| chunk size | L=128 | BT=64 | 调整 loop step 和 tile shapes |
| 状态维度 | [D, D] (D=128) | [K, V] (K=128, V=128) | 保持一致 |
| GQA | nv // nqk | H // Hg | 相同逻辑 |
| matmul 转置 | `a_trans=True` / `b_trans=True` | k.T @ v_new 需要 transpose | 使用 `a_trans=True` |
| 输入 dtype | DT_FP32 | DT_FP16 | 需要用 `pypto.cast` 转换 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| **无阻断问题** | PyPTO API 完整支持所需操作 | 可直接实现 |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| **精度控制** | v_new 计算需使用 float32 精度，避免累积误差。使用 `pypto.cast` 在 FP16 和 FP32 间转换 |
| **状态传递** | 必须使用 `last_state[:] = cur_state` 语法，而非直接赋值 |
| **Tiling 配置顺序** | Cube 操作前必须调用 `set_cube_tile_shapes`，Vector 操作前必须调用 `set_vec_tile_shapes` |
| **变长边界处理** | 使用 `valid_shape` 和 `fillpad` 处理最后一个 chunk 的非对齐情况 |
| **对齐约束** | matmul 的 K/N 轴需要 32 字节对齐（FP32 场景 16 元素对齐），K=128/V=128 已满足 |
| **Gate exp 数值稳定** | 设计文档使用 `compare + select` 处理负无穷，可用 `pypto.le + pypto.where` 替代 |

---

## 8. 证据索引

### 8.1 API 文档

| 信息 | 文档路径 |
|------|----------|
| API 索引 | `docs/api/operation/index.md` |
| matmul 文档 | `docs/api/operation/pypto-matmul.md` |
| add 文档 | `docs/api/operation/pypto-add.md` |
| sub 文档 | `docs/api/operation/pypto-sub.md` |
| mul 文档 | `docs/api/operation/pypto-mul.md` |
| exp 文档 | `docs/api/operation/pypto-exp.md` |
| cast 文档 | `docs/api/operation/pypto-cast.md` |
| where 文档 | `docs/api/operation/pypto-where.md` |
| zeros/full 文档 | `docs/api/operation/pypto-zeros.md`, `pypto-full.md` |
| expand_clone 文档 | `docs/api/operation/pypto-expand_clone.md` |
| view 文档 | `docs/api/operation/pypto-view.md` |
| assemble 文档 | `docs/api/operation/pypto-assemble.md` |

### 8.2 配置文档

| 信息 | 文档路径 |
|------|----------|
| Cube Tiling | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| Vector Tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` |

### 8.3 控制流文档

| 信息 | 文档路径 |
|------|----------|
| loop 文档 | `docs/api/controlflow/pypto-loop.md` |
| function 文档 | `docs/api/controlflow/pypto-function.md` |

### 8.4 参考实现

| 信息 | 文件路径 |
|------|----------|
| 主要参考 | `models/qwen3_next/gated_delta_rule_impl.py` |
| LSTM 参考 | `models/arctic/sum_lstm.py` |
| Attention 参考 | `models/glm_v4_5/glm_attention.py` |
| 动态 shape 示例 | `examples/02_intermediate/controlflow/others/dynamic.py` |
| softmax 示例 | `examples/02_intermediate/operators/softmax/softmax.py` |

---

## 9. 结论

- **可行性**: **可行**
- **主要问题**: 无阻断问题，可直接基于参考实现开发
- **推荐策略**: 
  1. 以 `models/qwen3_next/gated_delta_rule_impl.py` 为主要参考模板
  2. 调整 chunk size 从 L=128 到 BT=64
  3. 添加 FP16→FP32 类型转换支持精度控制
  4. 使用 `pypto.loop` + `pypto.view` + `pypto.assemble` 实现分块循环
  5. 使用 `[:]` 赋值实现状态传递

---

## 附录：参考实现核心代码片段

### recurrent_state_attn_all 函数（状态累积核心）

```python
def recurrent_state_attn_all(**kwargs) -> tuple[pypto.Tensor, pypto.Tensor]:
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    k_cumdecay = kwargs.get("k_cumdecay")
    gate = kwargs.get("gate")
    state = kwargs.get("state")
    decay_mask = kwargs.get("decay_mask")
    tril = kwargs.get("tril")

    dv = value.shape[-1]
    l = gate.valid_shape[0]
    gate_exp = gate.exp()
    
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    v_prime = pypto.matmul(k_cumdecay, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv]
    attn_inter = pypto.matmul(qgexp, state, pypto.DT_FP32, b_trans=True)
    
    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
    temp_matmul_vprime = pypto.matmul(v_prime, kgexp, pypto.DT_FP32, a_trans=True)
    
    _last_gate_2 = pypto.expand_clone(gate_exp[l - 1:l, :], (dv, 1))
    final_state_1 = state * _last_gate_2
    state_new = final_state_1 + temp_matmul_value - temp_matmul_vprime
    
    return chunk_attn_out, state_new
```

### 主循环结构

```python
for b_idx in pypto.loop(b, name="LOOP_B_TND", idx_name="b_idx"):
    s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
    b_ofs = act_seq_len[b_idx]
    for nv_idx in pypto.loop(nv, name="LOOP_Nv_TND", idx_name="nv_idx"):
        nqk_idx = nv_idx // group
        last_state = states[b_idx, nv_idx]
        for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx", unroll_list=[16, 1]):
            bs_ofs = b_ofs + s_idx
            actual_l = (s - s_idx).min(l)
            
            # view 切片
            query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
            
            # 计算
            chunk_attn_out, cur_state = recurrent_state_attn_all(...)
            
            # 状态传递
            last_state[:] = cur_state
            last_state_data[b_idx, nv_idx] = last_state
```