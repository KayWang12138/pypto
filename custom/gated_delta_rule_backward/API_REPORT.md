---
schema_version: 1
op_name: gated_delta_rule_backward
supported_dtypes: [DT_FP32]
dynamic_axes: ['B', 'T']
shape_constraints:
  - T % BT == 0
  - NT = T / BT
tiling_required: true
feasibility: feasible
---

# API 探索报告

> **生成时间**: 2026-04-21

---

## 1. 概述

### 1.1 输入摘要

`gated_delta_rule_backward` 是 Gated Delta Rule 注意力机制的反向传播算子。输入包括原始 q/k/v、门控参数 g_raw/beta、初始/最终状态梯度 do/dht，以及前向缓存（A, w, S_before, v_new, q_norm, k_norm, q_rstd, k_rstd）。输出 6 个梯度张量：dq, dk, dv, db, dg_raw, dh0。

### 1.2 算子分类

- **类型**: 混合（Cube + Vector）
- **判断依据**: 核心计算涉及大量矩阵乘法（matmul，Cube 类型）和逐元素/归约操作（Vector 类型）。前向实现 `gated_delta_rule_impl.py` 即采用混合 tiling 策略。

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | matmul | `g_cum = c_cum @ gc_raw` | 门控累积（cumsum） |
| 2 | elementwise | `eg = exp(g_cum)`, `decay = exp(g_cum[:,None] - g_cum[None,:])` | 指数衰减 |
| 3 | matmul | `qk = qc @ kc^T` | query-key 相似度 |
| 4 | elementwise | `a_local = (qk * decay) * m_le` | 局部注意力矩阵 |
| 5 | matmul | `dv0 = (a_local^T @ doc) * scale` | 局部 v 梯度 |
| 6 | matmul | `dv_state = (kc @ d_s) * s_tok` | 状态 v 梯度 |
| 7 | elementwise | `dv_total = dv_state + dv0` | 合并 v 梯度 |
| 8 | matmul | `q_eff^T @ doc * scale` | 状态 q 贡献 |
| 9 | matmul | `w^T @ dv_total` | 状态 w 贡献 |
| 10 | matmul | `dw = -(dv_total @ s_before^T)`, `du = dv_total` | WY 梯度 |
| 11 | matmul | `dvb = A^T @ du`, `dkbg = A^T @ dw` | 反推 dv/dk |
| 12 | matmul | `d_l = -(A^T @ (d_a @ A^T))` | L 矩阵梯度 |
| 13 | matmul | `(m_mat + m_mat^T) @ kc` | k 交叉梯度 |
| 14 | matmul | `dg_raw = c_rcum @ dg_cum` | 反向 cumsum |
| 15 | reduction | `(dy * y).sum(-1)` | L2 norm 反向中间步骤 |
| 16 | elementwise | `dx = dy * rstd - dot * y * rstd` | L2 norm 反向 |
| 17 | matmul | `kkt = kc @ kc^T` | key-key 外积 |
| 18 | matmul | 多处 `@` 运算 | 状态递推梯度 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| matmul (2D) | `a @ b` | `pypto.matmul(a, b, pypto.DT_FP32)` | direct | ✓ |
| matmul + transpose | `a @ b^T` | `pypto.matmul(a, b, pypto.DT_FP32, b_trans=True)` | direct | ✓ |
| matmul + transpose | `a^T @ b` | `pypto.matmul(a, b, pypto.DT_FP32, a_trans=True)` | direct | ✓ |
| 元素乘 | `a * b` | `pypto.mul(a, b)` | direct | ✓ |
| 元素加 | `a + b` | `pypto.add(a, b)` | direct | ✓ |
| 元素减 | `a - b` | `pypto.sub(a, b)` | direct | ✓ |
| 指数 | `exp(x)` | `pypto.exp(x)` | direct | ✓ |
| 指数差 | `exp(x - y)` | `pypto.expand_exp_dif(x, y)` | direct | ✓ |
| 归约求和 | `x.sum(dim)` | `pypto.sum(x, dim, keepdim)` | direct | ✓ (仅 FP32) |
| 转置 | `x^T` (2D) | `pypto.transpose(x, 0, 1)` 或 matmul `a_trans/b_trans` | direct | ✓ |
| L2 norm 反向 | `dx = dy*rstd - (dy*y).sum(-1)*y*rstd` | `mul + sum + mul + sub` 组合 | substitute | ✓ |
| 三角求解 | `linalg.solve_triangular` | **不需要**（反向已用 matmul 展开） | N/A | N/A |
| 常量构造 | identity, tril, triu | `ones + tril/triu`, `arange + one_hot + cast` | substitute | ✓ |
| 类型转换 | `x.to(fp32)` | `pypto.cast(x, pypto.DT_FP32)` | direct | ✓ |
| 填充/补齐 | 常量张量 | `pypto.zeros`, `pypto.ones`, `pypto.full` | direct | ✓ |
| 维度操作 | reshape, view | `pypto.reshape`, `pypto.view` | direct | ✓ |

### 3.2 Substitute 配方

```
L2_norm_backward(y, rstd, dy):
  term1 = pypto.mul(dy, rstd)           # dy * rstd
  dot = pypto.sum(pypto.mul(dy, y), -1, True)  # (dy * y).sum(-1)
  term2 = pypto.mul(dot, pypto.mul(y, rstd))   # dot * y * rstd
  dx = pypto.sub(term1, term2)
  return dx

identity_matrix(BT):
  indices = pypto.arange(BT)
  identity = pypto.cast(pypto.one_hot(indices, BT), pypto.DT_FP32)
  return identity

lower_triangular_ones(BT):
  return pypto.tril(pypto.ones([BT, BT], pypto.DT_FP32))
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP32 | float32 | ✓ |
| contiguous | 必须 | 需确保 from_torch 输入连续 | ✓ |
| shape | 非空 | 所有输入 shape 非空 | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `pypto.matmul` | dtype | FP16/BF16/FP32 | ✓ (使用 FP32) |
| `pypto.matmul` | shape | 2D-4D | ✓ (核心 2D matmul) |
| `pypto.matmul` | tiling | 调用前需 set_cube_tile_shapes | ✓ |
| `pypto.sum` | dtype | 仅 FP32/INT32/INT16 | ✓ (使用 FP32) |
| `pypto.exp` | dtype | FP16/BF16/FP32 | ✓ |
| `pypto.mul/add/sub` | dtype | 同类型输入 | ✓ |
| `pypto.tril/triu` | 2D+ | 需至少 2D | ✓ |
| `pypto.transpose` | 轴限制 | 4D 仅支持特定轴对 | ✓ (主要用 2D 转置) |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API | 说明 |
|----------|-----------|------|
| Cube (matmul) | `pypto.set_cube_tile_shapes([mL0,mL1], [kL0,kL1], [nL0,nL1])` | 每次 matmul 前需设置；k/n 轴 32 字节对齐 |
| Vector (elementwise/reduce) | `pypto.set_vec_tile_shapes(...)` | 每次逐元素/归约操作前需设置 |

前向实现参考 `gated_delta_rule_impl.py` 使用固定 tiling `[128, 128]` for cube 和 `128, 128` for vector。反向算子预计使用类似策略。

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/qwen3_next/gated_delta_rule_golden.py` | models | **极高** | 高 | 完整的 PyTorch backward golden，精确的反向算法和公式 |
| `models/qwen3_next/gated_delta_rule_impl.py` | models | **极高** | 高 | PyPTO 前向实现，API 模式、tiling、loop 结构、状态管理可直接复用 |
| `models/qwen3_next/qwen3_next_gated_delta_rule.py` | models | 中 | 高 | 测试基础设施和输入生成逻辑 |
| `models/qat/qat_impl.py` | models | 中 | 高 | 生产级 backward kernel，展示前向重算 + 多输出梯度 + tiling 模式 |
| `models/arctic/sum_lstm.py` | models | 低-中 | 高 | 状态递推模式（pypto.view/pypto.assemble 跨 loop 携带状态） |
| `examples/03_advanced/advanced_nn/attention/attention.py` | examples | 高 | 高 | matmul+transpose+softmax+loop 模式 |
| `examples/02_intermediate/basic_nn/ffn/ffn_module.py` | examples | 高 | 高 | 动态分批 view+assemble+matmul 模式 |
| `examples/02_intermediate/controlflow/loop/loop.py` | examples | 高 | 高 | loop+view+assemble 核心迭代模式 |

### 6.2 可复用模式

- **API 调用模式**: `pypto.matmul(a, b, pypto.DT_FP32, a_trans/b_trans)` 用于所有矩阵乘法；`pypto.exp`, `pypto.mul`, `pypto.add`, `pypto.sub`, `pypto.sum` 用于逐元素和归约
- **Tiling 策略**: 参考前向 `gated_delta_rule_impl.py`，cube tiling 使用 `[128, 128]` 固定分块，vector tiling 使用 `128, 128`
- **Loop 结构**: 三层嵌套 `pypto.loop` (batch, head, chunk)，逆序遍历 chunks。参考前向的 `chunk_gated_delta_rule` 函数。
- **状态管理**: 使用 `pypto.tensor` 创建可变状态 `d_s[B, H, K, V]`，通过 `pypto.view` 读取和 `d_s[:] = new_value` 更新
- **常量矩阵**: host 端预构造 `i_mat`, `m_le`, `m_lt`, `c_cum`, `c_rcum`，通过 `pypto.from_torch` 输入 kernel

### 6.3 差异分析

| 差异点 | 前向做法 | 反向需求 | 调整建议 |
|--------|----------|----------|----------|
| Chunk 遍历方向 | 正序 (0→NT-1) | **逆序** (NT-1→0) | 循环索引 `c = NT-1-i` |
| 状态初始值 | `initial_state` | `dht` (最终状态梯度) | 初始化 `d_s = dht` |
| 矩阵求逆 | `inverse_pto` 分块求逆 | 不需要（直接使用前向缓存的 A） | 直接读取 `A[b,h,c]` |
| L2 norm | 前向 `l2norm` | 反向 `l2norm_bwd` | 用 mul+sum+sub 组合实现 |
| 输出数量 | 3 (out, final_state, cache) | 6 (dq, dk, dv, db, dg_raw, dh0) | 需要多个输出 tensor |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | 所有必需 API 均有直接映射或可组合实现 | — |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 指数运算数值稳定性 | `exp(g_cum)` 可能产生极大/极小值，需确保 fp32 精度足够 |
| 矩阵乘法维度对齐 | 所有 matmul 的 K 轴必须对齐，使用 `a_trans/b_trans` 替代独立 transpose |
| Tiling 切换 | Cube 和 Vector 操作交替频繁，需确保每次切换前正确设置对应 tiling |
| 状态跨 loop 携带 | `d_s` 需在 chunk 循环内跨迭代更新，参考 `pypto.tensor` + slice 赋值模式 |
| sum 仅支持 FP32 | 所有 reduction 必须在 FP32 精度下执行 |
| 常量矩阵需预构造 | `i_mat`, `m_le`, `m_lt`, `c_cum`, `c_rcum` 需在 host 端构造后传入 kernel |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 全列表 | `docs/api/operation/index.md` |
| matmul 文档 | `docs/api/operation/pypto-matmul.md` |
| exp 文档 | `docs/api/operation/pypto-exp.md` |
| expand_exp_dif 文档 | `docs/api/operation/pypto-expand_exp_dif.md` |
| sum 文档 | `docs/api/operation/pypto-sum.md` |
| tril/triu 文档 | `docs/api/operation/pypto-tril.md`, `pypto-triu.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Vector tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| Cube tiling | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| DataType 枚举 | `docs/api/datatype/DataType.md` |
| Golden reference | `models/qwen3_next/gated_delta_rule_golden.py` |
| 前向实现 | `models/qwen3_next/gated_delta_rule_impl.py` |
| 测试模块 | `models/qwen3_next/qwen3_next_gated_delta_rule.py` |
| QAT backward | `models/qat/qat_impl.py` |
| LSTM 状态递推 | `models/arctic/sum_lstm.py` |
| Attention 示例 | `examples/03_advanced/advanced_nn/attention/attention.py` |
| FFN 示例 | `examples/02_intermediate/basic_nn/ffn/ffn_module.py` |
| Loop 示例 | `examples/02_intermediate/controlflow/loop/loop.py` |

---

## 9. 结论

- **可行性**: ✅ **可行**
- **主要问题**: 无阻断问题。所有核心操作（matmul、exp、sum、elementwise、tril/triu）均有直接 PyPTO API 支持。反向计算不需要矩阵求逆（前向缓存的 A 矩阵可直接使用）。L2 norm 反向可通过 mul+sum+sub 组合实现。
- **推荐参考**: 以 `gated_delta_rule_impl.py` 为 PyPTO API 模式主参考，以 `gated_delta_rule_golden.py` 的 `torch_golden_gated_delta_rule_backward_ref` 为算法逻辑主参考。
