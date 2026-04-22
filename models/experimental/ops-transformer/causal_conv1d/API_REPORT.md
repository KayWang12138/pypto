---
schema_version: 1
op_name: causal_conv1d
supported_dtypes: [float16, bfloat16, float32]
dynamic_axes: ['total_len', 'batch', 'dim', 'num_cache_lines']
shape_constraints: {width: [3, 6], state_len: '>= width-1', dim: '>= 256'}
tiling_required: true
feasibility: feasible
---

# API 探索报告

> **生成时间**: 2026-04-21

---

## 1. 概述

### 1.1 输入摘要

causal_conv1d 是一个因果卷积算子，支持 Prefill 和 Decode 两种模式：
- **Prefill**: 处理完整序列，支持变长（packed layout）
- **Decode**: 处理单个或多个 token（投机解码），状态缓存滚动更新

核心计算：`y[t] = activation(bias + Σ w[i] * x[t-width+1+i])`

### 1.2 算子分类

- **类型**: Vector（纯逐元素操作）
- **判断依据**: 计算仅涉及逐元素乘加（mul/add）和激活函数（exp/div），无矩阵乘法（matmul）

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | shape | `hist[0..width-2]` | 从 conv_state 或历史 token 加载 |
| 2 | shape | `x_cur = x[t]` | 加载当前 token |
| 3 | shape | `w[0..width-1]` | 加载卷积权重 |
| 4 | elementwise | `acc = Σ w[i] * hist[i] + w[n] * x_cur` | 逐元素乘加累加 |
| 5 | activation | `silu(acc) = acc / (1 + exp(-acc))` | SiLU 激活函数 |
| 6 | shape | `hist滚动更新` | 历史状态滚动 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | 加载 hist | `pypto.view` / `pypto.from_torch` | direct | ✓ |
| 2 | 加载 x_cur | `pypto.view` | direct | ✓ |
| 3 | 加载 w | `pypto.from_torch` | direct | ✓ |
| 4a | w * hist | `pypto.mul` | direct | ✓ |
| 4b | acc += ... | `pypto.add` | direct | ✓ |
| 5a | -acc | `pypto.mul(acc, -1)` | substitute | ✓ |
| 5b | exp(-acc) | `pypto.exp` | direct | ✓ |
| 5c | 1 + exp | `pypto.add` | direct | ✓ |
| 5d | acc / denom | `pypto.div` | direct | ✓ |
| 6 | 状态更新 | `pypto.assemble` | direct | ✓ |

### 3.2 Substitute 配方

```
silu: acc / (1 + exp(-acc))
  → neg_acc = pypto.mul(acc, -1.0)           # 手动实现负号
  → exp_neg = pypto.exp(neg_acc)             # exp(-acc)
  → denom = pypto.add(exp_neg, 1.0)          # 1 + exp(-acc)
  → silu_out = pypto.div(acc, denom)         # acc / denom

# 或使用 pypto.sigmoid (仅支持 FP32)
silu: acc * sigmoid(acc)
  → sigmoid_acc = pypto.sigmoid(acc)         # 仅 DT_FP32
  → silu_out = pypto.mul(acc, sigmoid_acc)
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT* | float16 | ✓ |
| contiguous | 必须 | — | ✓（需确保） |
| shape size | ≤ INT32_MAX | 2048×2048 | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `pypto.mul` | dtype | FP16/BF16/FP32/INT16/INT32 | ✓ |
| `pypto.add` | dtype | FP16/BF16/FP32/INT16/INT32 | ✓ |
| `pypto.sub` | dtype | FP16/BF16/FP32/INT16/INT32 | ✓ |
| `pypto.exp` | dtype | FP16/BF16/FP32 | ✓ |
| `pypto.div` | dtype | FP16/BF16/FP32 | ✓ |
| `pypto.sigmoid` | dtype | **仅 DT_FP32** | ⚠ 需先 cast 到 FP32 |
| `pypto.loop` | 返回值 | SymInt，不可作为列表下标 | ✓ |
| `pypto.view` | valid_shape | 动态轴需配合 valid_shape | ✓ |
| `pypto.assemble` | offsets | 需计算正确的 offset | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes(tile_m, tile_n)` |

**推荐配置**：
```python
# Prefill 模式：按 dim 分块
pypto.set_vec_tile_shapes(64, 512)  # [token_block, dim_block]

# Decode 模式：按 dim 分块
pypto.set_vec_tile_shapes(1, 512)   # [seqlen, dim_block]
```

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/qwen3_next/gated_delta_rule_impl.py` | models | **高** | 高 | 状态管理、序列分块、变长处理 |
| `examples/02_intermediate/operators/activation/activation.py` | examples | **高** | 高 | SiLU/SwiGLU 激活函数实现 |
| `examples/02_intermediate/basic_nn/ffn/ffn_module.py` | examples | **高** | 高 | 手动 sigmoid、pypto.loop |
| `examples/02_intermediate/controlflow/loop/loop.py` | examples | 中 | 高 | view/assemble、动态轴处理 |
| `models/arctic/sum_lstm.py` | models | 中 | 高 | sigmoid 激活、状态更新 |
| `models/glm_v4_5/glm_ffn_common_interface.py` | models | 中 | 高 | SwiGLU 实现 |

### 6.2 可复用模式

#### API 调用模式

```python
# 状态初始化（来源: gated_delta_rule_impl.py）
last_state = states[b_idx, nv_idx]  # 从输入获取初始状态
last_state[:] = cur_state  # 更新状态
last_state_data[b_idx, nv_idx] = last_state  # 输出最终状态

# 变长序列 view（来源: gated_delta_rule_impl.py）
query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])

# 尾部块填充（来源: gated_delta_rule_impl.py）
if pypto.is_loop_end(s_idx):
    pad_q = pypto.fillpad(query_view_2d, "constant", 0.0)
```

#### SiLU 激活函数实现

```python
# 来源: activation.py
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def silu_activation_kernel(x: pypto.Tensor(), out: pypto.Tensor()):
    configure_tiling(x)
    out[:] = x * pypto.sigmoid(x)  # 仅 FP32

# 来源: ffn_module.py（手动实现，支持 FP16）
def swiglu_activation_core(gate: pypto.tensor, up: pypto.tensor) -> pypto.tensor:
    gate_neg = pypto.mul(gate, -1.0)
    exp_neg = pypto.exp(gate_neg)
    ones = pypto.full(exp_neg.shape, 1.0, exp_neg.dtype, valid_shape=exp_neg.shape)
    sigmoid = pypto.div(ones, pypto.add(exp_neg, ones))
    swish = pypto.mul(gate, sigmoid)
    return pypto.mul(swish, up)
```

#### 序列循环处理

```python
# 来源: loop.py
for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
    b_offset = idx * tile_b
    valid_shape = [actual_len, w, n, c]
    t0_sub = pypto.view(input0, [tile_b, w, n, c], [b_offset, 0, 0, 0], valid_shape=valid_shape)
    # Process...
    pypto.assemble(result, [b_offset, 0, 0, 0], output)
```

#### Tiling 策略

```python
# 来源: gated_delta_rule_impl.py
pypto.set_vec_tile_shapes(128, 128)  # [tile_m, tile_n]
pypto.set_vec_tile_shapes(16, 16, 128, 128)  # 多维度配置
```

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 激活函数 | `pypto.sigmoid`（仅 FP32） | 需要 FP16 输入输出 | 使用手动实现：exp/add/div 组合 |
| 状态管理 | `last_state[:] = cur_state` | 需 conv_state 滚动更新 | 使用 view + assemble 组合 |
| 序列处理 | 固定分块 L=128 | 需支持动态 seqlen | 使用 `valid_shape` + `pypto.is_loop_end()` |
| 权重加载 | 直接 from_torch | 需多次加载 w[0..n] | 将权重预先加载到 buffer 或逐次 view |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | - | - |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| sigmoid dtype | `pypto.sigmoid` 仅支持 FP32；若输入为 FP16，需使用手动实现（exp/div 组合） |
| 状态缓存管理 | conv_state 滚动更新需要正确的 view/assemble offset 计算 |
| 变长序列尾部 | 使用 `pypto.is_loop_end()` + `pypto.fillpad()` 处理非对齐尾部块 |
| loop 返回值 | `pypto.loop` 返回 SymInt，不可直接作为列表下标，需用 view/assembl |
| 权重 buffer | 若多次访问权重，可考虑预先加载到 buffer 减少重复读取 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| add API | `docs/api/operation/pypto-add.md` |
| sub API | `docs/api/operation/pypto-sub.md` |
| mul API | `docs/api/operation/pypto-mul.md` |
| div API | `docs/api/operation/pypto-div.md` |
| exp API | `docs/api/operation/pypto-exp.md` |
| full API | `docs/api/operation/pypto-full.md` |
| loop API | `docs/api/controlflow/pypto-loop.md` |
| view API | `docs/api/operation/pypto-view.md` |
| assemble API | `docs/api/operation/pypto-assemble.md` |
| sigmoid API | `docs/api/operation/pypto-sigmoid.md`（仅 FP32） |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 最佳参考 | `models/qwen3_next/gated_delta_rule_impl.py` |
| SiLU 参考 | `examples/02_intermediate/operators/activation/activation.py` |
| FFN 参考 | `examples/02_intermediate/basic_nn/ffn/ffn_module.py` |

---

## 9. 结论

- **可行性**: **可行**
- **主要问题**: sigmoid 仅支持 FP32，需使用手动实现
- **推荐方案**: 
  1. 状态管理参考 `gated_delta_rule_impl.py`
  2. SiLU 激活参考 `ffn_module.py` 手动实现（支持 FP16）
  3. 序列循环参考 `loop.py` view/assemble 模式
- **预估实现难度**: 中等（需处理状态管理和变长序列）