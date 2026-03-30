# SiLU 算子 API 探索报告

## 1. 概述

- **算子名称**: silu (Sigmoid Linear Unit / Swish)
- **数学公式**: `y = x * sigmoid(x) = x / (1 + exp(-x))`
- **算子类型**: element-wise 激活函数
- **实现策略**: 组合实现 (基于 pypto.sigmoid 或手动展开 sigmoid)
- **可行性**: ✅ 完全可行

---

## 2. 公式分解

### 原始公式
```
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

### 计算步骤分解

**方案 A - 直接调用 sigmoid API（FP32 only）**:
```
1. sigmoid_x = pypto.sigmoid(x)      # sigmoid(x) = 1 / (1 + exp(-x))
2. y = pypto.mul(x, sigmoid_x)       # x * sigmoid(x)
```

**方案 B - 手动展开 sigmoid（支持 FP16/BF16/FP32）**:
```
1. neg_x = pypto.mul(x, -1.0)        # -x
2. exp_neg_x = pypto.exp(neg_x)      # exp(-x)
3. one_plus_exp = pypto.add(exp_neg_x, 1.0)  # 1 + exp(-x)
4. sigmoid_x = pypto.reciprocal(one_plus_exp)  # 1 / (1 + exp(-x))
5. y = pypto.mul(x, sigmoid_x)       # x * sigmoid(x)
```

---

## 3. API 映射

### 核心 API 列表

| 计算步骤 | PyPTO API | 功能 | 文档路径 |
|---------|-----------|------|---------|
| sigmoid(x) | `pypto.sigmoid(x)` | 计算 sigmoid | docs/api/operation/pypto-sigmoid.md |
| 乘法 | `pypto.mul(x, y)` | 逐元素乘法 | docs/api/operation/pypto-mul.md |
| 取负 | `pypto.mul(x, -1.0)` | 乘以 -1 | docs/api/operation/pypto-mul.md |
| 指数 | `pypto.exp(x)` | 计算 exp(x) | docs/api/operation/pypto-exp.md |
| 加法 | `pypto.add(x, y)` | 逐元素加法 | docs/api/operation/pypto-add.md |
| 倒数 | `pypto.reciprocal(x)` | 计算 1/x | docs/api/operation/pypto-reciprocal.md |

### dtype 支持矩阵

| API | FP32 | FP16 | BF16 | INT32 | 约束来源 |
|-----|------|------|------|-------|---------|
| `pypto.sigmoid` | ✅ | ❌ | ❌ | ❌ | pypto-sigmoid.md:29 |
| `pypto.mul` | ✅ | ✅ | ✅ | ✅ | pypto-mul.md:29-30 |
| `pypto.exp` | ✅ | ✅ | ✅ | ❌ | pypto-exp.md:25 |
| `pypto.add` | ✅ | ✅ | ✅ | ✅ | pypto-add.md:29-30 |
| `pypto.reciprocal` | ✅ | ✅ | ✅ | ❌ | pypto-reciprocal.md:25 |

**dtype 策略**:
- **FP32**: 可使用方案 A（直接 sigmoid）或方案 B（手动展开）
- **FP16/BF16**: 必须使用方案 B（手动展开）

---

## 4. 约束检查

### 4.1 入口约束 (from_torch)

| 约束项 | 要求 | 证据来源 |
|--------|------|---------|
| dtype 支持 | torch.float16, torch.bfloat16, torch.float32 | pypto-from_torch.md:41-43 |
| 内存连续性 | tensor.is_contiguous() == True | pypto-from_torch.md:39 |
| 空Tensor | 不支持 | pypto-from_torch.md:39 |

### 4.2 API 约束

#### pypto.sigmoid 约束
- **dtype**: 仅支持 DT_FP32 ❌ **限制**
- **shape**: 2-4维
- **shape size**: ≤ INT32_MAX

#### pypto.mul 约束
- **dtype**: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32 ✅
- **shape**: 2-4维
- **shape size**: ≤ INT32_MAX
- **广播**: 支持倒数第二轴广播自动 inline
- **特殊值**: other 不支持 nan、inf

#### pypto.exp 约束
- **dtype**: DT_FP16, DT_BF16, DT_FP32 ✅
- **shape**: 2-4维
- **shape size**: ≤ INT32_MAX

#### pypto.add 约束
- **dtype**: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32 ✅
- **shape**: 2-4维
- **shape size**: ≤ INT32_MAX
- **广播**: 支持多维度广播
- **特殊值**: other 不支持 nan、inf

#### pypto.reciprocal 约束
- **dtype**: DT_FP16, DT_BF16, DT_FP32 ✅
- **shape**: 2-4维
- **shape size**: ≤ INT32_MAX

### 4.3 Tiling 约束

**算子类型**: Vector（纯 element-wise 操作）

**必需配置**: `pypto.set_vec_tile_shapes(*args)`

**约束**:
- 每个维度必须 > 0
- 最多 4 个维度
- TileShape 维度应与输出 shape 一致

**示例配置**:
```python
# 对于 shape [m, n]
pypto.set_vec_tile_shapes(m1, n1)  # m1, n1 分别切分 m, n 轴

# 对于 shape [b, s, n, d]
pypto.set_vec_tile_shapes(b1, s1, n1, d1)
```

---

## 5. 实现方案

### 推荐方案：条件分支 + 手动展开

```python
@pypto.frontend.jit
def silu_kernel(x: pypto.Tensor(), out: pypto.Tensor()):
    """SiLU activation: x * sigmoid(x)"""
    configure_tiling(x)

    if x.dtype == pypto.DataType.DT_FP32:
        # 方案 A：直接使用 sigmoid API（FP32 only）
        sigmoid_x = pypto.sigmoid(x)
        out[:] = pypto.mul(x, sigmoid_x)
    else:
        # 方案 B：手动展开 sigmoid（支持 FP16/BF16）
        neg_x = pypto.mul(x, -1.0)
        exp_neg_x = pypto.exp(neg_x)
        one_plus_exp = pypto.add(exp_neg_x, 1.0)
        sigmoid_x = pypto.reciprocal(one_plus_exp)
        out[:] = pypto.mul(x, sigmoid_x)
```

### 方案优势

1. **完整 dtype 支持**: FP32/FP16/BF16 全覆盖
2. **性能优化**: FP32 路径使用优化的 sigmoid API
3. **精度保证**: 手动展开路径使用标准数值算法
4. **代码清晰**: 条件分支明确表达实现逻辑

---

## 6. 参考实现

### 最佳匹配参考

**文件**: `examples/02_intermediate/operators/activation/activation.py`

**路径**: `/workspace/code/pypto/examples/02_intermediate/operators/activation/activation.py`

**相似度**: ⭐⭐⭐⭐⭐ (完全匹配)

**置信度**: 高

**可复用点**:
1. ✅ silu_activation_kernel 函数完整实现 (行 110-123)
2. ✅ silu_golden 参考实现 (行 89-91)
3. ✅ configure_tiling 配置函数 (行 80-86)
4. ✅ test_silu 测试函数 (行 126-149)
5. ✅ bfloat16 dtype 支持
6. ✅ NPU/SIM 双模式支持

**核心代码片段**:
```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def silu_activation_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor()):
    """
    SiLU (Swish) activation function: x * sigmoid(x)
    """
    configure_tiling(x)
    out[:] = x * pypto.sigmoid(x)
```

**注意事项**:
- 参考实现直接使用 `pypto.sigmoid`，仅支持 FP32/BF16（文档说明 sigmoid 仅支持 FP32）
- 对于完整 FP16 支持，建议使用手动展开方案

### 其他参考

**文件**: `pypto/examples/language/intermediate/ffn_activations.py`

**路径**: `/workspace/code/pypto/pypto/examples/language/intermediate/ffn_activations.py`

**相似度**: ⭐⭐⭐⭐ (高度相关)

**置信度**: 高

**可复用点**:
1. ✅ swiglu_kernel 中的 silu 手动展开实现 (行 98-115)
2. ✅ 使用 exp/add/reciprocal/mul 组合
3. ✅ PyPTO language DSL 完整示例

**核心代码片段**:
```python
@pl.function(type=pl.FunctionType.InCore)
def swiglu_kernel(self, gate, up, output):
    """Vector InCore: apply SwiGLU activation — gate * sigmoid(gate) * up."""
    tile_gate = pl.load(gate, [0, 0], [64, 64])
    tile_up = pl.load(up, [0, 0], [64, 64])
    gate_neg = pl.mul(tile_gate, -1.0)
    exp_neg = pl.exp(gate_neg)
    denom = pl.add(exp_neg, 1.0)
    sigmoid = pl.recip(denom)
    swish = pl.mul(tile_gate, sigmoid)
    result = pl.mul(swish, tile_up)
    out = pl.store(result, [0, 0], output)
    return out
```

---

## 7. 风险点与缓解

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| pypto.sigmoid 仅支持 FP32 | ⚠️ 中 | 使用手动展开方案支持 FP16/BF16 |
| 大 shape 的 tiling 配置 | ℹ️ 低 | 参考实现已有 configure_tiling 函数 |
| 数值精度（exp 大值） | ℹ️ 低 | PyPTO exp API 已处理边界情况 |
| 广播场景 | ℹ️ 低 | silu 为单输入，无广播需求 |

---

## 8. 证据索引

### API 文档

| API | 文档路径 | 关键约束 |
|-----|---------|---------|
| pypto.sigmoid | docs/api/operation/pypto-sigmoid.md | dtype: FP32 only |
| pypto.mul | docs/api/operation/pypto-mul.md | dtype: FP16/BF16/FP32/INT16/INT32 |
| pypto.exp | docs/api/operation/pypto-exp.md | dtype: FP16/BF16/FP32 |
| pypto.add | docs/api/operation/pypto-add.md | dtype: FP16/BF16/FP32/INT16/INT32 |
| pypto.reciprocal | docs/api/operation/pypto-reciprocal.md | dtype: FP16/BF16/FP32 |
| pypto.from_torch | docs/api/others/pypto-from_torch.md | dtype: torch.float16/32/64, contiguous |
| set_vec_tile_shapes | docs/api/config/pypto-set_vec_tile_shapes.md | 每维 > 0，最多 4 维 |

### 参考实现

| 文件 | 路径 | 行号 |
|------|------|------|
| silu_activation_kernel | examples/02_intermediate/operators/activation/activation.py | 110-123 |
| silu_golden | examples/02_intermediate/operators/activation/activation.py | 89-91 |
| configure_tiling | examples/02_intermediate/operators/activation/activation.py | 80-86 |
| test_silu | examples/02_intermediate/operators/activation/activation.py | 126-149 |
| swiglu_kernel (手动展开) | pypto/examples/language/intermediate/ffn_activations.py | 98-115 |

---

## 9. 结论

### 可行性评估

✅ **算子完全可行**

**理由**:
1. ✅ PyPTO 提供完整的 element-wise API（sigmoid, mul, exp, add, reciprocal）
2. ✅ dtype 支持：FP32（直接 sigmoid）+ FP16/BF16（手动展开）
3. ✅ 存在高质量参考实现（examples/02_intermediate/operators/activation/activation.py）
4. ✅ 无特殊约束或限制

### 实现建议

1. **采用条件分支方案**：FP32 使用 `pypto.sigmoid`，FP16/BF16 使用手动展开
2. **复用参考实现**：`activation.py` 中的 `silu_activation_kernel` 和 `configure_tiling`
3. **Tiling 配置**：参考 `configure_tiling` 函数，根据 shape 维度动态设置
4. **测试覆盖**：复用 `test_silu` 的测试模式，覆盖 FP32/FP16/BF16

### 下一步

- ✅ Stage 2 完成，可进入 Stage 3（Golden 生成）
- 📝 建议 golden 实现使用 PyTorch `torch.nn.functional.silu`
- 📝 测试用例需覆盖 spec.md 中的典型配置（性能_P0、功能_P0/P1、边界_P0）

---

*生成时间: 2026-03-28*
*生成工具: pypto-api-explorer*
