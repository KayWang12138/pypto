# API 探索报告

> **生成时间**: 2026-03-30T09:20:00Z

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: mish
- **算子分类**: element-wise (activation)
- **数学公式**: $mish(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$
- **输入规格**: 单输入张量 x, dtype 支持 float32, shape 1D-4D
- **输出规格**: 单输出张量 y, dtype 与输入相同, shape 与输入相同
- **精度要求**: atol=0.001, rtol=0.001
- **特殊要求**:
  - 支持动态轴，使用隐式 shape 推断 `pypto.Tensor([], pypto.DT_FP32)`
  - 4D 输入需 reshape 为 2D 避免 tiling 编译问题
  - 使用 `pypto.set_vec_tile_shapes(64, 128)` 用于 2D kernels

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: Mish 为逐元素激活函数，仅包含 element-wise 操作（exp, log, add, mul, sigmoid），无 matmul 操作

---

## 2. 公式分解

### 2.1 原始公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | $e^x$ | 计算 x 的指数 |
| 2 | elementwise | $1 + e^x$ | 加 1 |
| 3 | elementwise | $\ln(1 + e^x)$ | 自然对数，即 softplus(x) |
| 4 | activation | $\tanh(\text{softplus}(x))$ | 双曲正切，**PyPTO 无直接 API** |
| 5 | elementwise | $x \cdot \tanh(\text{softplus}(x))$ | 逐元素乘法得到输出 |

### 2.2 tanh 替代方案

由于 PyPTO 无直接 tanh API，使用数学恒等式：

$$\tanh(x) = 2 \cdot \sigma(2x) - 1$$

其中 $\sigma$ 为 sigmoid 函数。

**优化后的计算步骤**:

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | $e^x$ | 计算 x 的指数 |
| 2 | elementwise | $1 + e^x$ | 加 1 |
| 3 | elementwise | $\ln(1 + e^x)$ | 自然对数，即 softplus(x) |
| 4 | elementwise | $2 \cdot \text{softplus}(x)$ | softplus 乘以 2 |
| 5 | activation | $\sigma(2 \cdot \text{softplus}(x))$ | sigmoid 函数 |
| 6 | elementwise | $2 \cdot \sigma(\cdot) - 1$ | 得到 tanh(softplus(x)) |
| 7 | elementwise | $x \cdot \tanh(\cdot)$ | 逐元素乘法得到输出 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | $e^x$ | `pypto.exp(x)` | direct | ✓ |
| 2 | $1 + e^x$ | `pypto.add(exp_x, 1.0)` | direct | ✓ |
| 3 | $\ln(1 + e^x)$ | `pypto.log(one_plus_exp)` | direct | ✓ |
| 4 | $2 \cdot \text{softplus}(x)$ | `pypto.mul(softplus, 2.0)` | direct | ✓ |
| 5 | $\sigma(\cdot)$ | `pypto.sigmoid(sp_2x)` | direct | ✓ |
| 6 | $2 \cdot \sigma - 1$ | `pypto.add(pypto.mul(sigmoid, 2.0), -1.0)` | direct | ✓ |
| 7 | $x \cdot \tanh(\cdot)$ | `pypto.mul(x, tanh_out)` | direct | ✓ |

### 3.2 完整实现配方

```python
def mish_core(x: pypto.Tensor) -> pypto.Tensor:
    """Mish 核心计算: mish(x) = x * tanh(softplus(x))"""
    # Step 1-3: softplus(x) = ln(1 + exp(x))
    exp_x = pypto.exp(x)
    one_plus_exp = pypto.add(exp_x, 1.0)
    softplus = pypto.log(one_plus_exp)

    # Step 4-6: tanh(softplus) = 2 * sigmoid(2 * softplus) - 1
    sp_2x = pypto.mul(softplus, 2.0)
    sigmoid_2x = pypto.sigmoid(sp_2x)
    tanh_out = pypto.add(pypto.mul(sigmoid_2x, 2.0), -1.0)

    # Step 7: x * tanh(softplus(x))
    result = pypto.mul(x, tanh_out)

    return result
```

---

## 4. 约束检查

### 4.1 入口约束（from_torch）

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | float32 | ✓ |
| contiguous | 必须 | 需确保 | ✓ 需确保 |
| tensor_format | 可选 ND/NZ | 默认 ND | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `pypto.exp` | dtype | DT_FP16/DT_BF16/DT_FP32 | ✓ |
| `pypto.exp` | shape | 2-4 维 | ⚠ 1D 需 reshape |
| `pypto.exp` | shape_size | ≤ INT32_MAX | ✓ |
| `pypto.log` | dtype | DT_FP32/DT_FP16/DT_BF16 | ✓ |
| `pypto.log` | shape | 1-4 维 | ✓ |
| `pypto.add` | dtype | DT_FP16/DT_BF16/DT_INT16/DT_INT32/DT_FP32 | ✓ |
| `pypto.add` | shape | 2-4 维 | ⚠ 1D 需 reshape |
| `pypto.mul` | dtype | DT_FP16/DT_BF16/DT_INT16/DT_INT32/DT_FP32 | ✓ |
| `pypto.mul` | shape | 2-4 维 | ⚠ 1D 需 reshape |
| `pypto.sigmoid` | dtype | DT_FP32（公开文档） | ⚠ FP16 需验证或 cast |

### 4.3 关键约束汇总

| 约束 | 影响 | 解决方案 |
|------|------|----------|
| exp/mul/add 限制 2-4 维 | 1D 输入需处理 | 1D 输入 reshape 为 2D |
| 4D 输入 tiling 编译问题 | 4D 输入需处理 | 4D 输入 flatten 为 2D 计算 |
| sigmoid 仅文档支持 DT_FP32 | FP16 需处理 | 实测验证或 cast 到 FP32 |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**Tiling 策略**:
- 根据用户需求，使用 `pypto.set_vec_tile_shapes(64, 128)` 用于 2D kernels
- 对于非 2D 输入，先 reshape 为 2D 再计算

**推荐实现**:
```python
def configure_tiling():
    """配置 TileShape，使用用户指定的 64, 128"""
    pypto.set_vec_tile_shapes(64, 128)
```

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/02_intermediate/operators/activation/activation.py` | examples | **高** | **高** | GELU/SiLU 实现、sigmoid 近似 tanh、tiling 配置 |
| `operators/gelu/gelu_impl.py` | operators | **高** | **高** | tanh 近似实现、动态 tiling、wrapper 模式 |
| `operators/gelu/api_report.md` | operators | **高** | **高** | API 映射方法、约束分析 |

### 6.2 可复用模式

**来自 `operators/gelu/gelu_impl.py`**:

```python
# tanh 近似实现 (line 72-75)
inner_2x = pypto.mul(inner, 2.0)
sigmoid_2x = pypto.sigmoid(inner_2x)
tanh_out = pypto.add(pypto.mul(sigmoid_2x, 2.0), -1.0)
```

- **API 调用模式**: tanh(x) = 2 * sigmoid(2x) - 1
- **Tiling 策略**: 根据输入维度动态配置
- **Wrapper 模式**: 检查连续性、根据 dtype 调用 kernel

**来自 `examples/02_intermediate/operators/activation/activation.py`**:

```python
def configure_tiling(x):
    if len(x.shape) >= 2:
        tile_list = [32 for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_list)
    else:
        pypto.set_vec_tile_shapes(32, 128)
```

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| Tiling 配置 | 动态配置 32 | 用户指定 64, 128 | 使用固定 `set_vec_tile_shapes(64, 128)` |
| 4D 处理 | 未特殊处理 | 需 reshape 为 2D | 添加 4D → 2D reshape 逻辑 |
| 动态轴 | 未使用隐式推断 | 需隐式 shape | 使用 `pypto.Tensor([], pypto.DT_FP32)` |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | 无阻断性问题 | — |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| sigmoid dtype 限制 | 公开文档仅标注 DT_FP32，FP16 需实测验证或先 cast |
| 4D tiling 编译问题 | 需将 4D 输入 reshape 为 2D 避免编译问题 |
| 1D 输入限制 | exp/mul/add 限制 2-4 维，1D 需 reshape |
| 数值稳定性 | exp(x) 可能溢出，但 softplus 计算通常稳定 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性（exp） | `docs/api/operation/index.md` |
| API 存在性（log） | `docs/api/operation/index.md` |
| API 存在性（mul） | `docs/api/operation/index.md` |
| API 存在性（add） | `docs/api/operation/index.md` |
| API 存在性（sigmoid） | `docs/api/operation/index.md` |
| API 不存在（tanh） | `docs/api/operation/index.md` - 无 tanh 条目 |
| exp 文档 | `docs/api/operation/pypto-exp.md` |
| log 文档 | `docs/api/operation/pypto-log.md` |
| mul 文档 | `docs/api/operation/pypto-mul.md` |
| add 文档 | `docs/api/operation/pypto-add.md` |
| sigmoid 文档 | `docs/api/operation/pypto-sigmoid.md` |
| reshape 文档 | `docs/api/operation/pypto-reshape.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 约束 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| Torch 接口差异 | `docs/tutorials/network_integration/pypto_torch_api_diff.md` |
| GELU 参考实现 | `operators/gelu/gelu_impl.py` |
| 激活函数示例 | `examples/02_intermediate/operators/activation/activation.py` |

---

## 9. 结论

- **可行性**: **可行**
- **主要问题**: 无阻断性问题，所有操作均有对应 PyPTO API

### 实现方案

**推荐方案**: 使用 tanh 的 sigmoid 近似

```python
@pypto.frontend.jit
def mish_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    """Mish JIT kernel for FP32."""
    pypto.set_vec_tile_shapes(64, 128)

    # softplus(x) = ln(1 + exp(x))
    exp_x = pypto.exp(x)
    one_plus_exp = pypto.add(exp_x, 1.0)
    softplus = pypto.log(one_plus_exp)

    # tanh(softplus) = 2 * sigmoid(2 * softplus) - 1
    sp_2x = pypto.mul(softplus, 2.0)
    sigmoid_2x = pypto.sigmoid(sp_2x)
    tanh_out = pypto.add(pypto.mul(sigmoid_2x, 2.0), -1.0)

    # mish(x) = x * tanh(softplus(x))
    out[:] = pypto.mul(x, tanh_out)
```

### 待验证项

1. **4D reshape**: 验证 4D → 2D reshape 后精度是否满足
2. **1D 处理**: 验证 1D 输入 reshape 为 2D 后的正确性
3. **性能验证**: 使用 64, 128 tiling 配置的性能表现

---

*报告生成完成*
