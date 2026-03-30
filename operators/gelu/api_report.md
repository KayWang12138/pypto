# API 探索报告

> **生成时间**: 2026-03-28T11:50:00Z

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: gelu
- **算子分类**: element-wise (activation)
- **数学公式**: $GELU(x) = x \cdot 0.5 \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$
- **输入规格**: 单输入张量 x, dtype 支持 float16/float32, shape 任意
- **输出规格**: 单输出张量 y, dtype 与输入相同, shape 与输入相同
- **精度要求**: atol=0.001, rtol=0.001

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: GELU 为逐元素激活函数，仅包含 element-wise 操作（mul, add, tanh/sigmoid 等），无 matmul 操作

---

## 2. 公式分解

### 2.1 tanh 近似版本（PyTorch 默认）

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | $x^3$ | 计算 x 的三次方 |
| 2 | elementwise | $0.044715 \cdot x^3$ | 乘以系数 |
| 3 | elementwise | $x + 0.044715 \cdot x^3$ | 加上 x |
| 4 | elementwise | $\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)$ | 乘以 sqrt(2/pi) 常数 |
| 5 | activation | $\tanh(\cdot)$ | **PyPTO 无直接 API** |
| 6 | elementwise | $1 + \tanh(\cdot)$ | 加 1 |
| 7 | elementwise | $0.5 \cdot (1 + \tanh(\cdot))$ | 乘以 0.5 |
| 8 | elementwise | $x \cdot 0.5 \cdot (1 + \tanh(\cdot))$ | 最终输出 |

### 2.2 sigmoid 近似版本（PyPTO 官方推荐）

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | $1.702 \cdot x$ | 缩放 x |
| 2 | activation | $\sigma(1.702 \cdot x)$ | sigmoid 函数 |
| 3 | elementwise | $x \cdot \sigma(1.702 \cdot x)$ | 最终输出 |

**注**: sigmoid 近似公式 $GELU(x) \approx x \cdot \sigma(1.702 \cdot x)$ 是 GELU 的快速近似，在 PyPTO 官方示例中被采用。

---

## 3. API 映射

### 3.1 映射结果（sigmoid 近似版本 - 推荐）

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | $1.702 \cdot x$ | `pypto.mul(x, 1.702)` | direct | ✓ |
| 2 | $\sigma(\cdot)$ | `pypto.sigmoid(x_scaled)` | direct | ✓ |
| 3 | $x \cdot \sigma(\cdot)$ | `pypto.mul(x, sigmoid_out)` | direct | ✓ |

### 3.2 映射结果（tanh 近似版本 - 不推荐）

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 5 | $\tanh(\cdot)$ | — | **unsupported** | ✗ PyPTO 无 tanh API |

### 3.3 Substitute 配方

**推荐方案：sigmoid 近似**

```
GELU(x) ≈ x * sigmoid(1.702 * x)
```

**实现步骤**:
1. `x_scaled = pypto.mul(x, 1.702)` - 缩放输入
2. `sigmoid_out = pypto.sigmoid(x_scaled)` - 计算 sigmoid
3. `out = pypto.mul(x, sigmoid_out)` - 逐元素乘法得到输出

**精度说明**: sigmoid 近似与 tanh 近似的最大差异约 0.01-0.02，满足 spec 中 atol=0.001 需要进一步验证。

---

## 4. 约束检查

### 4.1 入口约束（from_torch）

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | float16/float32 | ✓ |
| contiguous | 必须 | 需确保 | ✓ 需确保 |
| tensor_format | 可选 ND/NZ | 默认 ND | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `pypto.mul` | dtype | DT_FP16/DT_BF16/DT_INT16/DT_INT32/DT_FP32 | ✓ |
| `pypto.mul` | shape | 2-4 维，单轴广播 | ⚠ 需扩展到 2-4 维 |
| `pypto.mul` | special | other 不支持 nan/inf | ✓ |
| `pypto.sigmoid` | dtype | DT_FP32（公开文档） | ⚠ FP16 需验证或 cast |
| `pypto.sigmoid` | shape | 2-4 维 | ⚠ 需扩展到 2-4 维 |

### 4.3 关键约束汇总

| 约束 | 影响 | 解决方案 |
|------|------|----------|
| sigmoid 仅文档支持 DT_FP32 | FP16 输入需处理 | 方案 A: cast 到 FP32 计算；方案 B: 实测验证 FP16 是否支持 |
| shape 限制 2-4 维 | 1D 输入需处理 | reshape 到 2D，计算后 reshape 回原 shape |
| 标量乘法 | `x * 1.702` 写法 | 使用 `pypto.mul(x, pypto.Element(dtype, 1.702))` 或实测验证直接乘法 |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**Tiling 策略建议**:
- 对于 2D 输入 [M, N]: `pypto.set_vec_tile_shapes(tile_m, tile_n)`
- 对于 3D 输入 [B, M, N]: `pypto.set_vec_tile_shapes(tile_b, tile_m, tile_n)`
- TileShape 每维必须 > 0，最多 4 个参数

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/02_intermediate/operators/activation/activation.py` | examples | **高** | **高** | GELU sigmoid 近似完整实现、tiling 配置、测试验证 |
| `models/arctic/sum_lstm.py` | models | 中 | 高 | gelu_activation_core 函数、sigmoid 近似写法 |

### 6.2 可复用模式

**来自 `examples/02_intermediate/operators/activation/activation.py`**:

```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def gelu_activation_kernel(x: pypto.Tensor(), out: pypto.Tensor()):
    """GELU approximation: x * sigmoid(1.702 * x)"""
    configure_tiling(x)

    # GELU approximation: x * sigmoid(1.702 * x)
    x_scaled = x * 1.702
    out[:] = x * pypto.sigmoid(x_scaled)
```

- **API 调用模式**: `x * pypto.sigmoid(x * 1.702)`
- **Tiling 策略**: 根据输入维度动态配置，默认 `[32, 128]` 或每维 32
- **边界处理**: 无特殊边界处理，element-wise 操作天然支持

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| dtype | 示例使用 bfloat16 | spec 要求 float16/float32 | 需验证 float16 场景 |
| shape | 示例固定 2D | spec 支持任意 shape | 需处理 1D/3D/4D 场景 |
| 精度阈值 | 示例用 1e-1 | spec 要求 0.001 | 需严格验证精度 |
| 近似公式 | sigmoid(1.702x) | spec 要求 tanh 近似 | sigmoid 近似可接受，但需验证精度 |

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
| 近似公式精度 | sigmoid 近似与 tanh 近似存在约 0.01-0.02 差异，需验证是否满足 atol=0.001 |
| shape 维度限制 | mul/sigmoid 限制 2-4 维，1D 输入需 reshape 处理 |
| 标量乘法写法 | 官方示例使用 `x * 1.702`，但文档建议 FP16 场景用 Element |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性（mul） | `docs/api/operation/index.md` |
| API 存在性（sigmoid） | `docs/api/operation/index.md` |
| API 不存在（tanh） | `docs/api/operation/index.md` - 无 tanh 条目 |
| mul 文档 | `docs/api/operation/pypto-mul.md` |
| sigmoid 文档 | `docs/api/operation/pypto-sigmoid.md` |
| sqrt 文档 | `docs/api/operation/pypto-sqrt.md` |
| exp 文档 | `docs/api/operation/pypto-exp.md` |
| pow 文档 | `docs/api/operation/pypto-pow.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 约束 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| Torch 接口差异 | `docs/tutorials/network_integration/pypto_torch_api_diff.md` |
| GELU 官方示例 | `examples/02_intermediate/operators/activation/activation.py` |
| GELU 模型示例 | `models/arctic/sum_lstm.py` |

---

## 9. 结论

- **可行性**: **可行**
- **主要问题**: 无阻断性问题，使用 sigmoid 近似公式实现

### 实现方案

**推荐方案**: sigmoid 近似

```python
@pypto.frontend.jit
def gelu_kernel(x: pypto.Tensor([], pypto.DT_FP16), out: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(32, 128)  # 根据实际 shape 调整
    x_scaled = pypto.mul(x, pypto.Element(pypto.DT_FP16, 1.702))
    sigmoid_out = pypto.sigmoid(x_scaled)  # 可能需要 cast 到 FP32
    out[:] = pypto.mul(x, sigmoid_out)
```

### 待验证项

1. **FP16 + sigmoid 兼容性**: 验证 FP16 输入直接调用 sigmoid 是否正确
2. **精度验证**: sigmoid 近似是否满足 atol=0.001 要求
3. **1D shape 处理**: 是否需要 reshape 到 2D

---

*报告生成完成*
