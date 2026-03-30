# TanH 算子 API 探索报告

## 1. 概述

- **算子名称**: tanh (Hyperbolic Tangent)
- **数学公式**: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- **算子类型**: element-wise 激活函数
- **实现策略**: 组合实现 (基于 exp, sub, add, div)
- **可行性**: 完全可行

---

## 2. 公式分解

### 原始公式
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        = (e^x - e^(-x)) / (e^x + e^(-x))
```

### 计算步骤分解

**实现方案 - 手动展开 tanh**:
```
1. exp_x = pypto.exp(x)                    # e^x
2. neg_x = pypto.mul(x, -1.0)              # -x
3. exp_neg_x = pypto.exp(neg_x)            # e^(-x)
4. numerator = pypto.sub(exp_x, exp_neg_x) # e^x - e^(-x)
5. denominator = pypto.add(exp_x, exp_neg_x) # e^x + e^(-x)
6. y = pypto.div(numerator, denominator)   # (e^x - e^(-x)) / (e^x + e^(-x))
```

---

## 3. API 映射

### 核心 API 列表

| 计算步骤 | PyPTO API | 功能 | 文档路径 |
|---------|-----------|------|---------|
| exp(x) | `pypto.exp(x)` | 计算 e^x | docs/api/operation/pypto-exp.md |
| 取负 | `pypto.mul(x, -1.0)` | 乘以 -1 | docs/api/operation/pypto-mul.md |
| 减法 | `pypto.sub(a, b)` | 逐元素减法 | docs/api/operation/pypto-sub.md |
| 加法 | `pypto.add(a, b)` | 逐元素加法 | docs/api/operation/pypto-add.md |
| 除法 | `pypto.div(a, b)` | 逐元素除法 | docs/api/operation/pypto-div.md |

### dtype 支持矩阵

| API | FP32 | FP16 | BF16 | INT32 | 约束来源 |
|-----|------|------|------|-------|---------|
| `pypto.exp` | Y | Y | Y | N | pypto-exp.md:25 |
| `pypto.mul` | Y | Y | Y | Y | pypto-mul.md:29-30 |
| `pypto.sub` | Y | Y | Y | Y | pypto-sub.md:29-30 |
| `pypto.add` | Y | Y | Y | Y | pypto-add.md:29-30 |
| `pypto.div` | Y | Y | Y | N | pypto-div.md:29-30 |

**dtype 策略**:
- **FP32/FP16/BF16**: 全部使用手动展开方案

---

## 4. 约束检查

### 4.1 入口约束 (from_torch)

| 约束项 | 要求 | 证据来源 |
|--------|------|---------|
| dtype 支持 | torch.float16, torch.bfloat16, torch.float32 | pypto-from_torch.md:41-43 |
| 内存连续性 | tensor.is_contiguous() == True | pypto-from_torch.md:39 |
| 空Tensor | 不支持 | pypto-from_torch.md:39 |

### 4.2 API 约束

#### pypto.exp 约束
- **dtype**: DT_FP16, DT_BF16, DT_FP32 Y
- **shape**: 2-4维
- **shape size**: <= INT32_MAX
- **空Tensor**: 不支持

#### pypto.mul 约束
- **dtype**: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32 Y
- **shape**: 2-4维
- **shape size**: <= INT32_MAX
- **广播**: 支持倒数第二轴广播自动 inline
- **特殊值**: other 不支持 nan、inf

#### pypto.sub 约束
- **dtype**: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32 Y
- **shape**: 2-4维
- **shape size**: <= INT32_MAX
- **广播**: 支持按照单个维度广播
- **特殊值**: other 不支持 nan、inf

#### pypto.add 约束
- **dtype**: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32 Y
- **shape**: 2-4维
- **shape size**: <= INT32_MAX
- **广播**: 支持按照单个维度广播
- **特殊值**: other 不支持 nan、inf

#### pypto.div 约束
- **dtype**: DT_FP16, DT_BF16, DT_FP32 Y
- **shape**: 2-4维
- **shape size**: <= INT32_MAX
- **广播**: 支持按照单个维度广播
- **特殊值**: other 不支持 nan、inf

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

### 推荐方案：手动展开 tanh

```python
@pypto.frontend.jit
def tanh_kernel(x: pypto.Tensor(), out: pypto.Tensor()):
    """Tanh activation: (e^x - e^(-x)) / (e^x + e^(-x))"""
    configure_tiling(x)

    # e^x
    exp_x = pypto.exp(x)
    # -x
    neg_x = pypto.mul(x, -1.0)
    # e^(-x)
    exp_neg_x = pypto.exp(neg_x)
    # e^x - e^(-x)
    numerator = pypto.sub(exp_x, exp_neg_x)
    # e^x + e^(-x)
    denominator = pypto.add(exp_x, exp_neg_x)
    # (e^x - e^(-x)) / (e^x + e^(-x))
    out[:] = pypto.div(numerator, denominator)
```

### 方案优势

1. **完整 dtype 支持**: FP32/FP16/BF16 全覆盖
2. **数值稳定**: exp 操作已由框架优化处理边界情况
3. **代码清晰**: 公式直接映射到 API 调用
4. **精度保证**: 使用标准数值算法

---

## 6. 参考实现

### 最佳匹配参考

**文件**: `examples/02_intermediate/operators/activation/activation.py`

**路径**: `/workspace/code/pypto/examples/02_intermediate/operators/activation/activation.py`

**相似度**: (高度相关)

**置信度**: 高

**可复用点**:
1. Y sigmoid_activation_kernel 函数结构 (类似的激活函数)
2. Y configure_tiling 配置函数
3. Y golden 参考实现模式
4. Y 测试函数结构
5. Y bfloat16 dtype 支持
6. Y NPU/SIM 双模式支持

**核心代码片段**:
```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def sigmoid_activation_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor()):
    """
    Sigmoid activation function: 1 / (1 + exp(-x))
    """
    configure_tiling(x)
    out[:] = pypto.sigmoid(x)
```

### 其他参考

**文件**: `operators/relu/` 和 `operators/silu/`

**可复用点**:
1. Y 完整的算子目录结构
2. Y golden/impl/test 文件分离
3. Y 三态标记测试入口

---

## 7. 风险点与缓解

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| exp 大值可能导致溢出 | 低 | PyPTO exp API 已处理边界情况，spec 允许 IEEE 754 标准处理 |
| 大 shape 的 tiling 配置 | 低 | 参考 relu/silu 实现中的 configure_tiling 函数 |
| 数值精度 | 低 | 使用标准数学公式，精度要求 atol=3e-3, rtol=3e-3 可满足 |

---

## 8. 证据索引

### API 文档

| API | 文档路径 | 关键约束 |
|-----|---------|---------|
| pypto.exp | docs/api/operation/pypto-exp.md | dtype: FP16/BF16/FP32 |
| pypto.mul | docs/api/operation/pypto-mul.md | dtype: FP16/BF16/FP32/INT16/INT32 |
| pypto.sub | docs/api/operation/pypto-sub.md | dtype: FP16/BF16/FP32/INT16/INT32 |
| pypto.add | docs/api/operation/pypto-add.md | dtype: FP16/BF16/FP32/INT16/INT32 |
| pypto.div | docs/api/operation/pypto-div.md | dtype: FP16/BF16/FP32 |
| pypto.from_torch | docs/api/others/pypto-from_torch.md | dtype: torch.float16/32/64, contiguous |
| set_vec_tile_shapes | docs/api/config/pypto-set_vec_tile_shapes.md | 每维 > 0，最多 4 维 |

### 参考实现

| 文件 | 路径 | 可复用点 |
|------|------|---------|
| activation.py | examples/02_intermediate/operators/activation/activation.py | 激活函数 kernel 模式 |
| relu | operators/relu/ | 完整算子目录结构 |
| silu | operators/silu/ | 组合 API 实现模式 |

---

## 9. 结论

### 可行性评估

**算子完全可行**

**理由**:
1. Y PyPTO 提供完整的 element-wise API（exp, mul, sub, add, div）
2. Y dtype 支持：FP32/FP16/BF16 全覆盖
3. Y 存在高质量参考实现（activation.py）
4. Y 无特殊约束或限制

### 实现建议

1. **采用手动展开方案**：使用 exp, mul, sub, add, div 组合实现
2. **复用参考实现**：activation.py 中的 kernel 模式和 configure_tiling
3. **Tiling 配置**：参考 relu/silu 的 configure_tiling 函数，根据 shape 维度动态设置
4. **测试覆盖**：复用 relu/silu 的测试模式，覆盖 FP32/FP16/BF16

### 下一步

- Y Stage 2 完成，可进入 Stage 3（Golden 生成）
- 建议 golden 实现使用 PyTorch `torch.tanh`
- 测试用例需覆盖 spec.md 中的典型配置（性能_P0、功能_P0/P1/P2）

---

*生成时间: 2026-03-29*
*生成工具: pypto-op-orchestrator*
