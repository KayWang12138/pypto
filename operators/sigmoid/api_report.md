# Sigmoid 算子 API 探索报告

> **生成时间**: 2026-03-29

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: sigmoid
- **算子分类**: element-wise
- **数学公式**: sigmoid(x) = 1 / (1 + exp(-x))
- **功能描述**: Sigmoid 激活函数，将输入逐元素映射到 (0, 1) 区间
- **输入 Shape**: [m, n] 或 [b, m, n] 或 [b, s, m, n]（2-4维）
- **数据类型**: DT_FP16/DT_FP32/DT_BF16
- **精度要求**: atol=3e-3, rtol=3e-3

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: Sigmoid 是纯逐元素操作（element-wise），不涉及矩阵乘法，仅需要对每个输入元素执行 sigmoid 计算，因此使用 Vector 类型的 tiling 配置。

---

## 2. 公式分解

### 2.1 原始公式

```
sigmoid(x) = 1 / (1 + exp(-x))
```

### 2.2 计算步骤分解

**方案 A - 直接 API（仅 FP32）**:
```
y = pypto.sigmoid(x)
```

**方案 B - 组合实现（支持 FP16/BF16/FP32）**:
```
1. neg_x = pypto.mul(x, -1.0)              # -x
2. exp_neg_x = pypto.exp(neg_x)            # e^(-x)
3. denominator = pypto.add(exp_neg_x, 1.0) # 1 + e^(-x)
4. y = pypto.div(1.0, denominator)         # 1 / (1 + e^(-x))
```

---

## 3. API 映射

### 3.1 核心 API 列表

| 计算步骤 | PyPTO API | 功能 | 文档路径 |
|---------|-----------|------|---------|
| sigmoid(x) | `pypto.sigmoid(x)` | 直接计算 sigmoid | docs/api/operation/pypto-sigmoid.md |
| 取负 | `pypto.mul(x, -1.0)` | 乘以 -1 | docs/api/operation/pypto-mul.md |
| exp | `pypto.exp(x)` | 计算 e^x | docs/api/operation/pypto-exp.md |
| 加法 | `pypto.add(a, b)` | 逐元素加法 | docs/api/operation/pypto-add.md |
| 除法 | `pypto.div(a, b)` | 逐元素除法 | docs/api/operation/pypto-div.md |

### 3.2 dtype 支持矩阵

| API | FP32 | FP16 | BF16 | 约束来源 |
|-----|------|------|------|---------|
| `pypto.sigmoid` | Y | N | N | pypto-sigmoid.md: 仅支持 DT_FP32 |
| `pypto.exp` | Y | Y | Y | pypto-exp.md:25 |
| `pypto.mul` | Y | Y | Y | pypto-mul.md:29-30 |
| `pypto.add` | Y | Y | Y | pypto-add.md:29-30 |
| `pypto.div` | Y | Y | Y | pypto-div.md:29-30 |

### 3.3 实现策略选择

**推荐方案**: 直接使用 `pypto.sigmoid(x)`

**理由**:
1. PyPTO 提供了直接的 `pypto.sigmoid` API
2. 文档显示只支持 DT_FP32，但根据 `pypto.relu` 的模式（文档也说只支持 FP32，实际可能支持更多），实际测试可能支持 FP16/BF16
3. 如果直接 API 只支持 FP32，可采用组合方案（exp/add/div）支持 FP16/BF16

**备选方案**: 如果 dtype 测试失败，使用组合实现
```python
# sigmoid(x) = 1 / (1 + exp(-x))
neg_x = pypto.mul(x, -1.0)
exp_neg_x = pypto.exp(neg_x)
denominator = pypto.add(exp_neg_x, 1.0)
out[:] = pypto.div(1.0, denominator)
```

---

## 4. 约束检查

### 4.1 入口约束（from_torch）

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | FP16/FP32/BF16 | ✓ |
| contiguous | 必须 | — | 需确保 |
| 非空 Tensor | 必须 | spec 约束不支持空 Tensor | ✓ |

### 4.2 API 约束（pypto.sigmoid）

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.sigmoid | dtype | DT_FP32（文档声明） | ⚠ 需验证 FP16/BF16 |
| pypto.sigmoid | shape | 仅支持 2-4 维 | ✓ (spec: 2-4维) |
| pypto.sigmoid | shape size | ≤ INT32_MAX (2147483647) | ✓ (spec: ≤ INT32_MAX) |
| pypto.sigmoid | 空Tensor | 不支持 | ✓ (spec: 不支持空Tensor) |

### 4.3 API 约束（组合方案 - pypto.exp/add/div）

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.exp | dtype | DT_FP16, DT_BF16, DT_FP32 | ✓ |
| pypto.exp | shape | 仅支持 2-4 维 | ✓ |
| pypto.add | dtype | DT_FP16, DT_BF16, DT_FP32 | ✓ |
| pypto.div | dtype | DT_FP16, DT_BF16, DT_FP32 | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**TileShape 配置说明**（来自 pypto.sigmoid 文档）:
- TileShape 维度应和输出一致
- 示例：输入 input shape 为 [m, n]，输出为 [m, n]，TileShape 设置为 [m1, n1]，则 m1, n1 分别用于切分 m, n 轴
- 每个维度必须 > 0
- 最多 4 个 inputs

**推荐 TileShape**（参考 relu/tanh 实现）:
- 2D: `pypto.set_vec_tile_shapes(32, 128)`
- 3D: `pypto.set_vec_tile_shapes(1, 32, 128)`
- 4D: `pypto.set_vec_tile_shapes(1, 1, 32, 128)`

---

## 6. 参考实现

### 6.1 最佳匹配参考

**文件**: `examples/02_intermediate/operators/activation/activation.py`

**路径**: `/workspace/code/pypto/examples/02_intermediate/operators/activation/activation.py`

**相似度**: 高（直接使用 pypto.sigmoid）

**置信度**: 高

**可复用点**:
1. `pypto.sigmoid(x)` 的直接调用方式
2. `configure_tiling` 函数设计模式
3. golden 参考实现模式（使用 `torch.sigmoid`）
4. 测试函数结构
5. NPU/SIM 双模式支持

**核心代码片段**:
```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def sigmoid_activation_kernel(
    x: pypto.Tensor(),
    out: pypto.Tensor()):
    """Sigmoid activation function: 1 / (1 + exp(-x))"""
    configure_tiling(x)
    out[:] = pypto.sigmoid(x)
```

### 6.2 其他参考

**文件**: `operators/relu/` 和 `operators/tanh/`

**可复用点**:
1. 完整的算子目录结构
2. golden/impl/test 文件分离模式
3. 三态标记测试入口
4. wrapper 函数设计

---

## 7. 风险点与缓解

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| pypto.sigmoid 只支持 FP32 | 中 | 准备组合实现方案作为备选，使用 exp/add/div 组合 |
| exp 大值可能导致溢出 | 低 | PyPTO exp API 已处理边界情况，spec 允许 IEEE 754 标准处理 |
| 大 shape 的 tiling 配置 | 低 | 参考 relu/tanh 实现中的 configure_tiling 函数 |

---

## 8. 证据索引

### API 文档

| API | 文档路径 | 关键约束 |
|-----|---------|---------|
| pypto.sigmoid | docs/api/operation/pypto-sigmoid.md | dtype: DT_FP32（仅） |
| pypto.exp | docs/api/operation/pypto-exp.md | dtype: FP16/BF16/FP32 |
| pypto.mul | docs/api/operation/pypto-mul.md | dtype: FP16/BF16/FP32/INT16/INT32 |
| pypto.add | docs/api/operation/pypto-add.md | dtype: FP16/BF16/FP32/INT16/INT32 |
| pypto.div | docs/api/operation/pypto-div.md | dtype: FP16/BF16/FP32 |
| pypto.from_torch | docs/api/others/pypto-from_torch.md | dtype: torch.float16/32/64, contiguous |
| set_vec_tile_shapes | docs/api/config/pypto-set_vec_tile_shapes.md | 每维 > 0，最多 4 维 |

### 参考实现

| 文件 | 路径 | 可复用点 |
|------|------|---------|
| activation.py | examples/02_intermediate/operators/activation/activation.py | sigmoid kernel 模式、configure_tiling |
| relu | operators/relu/ | 完整算子目录结构 |
| tanh | operators/tanh/ | 组合 API 实现模式 |

---

## 9. 结论

### 可行性评估

**算子完全可行**

**理由**:
1. ✓ PyPTO 提供直接的 `pypto.sigmoid()` API
2. ⚠ 文档声明只支持 FP32，但存在备选组合方案支持 FP16/BF16
3. ✓ 存在高质量参考实现（activation.py）
4. ✓ 无特殊约束或限制

### 实现建议

1. **优先使用直接 API**: `pypto.sigmoid(x)`
2. **备选组合方案**: 如果 dtype 测试失败，使用 exp/mul/add/div 组合实现
3. **复用参考实现**: activation.py 中的 kernel 模式和 configure_tiling
4. **Tiling 配置**: 参考 relu/tanh 的 configure_tiling 函数，根据 shape 维度动态设置
5. **测试覆盖**: 复用 relu/tanh 的测试模式，覆盖 FP32/FP16/BF16

### 下一步

- ✓ Stage 2 完成，可进入 Stage 3（Golden 生成）
- 建议 golden 实现使用 PyTorch `torch.sigmoid`
- 测试用例需覆盖 spec.md 中的典型配置（性能_P0、功能_P0/P1/P2）

---

*生成时间: 2026-03-29*
*生成工具: pypto-op-orchestrator*
