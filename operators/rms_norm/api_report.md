# API 探索报告

> **生成时间**: 2026-03-28T00:00:00Z

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: rms_norm
- **数学公式**: y = x * γ / sqrt(mean(x^2) + ε)
- **输入规格**:
  - x: [b, s, d] float32/bfloat16，动态轴 b, s
  - weight (γ): [d] float32/bfloat16
  - eps: scalar float，默认 1e-6
- **输出规格**: y [b, s, d] float32/bfloat16，动态轴 b, s
- **归一化维度**: 最后一个维度 d

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 计算仅含逐元素操作（mul、div、sqrt）和归约操作（sum/mean），无矩阵乘法（matmul），因此属于 Vector 类型算子。

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | x² = x * x | 对输入求平方 |
| 2 | reduction | mean_sq = mean(x², dim=-1) | 在最后一个维度上求均值 |
| 3 | elementwise | rms_input = mean_sq + eps | 添加防止除零的小常数 |
| 4 | elementwise | rms = sqrt(rms_input) | 计算均方根 |
| 5 | elementwise | normalized = x / rms | 归一化（广播 rms） |
| 6 | elementwise | y = normalized * γ | 应用缩放参数（广播 γ） |

---

## 3. API 映射

### 3.1 映射结果

**方案 A：使用内置 API（推荐）**

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1-6 | y = x * γ / sqrt(mean(x²) + ε) | `pypto.rms_norm(x, gamma, epsilon)` | direct | ✓ |

**方案 B：使用基础 API 组合**

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | x² | `pypto.mul(x, x)` | direct | ✓ |
| 2 | mean(x²) | `pypto.sum(x², dim=-1) / hidden_size` | substitute | ✓ |
| 3 | + eps | `pypto.add(mean_sq, eps)` | direct | ✓ |
| 4 | sqrt() | `pypto.sqrt(mean_sq_eps)` | direct | ✓ |
| 5 | x / rms | `pypto.div(x, rms)` | direct | ✓ |
| 6 | * γ | `pypto.mul(normalized, gamma)` | direct | ✓ |

### 3.2 Substitute 配方

**mean 操作（PyPTO 无直接 mean API）**：
```
mean(x, dim=-1, keepdim=True) = sum(x, dim=-1, keepdim=True) / dim_size
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | FP32/BF16 | ✓ |
| contiguous | 必须 | 需确保 | ⚠ 需确保输入连续 |
| tensor type | torch.Tensor | torch.Tensor | ✓ |

### 4.2 API 约束

**pypto.rms_norm API 约束**：

| 约束项 | 要求 | 结果 |
|--------|------|------|
| input shape | [..., C] 任意 shape | ✓ 支持 [b, s, d] |
| gamma shape | [C] 与 input 最后一维相同 | ✓ 支持 [d] |
| epsilon | float，默认 1e-6 | ✓ 支持 |
| dtype | PyPto 支持的数据类型 | ✓ FP32/BF16 |

### 4.3 动态轴约束

| 约束项 | 要求 | 结果 |
|--------|------|------|
| 动态轴标记 | 通过 `dynamic_axis` 参数指定 | ✓ 支持 b=0, s=1 |
| 动态 shape | 使用 -1 或 `dynamic_axis` | ✓ 均支持 |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API | 说明 |
|----------|-----------|------|
| Vector | `pypto.set_vec_tile_shapes()` | 设置 Vector 计算的 TileShape |

**推荐 Tiling 配置**（基于参考实现）：
```python
pypto.set_vec_tile_shapes(64, 128)  # 适用于 [batch, hidden_size] 场景
```

**动态轴 Tiling 注意事项**：
- 动态维度（b, s）在编译时未知，需在运行时确定
- TileShape 需要合理设置以平衡并行度和内存访问

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `pypto/examples/language/intermediate/rms_norm.py` | examples | 高 | 高 | PyPTO Language DSL 实现模式 |
| `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` | examples | 高 | 高 | pypto.rms_norm API 调用、测试框架 |

### 6.2 可复用模式

**API 调用模式（方案 A - 推荐）**：
```python
@pypto.frontend.jit(runtime_options={"run_mode": run_mode})
def rms_norm_kernel(x, gamma, output, config):
    eps = config.eps
    pypto.set_vec_tile_shapes(64, 128)
    out = pypto.rms_norm(x, gamma, eps)
    pypto.assemble(out, [0, 0], output)
```

**API 调用模式（方案 B - 基础 API 组合）**：
```python
def rms_norm_core(x, gamma, eps, hidden_size):
    squared = x * x
    mean_sq = pypto.sum(squared, dim=-1, keepdim=True)
    mean_sq = mean_sq / hidden_size
    rms = pypto.sqrt(mean_sq + eps)
    normalized = x / rms
    return normalized * gamma
```

**Tiling 策略**：
- 使用 `pypto.set_vec_tile_shapes(64, 128)` 配置
- 根据 hidden_size 调整 tiling 参数

**动态轴处理**：
```python
# 方式 1：使用 -1 标记动态维度
pto_input = pypto.tensor([-1, -1, hidden_size], dtype, "x")

# 方式 2：使用 from_torch 的 dynamic_axis 参数
pto_input = pypto.from_torch(x_torch, "x", dynamic_axis=[0, 1])
```

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 输入维度 | 示例为 [32, 64] 二维 | 需支持 [b, s, d] 三维 | 使用动态轴支持 b, s |
| 动态 shape | 示例为静态 shape | 需支持动态 batch 和 seq_len | 使用 `dynamic_axis=[0, 1]` |
| eps 值 | 示例使用 1e-5 | spec 要求 1e-6 | 通过 config.eps 参数传入 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | PyPTO 内置 `pypto.rms_norm` API 直接支持 | 直接使用内置 API |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 动态轴支持 | 必须通过 `dynamic_axis` 参数标记动态维度（batch=0, seq_len=1） |
| contiguous 要求 | 输入 torch.Tensor 必须是连续的（`tensor.is_contiguous() == True`） |
| dtype 一致性 | x 和 gamma 的 dtype 必须一致 |
| 精度要求 | bfloat16 使用 atol=0.01, rtol=0.01；float32 使用 atol=0.001, rtol=0.001 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` (第 81 行) |
| rms_norm API 文档 | `docs/api/operation/pypto-rms_norm.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Vector Tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| DataType 枚举 | `docs/api/datatype/DataType.md` |
| 动态轴处理 | `docs/tutorials/development/tensor_creation.md` |
| 参考实现 1 | `pypto/examples/language/intermediate/rms_norm.py` |
| 参考实现 2 | `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` |
| 测试用例 | `pypto/tests/st/examples/02_intermediate/test_rms_norm.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无阻断问题
- **推荐方案**: 使用 PyPTO 内置 `pypto.rms_norm` API，配合 `dynamic_axis` 支持动态 shape
- **实现建议**:
  1. 优先使用 `pypto.rms_norm(x, gamma, eps)` 内置 API
  2. 通过 `pypto.from_torch(..., dynamic_axis=[0, 1])` 支持动态 batch 和 seq_len
  3. 使用 `pypto.set_vec_tile_shapes()` 配置 Tiling 参数
  4. 参考 `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` 的实现模式
