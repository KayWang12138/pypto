# API 探索报告

> **生成时间**: 2026-03-28T05:35:00Z
> **算子名称**: layer_norm

---

## 1. 概述

### 1.1 输入摘要

Layer Normalization 算子对输入张量的最后一个维度进行归一化处理，通过计算均值和方差进行标准化，然后应用可学习的缩放参数 gamma 和偏移参数 beta 进行仿射变换。

**数学公式**: $y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$

**输入规格**:
- x: [batch, seq_len, normalized_shape], float32/float16/bfloat16
- gamma (weight): [normalized_shape], float32/float16/bfloat16
- beta (bias): [normalized_shape], float32/float16/bfloat16
- eps: scalar, 默认 1e-5

**输出规格**:
- y: [batch, seq_len, normalized_shape], float32/float16/bfloat16

**动态轴**: batch, seq_len

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 算子仅包含逐元素运算（sub, mul, div, add）和归约运算（sum），无矩阵乘法操作，属于纯 Vector 类型算子。

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | reduction | mean = reduce_mean(x, axis=-1, keepdims=True) | 计算最后一个维度的均值 |
| 2 | elementwise | centered = x - mean | 中心化 |
| 3 | elementwise | squared = centered * centered | 计算平方 |
| 4 | reduction | var = reduce_mean(squared, axis=-1, keepdims=True) | 计算方差 |
| 5 | elementwise | std = sqrt(var + eps) | 计算标准差 |
| 6 | elementwise | normalized = centered / std | 归一化 |
| 7 | elementwise | scaled = normalized * gamma | 缩放 |
| 8 | elementwise | output = scaled + beta | 偏移 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1a | reduce_mean(x, axis=-1) | `pypto.sum(x, dim=-1, keepdim=True)` | substitute | ✓ |
| 1b | mean / hidden_size | `pypto.mul(sum_result, 1.0/hidden_size)` | substitute | ✓ |
| 2 | x - mean | `pypto.sub(x, mean)` | direct | ✓ |
| 3 | centered * centered | `pypto.mul(centered, centered)` | direct | ✓ |
| 4a | reduce_mean(squared, axis=-1) | `pypto.sum(squared, dim=-1, keepdim=True)` | substitute | ✓ |
| 4b | var / hidden_size | `pypto.mul(sum_result, 1.0/hidden_size)` | substitute | ✓ |
| 5 | sqrt(var + eps) | `pypto.sqrt(pypto.add(var, eps))` | direct | ✓ |
| 6 | centered / std | `pypto.div(centered, std)` (或用 mul + rsqrt) | direct | ✓ |
| 7 | normalized * gamma | `pypto.mul(normalized, gamma)` | direct | ✓ |
| 8 | scaled + beta | `pypto.add(scaled, beta)` | direct | ✓ |

### 3.2 Substitute 配方

```
reduce_mean: pypto.sum(x, dim=-1, keepdim=True) * (1.0 / dim_size)
# PyPTO 无直接的 mean API，使用 sum + mul 组合实现
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32 | FP32/FP16/BF16 | ✓ |
| contiguous | 必须 | 需确保 | 需确保 |
| shape | 非空 Tensor | [batch, seq_len, normalized_shape] | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.sum | dtype | DT_FP32, DT_INT32, DT_INT16 | ✓ (FP32 支持) |
| pypto.sum | shape | 2-4 维, Shape Size <= INT32_MAX | ✓ |
| pypto.sum | 尾轴 | 32 bytes 对齐 | ⚠ 需检查 |
| pypto.mul | dtype | DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32 | ✓ |
| pypto.mul | shape | 2-4 维, 支持广播 | ✓ |
| pypto.sub | dtype | DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32 | ✓ |
| pypto.sqrt | dtype | DT_FP16, DT_BF16, DT_FP32 | ✓ |
| pypto.div | dtype | 同 mul | ✓ |
| pypto.add | dtype | DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32 | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**Tiling 策略建议**:
- 对于 3D 输入 [batch, seq_len, hidden_size]，建议 TileShape 设置为 `(batch_tile, seq_len_tile, hidden_size)`
- 参考实现使用 `pypto.set_vec_tile_shapes(64, 128)` 针对 2D 输入
- 需要确保 hidden_size 维度能被完整处理以保证归约精度
- 尾轴需要 32 bytes 对齐

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` | examples | 高 | 高 | 完整 LayerNorm 实现、Tiling 配置、精度验证 |
| `models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py` | models | 中 | 高 | RMSNorm 实现、FP32 中间精度、动态 shape 处理 |

### 6.2 可复用模式

**从 `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` 提取**:

- **API 调用模式**:
  ```python
  def layernorm_core(x, gamma, beta, eps, hidden_size):
      # Compute mean
      mean = pypto.sum(x, dim=-1, keepdim=True)
      mean = mean / hidden_size  # 或用 mul

      centered = x - mean

      # Compute variance
      squared = centered * centered
      var = pypto.sum(squared, dim=-1, keepdim=True)
      var = var / hidden_size

      # Normalize
      var_eps = var + eps
      std = pypto.sqrt(var_eps)
      normalized = centered / std

      # Affine transform
      scaled = normalized * gamma
      return scaled + beta
  ```

- **Tiling 策略**: `pypto.set_vec_tile_shapes(64, 128)` 针对 2D [batch, hidden_size]

- **JIT 装饰器**:
  ```python
  @pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
  def layer_norm_kernel(x, gamma, beta, output, config):
      ...
      pypto.assemble(out, [0, 0], output)
  ```

**从 `models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py` 提取**:

- **FP32 中间精度**: 在 RMSNorm 计算中使用 FP32 保证数值稳定性
  ```python
  add_out = pypto.cast(residual_tile, pypto.DT_FP32)
  # ... 归一化计算 ...
  in_tensor_norm = pypto.cast(res_add, in_tensor.dtype)  # 转回原类型
  ```

- **动态 shape 处理**:
  ```python
  in_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
  ```

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 输入维度 | 2D [batch, hidden_size] | 3D [batch, seq_len, hidden_size] | 扩展 Tiling 到 3D，循环处理 seq_len 维度 |
| 动态轴 | 静态 batch | 动态 batch, seq_len | 使用 `dynamic_axis=[0, 1]` 标记 |
| 精度处理 | BF16 输入输出 | FP32/FP16/BF16 | 添加 dtype 参数，内部用 FP32 计算 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | 所有必需 API 均可用 | - |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 精度稳定性 | 建议中间计算使用 FP32，最后转回原 dtype，避免数值溢出 |
| 尾轴对齐 | sum 操作要求尾轴 32 bytes 对齐，需检查 normalized_shape 是否满足 |
| 动态 shape | 需要使用 `pypto.from_torch(x, dynamic_axis=[0, 1])` 标记动态轴 |
| Tiling 设置 | 对于 3D 输入，需要合理设置 TileShape 以平衡计算效率和内存访问 |
| 广播处理 | gamma/beta 为 1D，需要广播到 3D，PyPTO 支持自动广播 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 列表 | `docs/api/operation/index.md` |
| pypto.sum 文档 | `docs/api/operation/pypto-sum.md` |
| pypto.mul 文档 | `docs/api/operation/pypto-mul.md` |
| pypto.sub 文档 | `docs/api/operation/pypto-sub.md` |
| pypto.add 文档 | `docs/api/operation/pypto-add.md` |
| pypto.sqrt 文档 | `docs/api/operation/pypto-sqrt.md` |
| pypto.rms_norm 文档 | `docs/api/operation/pypto-rms_norm.md` |
| pypto.var 文档 | `docs/api/operation/pypto-var.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| LayerNorm 参考实现 | `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` |
| RMSNorm 参考实现 | `models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无阻断问题，所有必需 API 均已支持
- **实现建议**:
  1. 参考 `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` 的 `layernorm_core` 函数结构
  2. 使用 `pypto.sum` + `mul` 组合实现 mean 计算
  3. 中间计算使用 FP32 保证数值稳定性
  4. 支持动态 batch 和 seq_len 维度
  5. 合理设置 Tiling 策略以优化性能

- **精度策略**: 内部使用 FP32 计算，输入输出支持 FP32/FP16/BF16

- **性能优化方向**:
  1. 考虑使用 `pypto.rsqrt` 替代 `sqrt + div` 组合，减少一次运算
  2. 优化 Tiling 配置以提高向量化利用率
