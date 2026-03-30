# API 探索报告

> **生成时间**: 2026-03-29T14:15:00Z
> **算子名称**: batchnorm

---

## 1. 概述

### 1.1 输入摘要

Batch Normalization 算子对输入张量在通道维度上进行归一化处理。训练时使用当前 batch 的均值和方差，推理时使用 running mean 和 running variance。

**数学公式**: $y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$

**输入规格**:
- x: [batch, seq_len, channels, H, W], float32/float16/bfloat16, 动态轴: batch, seq_len
- gamma (weight): [channels], float32/float16/bfloat16
- beta (bias): [channels], float32/float16/bfloat16
- eps: scalar, 默认 1e-5

**输出规格**:
- y: [batch, seq_len, channels, H, W], float32/float16/bfloat16, 动态轴: batch, seq_len

**动态轴**: batch, seq_len

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 算子仅包含逐元素运算（sub, mul, div, add）和归约运算（sum/var），无矩阵乘法操作，属于纯 Vector 类型算子。

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | reduction | mean = reduce_mean(x, axis=[0,1,*spatial], keepdims=True) | 在非通道维度计算均值 |
| 2 | elementwise | centered = x - mean | 中心化 |
| 3 | elementwise | squared = centered * centered | 计算平方 |
| 4 | reduction | var = reduce_mean(squared, axis=[0,1,*spatial], keepdims=True) | 计算方差 |
| 5 | elementwise | std = sqrt(var + eps) | 计算标准差 |
| 6 | elementwise | normalized = centered / std | 归一化 |
| 7 | elementwise | scaled = normalized * gamma | 缩放 |
| 8 | elementwise | output = scaled + beta | 偏移 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1a | reduce_sum(x, axis=[0,1,*spatial]) | `pypto.sum(x, dim=...)` | substitute | ✓ 需要多次调用 |
| 1b | mean / count | `pypto.mul(sum_result, 1.0/count)` | substitute | ✓ |
| 2 | x - mean | `pypto.sub(x, mean)` | direct | ✓ |
| 3 | centered * centered | `pypto.mul(centered, centered)` | direct | ✓ |
| 4a | reduce_sum(squared, axis=[0,1,*spatial]) | `pypto.sum(squared, dim=...)` | substitute | ✓ 需要多次调用 |
| 4b | var / count | `pypto.mul(sum_result, 1.0/count)` | substitute | ✓ |
| 4c | var (单轴) | `pypto.var(x, dim=..., correction=0, keepdim=True)` | direct | ✓ 有限制 |
| 5 | sqrt(var + eps) | `pypto.sqrt(pypto.add(var, eps))` | direct | ✓ |
| 6 | centered / std | `pypto.div(centered, std)` (或用 mul + rsqrt) | direct | ✓ |
| 7 | normalized * gamma | `pypto.mul(normalized, gamma)` | direct | ✓ |
| 8 | scaled + beta | `pypto.add(scaled, beta)` | direct | ✓ |

### 3.2 Substitute 配方

```
多轴 reduce_mean: 需要 reshape 或多次 sum 调用
# PyPTO sum/var 仅支持单轴归约，需要先 reshape 到 2D 再归约

方案 A: 先 reshape 到 2D
  x_2d = reshape(x, [-1, channels])  # [batch*seq_len*H*W, channels]
  mean = pypto.sum(x_2d, dim=0, keepdim=True) / count
  var = pypto.var(x_2d, dim=0, correction=0, keepdim=True)

方案 B: 使用 pypto.var 多轴支持
  # pypto.var 支持 dim 为 list/tuple，但约束：dim 轴不可切
  var = pypto.var(x, dim=[0,1,3,4], correction=0, keepdim=True)
```

### 3.3 关键约束

**pypto.var 约束**:
- input.shape 的 dim 轴不可切
- viewshape 的维度与 input 维度相同
- viewshape[dim] == input.shape[dim]

**这意味着**: 使用 pypto.var 时，归约轴必须是完整的（不可被 tile 切分）

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32 | FP32/FP16/BF16 | ✓ |
| contiguous | 必须 | 需确保 | 需确保 |
| shape | 非空 Tensor | [batch, seq_len, channels, H, W] | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.sum | dtype | DT_FP32, DT_INT32, DT_INT16 | ✓ (FP32 支持) |
| pypto.sum | shape | 2-4 维, Shape Size <= INT32_MAX | ✓ |
| pypto.sum | 尾轴 | 32 bytes 对齐 | ⚠ 需检查 |
| pypto.var | dtype | DT_FP32, DT_FP16, DT_BF16 | ✓ |
| pypto.var | shape | 2-4 维 | ✓ |
| pypto.var | dim 轴 | 不可切 | ⚠ 需要特殊处理 |
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

对于 BatchNorm，有两种实现策略：

### 策略 A: 2D Reshape 方案
1. 将 5D 输入 reshape 为 2D: `[batch*seq_len*H*W, channels]`
2. 在 channels 维度进行归一化（LayerNorm 风格）
3. reshape 回 5D

**Tiling**: `pypto.set_vec_tile_shapes(tile_rows, channels)`
- tile_rows: 每次处理的行数
- channels: 必须完整（用于 sum/var 归约）

### 策略 B: 直接多轴归约方案
1. 使用 `pypto.var(x, dim=[0,1,3,4], ...)` 直接计算
2. 约束: dim 轴不可切，需要 viewshape[dim] == input.shape[dim]

**Tiling**: 需要确保归约轴不被切分

**推荐**: 策略 A 更稳定，与 layer_norm 参考实现一致

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` | examples | 高 | 高 | LayerNorm 实现、Tiling 配置、精度验证 |
| `models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py` | models | 中 | 高 | RMSNorm 实现、FP32 中间精度、动态 shape 处理 |
| `operators/layer_norm/api_report.md` | 已有算子 | 高 | 高 | API 映射模式、约束分析 |

### 6.2 可复用模式

**从 `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` 提取**:

- **API 调用模式** (layernorm_core):
  ```python
  def layernorm_core(x, gamma, beta, eps, hidden_size):
      # Compute mean
      mean = pypto.sum(x, dim=-1, keepdim=True)
      mean = mean / hidden_size

      centered = x - mean
      squared = centered * centered
      var = pypto.sum(squared, dim=-1, keepdim=True)
      var = var / hidden_size

      var_eps = var + eps
      std = pypto.sqrt(var_eps)
      normalized = centered / std

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

- **FP32 中间精度**: 在归一化计算中使用 FP32 保证数值稳定性
  ```python
  add_out = pypto.cast(residual_tile, pypto.DT_FP32)
  # ... 归一化计算 ...
  in_tensor_norm = pypto.cast(res_add, in_tensor.dtype)  # 转回原类型
  ```

- **动态 shape 处理**:
  ```python
  in_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
  ```

- **RMSNorm 核心计算**:
  ```python
  square = pypto.mul(add_out, add_out)
  mean_res = pypto.mul(square, in_tensor_mean_coff)
  reduce_asum = pypto.sum(mean_res, -1, True)
  reduce_sum = pypto.add(reduce_asum, eps)
  reduce_sqrt = pypto.sqrt(reduce_sum)
  res_div = pypto.div(add_out, reduce_sqrt)
  ```

### 6.3 差异分析

| 差异点 | LayerNorm/RMSNorm | BatchNorm | 调整建议 |
|--------|----------|------------|----------|
| 归一化轴 | 最后一个维度 | 通道维度 (channels) | 需要调整 reshape 或 dim 参数 |
| 输入维度 | 2D/3D | 5D | 需要先 reshape 或支持多轴归约 |
| 动态轴 | batch | batch, seq_len | 使用 `dynamic_axis=[0, 1]` 标记 |
| 归约方式 | 单轴 sum | 多轴 mean | 使用 var API 或多次 sum |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| pypto.var 多轴归约约束 | dim 轴不可切 | 使用 reshape 到 2D 方案规避 |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 精度稳定性 | 建议中间计算使用 FP32，最后转回原 dtype，避免数值溢出 |
| 尾轴对齐 | sum 操作要求尾轴 32 bytes 对齐，需检查 channels 是否满足 |
| 动态 shape | 需要使用 `pypto.from_torch(x, dynamic_axis=[0, 1])` 标记动态轴 |
| Tiling 设置 | channels 维度需要完整处理以保证归约精度 |
| 广播处理 | gamma/beta 为 1D [channels]，需要广播到 5D，PyPTO 支持自动广播 |
| 多轴归约 | pypto.var 支持 list/tuple dim，但有 dim 轴不可切的约束 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 列表 | `docs/api/operation/index.md` |
| pypto.sum 文档 | `docs/api/operation/pypto-sum.md` |
| pypto.var 文档 | `docs/api/operation/pypto-var.md` |
| pypto.mul 文档 | `docs/api/operation/pypto-mul.md` |
| pypto.sub 文档 | `docs/api/operation/pypto-sub.md` |
| pypto.add 文档 | `docs/api/operation/pypto-add.md` |
| pypto.sqrt 文档 | `docs/api/operation/pypto-sqrt.md` |
| pypto.div 文档 | `docs/api/operation/pypto-div.md` |
| pypto.rsqrt 文档 | `docs/api/operation/pypto-rsqrt.md` |
| pypto.cast 文档 | `docs/api/operation/pypto-cast.md` |
| pypto.assemble 文档 | `docs/api/operation/pypto-assemble.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| LayerNorm 参考实现 | `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` |
| RMSNorm 参考实现 | `models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py` |
| LayerNorm API 报告 | `operators/layer_norm/api_report.md` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无阻断问题，所有必需 API 均已支持
- **实现建议**:
  1. 采用 2D Reshape 方案: 将 5D 输入 reshape 为 2D `[N, channels]`，其中 N = batch*seq_len*H*W
  2. 参考 `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` 的 `layernorm_core` 函数结构
  3. 使用 `pypto.sum` + `mul` 组合实现 mean 计算，或使用 `pypto.var(dim=0)` 计算方差
  4. 中间计算使用 FP32 保证数值稳定性
  5. 支持动态 batch 和 seq_len 维度
  6. 合理设置 Tiling 策略: `pypto.set_vec_tile_shapes(tile_size, channels)`

- **精度策略**: 内部使用 FP32 计算，输入输出支持 FP32/FP16/BF16

- **性能优化方向**:
  1. 考虑使用 `pypto.rsqrt` 替代 `sqrt + div` 组合，减少一次运算
  2. 优化 Tiling 配置以提高向量化利用率
  3. 对于推理模式，可预先 reshape running_mean/running_var

---

## 10. 实现方案

### 10.1 推荐实现架构

```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def batchnorm_kernel(x, gamma, beta, output, config):
    """
    BatchNorm kernel 实现

    输入:
      x: [batch, seq_len, channels, H, W]
      gamma: [channels]
      beta: [channels]
    输出:
      output: [batch, seq_len, channels, H, W]
    """
    batch, seq_len, channels, H, W = x.shape
    N = batch * seq_len * H * W

    # Reshape 到 2D: [N, channels]
    x_2d = pypto.reshape(x, [N, channels])

    # 设置 Tiling
    pypto.set_vec_tile_shapes(64, channels)

    # 计算均值 (沿 axis=0)
    mean = pypto.sum(x_2d, dim=0, keepdim=True)
    mean = pypto.mul(mean, 1.0 / N)

    # 中心化
    centered = pypto.sub(x_2d, mean)

    # 计算方差
    squared = pypto.mul(centered, centered)
    var = pypto.sum(squared, dim=0, keepdim=True)
    var = pypto.mul(var, 1.0 / N)

    # 归一化
    var_eps = pypto.add(var, config.eps)
    std = pypto.sqrt(var_eps)
    normalized = pypto.div(centered, std)

    # 仿射变换
    scaled = pypto.mul(normalized, gamma)
    result = pypto.add(scaled, beta)

    # Reshape 回 5D
    result_5d = pypto.reshape(result, [batch, seq_len, channels, H, W])

    # 输出
    pypto.assemble(result_5d, [0, 0, 0, 0, 0], output)
```

### 10.2 关键约束规避

| 约束 | 规避方法 |
|------|----------|
| pypto.sum 尾轴 32B 对齐 | 确保 channels * dtype_size 是 32 的倍数，或 padding |
| pypto.var dim 轴不可切 | 使用 reshape + pypto.sum 方案，不直接用 pypto.var |
| Shape 仅支持 2-4 维 | reshape 5D 到 2D 处理 |

---

*报告生成完成*
