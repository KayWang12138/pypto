# API 探索报告

> **生成时间**: 2026-03-28

---

<!-- REQUIRED -->
## 1. 概述

### 1.1 输入摘要

- **算子名称**: linear
- **数学公式**: output = input @ weight^T + bias
- **功能描述**: 线性层（全连接层）算子，对输入张量进行线性变换。支持任意维度的输入张量，对最后一维进行线性变换，保持其他维度不变。
- **输入规格**:
  - input: [..., in_features], float32, 动态轴 batch, seq_len
  - weight: [out_features, in_features], float32
  - bias: [out_features], float32 (可选)
- **输出规格**: output: [..., out_features], float32

### 1.2 算子分类

- **类型**: Cube + Vector 混合类型
- **判断依据**: 公式包含 matmul 操作（Cube 类型）和 bias add 操作（Vector 类型）。matmul 是核心计算，bias_add 是可选的逐元素加法。

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | matmul | input @ weight^T | 矩阵乘法，weight 需要转置 |
| 2 | add | result + bias | 逐元素加法，bias 广播到 result shape |

**关键点**: PyPTO matmul API 支持 `b_trans=True` 参数，可以直接对右矩阵进行转置，无需单独调用 transpose。

---

<!-- REQUIRED -->
## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1a | input @ weight^T | `pypto.matmul(input, weight, dtype, b_trans=True)` | direct | ✓ |
| 1b | input @ weight^T + bias | `pypto.matmul(input, weight, dtype, b_trans=True, extend_params={'bias_tensor': bias})` | direct | ✓ |
| 2 | result + bias | `pypto.add(matmul_result, bias)` | direct | ✓ |

### 3.2 实现方案

**方案一：bias 融合（推荐，仅限 2D 场景）**
```python
# bias_tensor 融合到 matmul 中，仅支持 2D 输入
extend_params = {'bias_tensor': bias_reshaped}  # bias shape: [1, out_features]
out = pypto.matmul(input_2d, weight, dtype, b_trans=True, extend_params=extend_params)
```

**方案二：分开实现（通用方案）**
```python
# 先做 matmul，再做 add
matmul_result = pypto.matmul(input, weight, dtype, b_trans=True)
out = pypto.add(matmul_result, bias)
```

**选择建议**:
- 2D 输入场景：优先使用方案一（bias 融合），性能更优
- 3D/4D 输入场景：使用方案二（分开实现），bias_tensor 融合不支持 3D+

---

## 4. 约束检查

### 4.1 入口约束（from_torch）

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | float32 | ✓ |
| contiguous | 必须 | — | 需确保 |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| matmul | input dtype | DT_INT8, DT_FP16, DT_BF16, DT_FP32 | ✓ (FP32) |
| matmul | input shape | 2-4 维 | ✓ |
| matmul | weight shape | 与 input 维度一致 | ✓ |
| matmul | 内轴范围 | [1, 65535] (ND格式) | ✓ |
| matmul | 外轴范围 | [1, 2^31-1] (ND格式) | ✓ |
| matmul (bias_tensor) | 维度限制 | 仅支持 2D | ⚠ 3D+需分开实现 |
| matmul (bias_tensor) | bias shape | [1, N] N=mat2的N维度 | ✓ |
| add | input dtype | DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32 | ✓ |
| add | shape | 2-4 维，支持单维度广播 | ✓ |

### 4.3 Tiling 约束

| API | 约束项 | 要求 |
|-----|--------|------|
| set_cube_tile_shapes | kL0, kL1, nL0, nL1 | 32 字节对齐（FP32 为 16 元素对齐） |
| set_cube_tile_shapes | mL0, mL1 | 0 < mL0 <= mL1, mL1 % mL0 == 0 |
| set_cube_tile_shapes | bias_tensor | nL0 * 4 <= BTBuffer_size (1KB) |
| set_vec_tile_shapes | 每维 | > 0 |

---

## 5. Tiling 需求

### 5.1 Cube Tiling（matmul 必需）

```python
pypto.set_cube_tile_shapes(
    m=[mL0, mL1],   # M 轴切分
    k=[kL0, kL1],   # K 轴切分
    n=[nL0, nL1],   # N 轴切分
    enable_split_k=False  # 3D/4D 不支持 split_k
)
```

### 5.2 Vector Tiling（3D/4D matmul 和 add 必需）

```python
# 3D 输入
pypto.set_vec_tile_shapes(batch_tile, seq_tile, feature_tile)

# 4D 输入
pypto.set_vec_tile_shapes(b1_tile, b2_tile, seq_tile, feature_tile)
```

### 5.3 推荐配置

| 场景 | cube_tile_shapes | vec_tile_shapes |
|------|------------------|-----------------|
| 小矩阵 (M,N,K <= 64) | [32,32], [64,64], [64,64] | [32, 64] |
| 中等矩阵 (<= 2048) | [128,128], [128,128], [128,128] | [128, 128] |
| 大矩阵 (> 2048) | [256,256], [256,256], [256,256] | [128, 128] |

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `operators/matmul/matmul_impl.py` | operators | 高 | 高 | matmul core 实现模式、tiling 选择策略、多维度 kernel 分发 |
| `examples/01_beginner/compute/matmul_ops.py` | examples | 高 | 高 | b_trans=True 使用方式、cube tiling 配置 |
| `examples/02_intermediate/basic_nn/ffn/ffn_module.py` | examples | 高 | 高 | FFN 中 matmul 调用、dynamic shape 处理、activation 融合 |

### 6.2 可复用模式

- **API 调用模式**:
  ```python
  # 使用 b_trans=True 实现 weight 转置
  out = pypto.matmul(input, weight, dtype, b_trans=True)
  ```
- **Tiling 策略**:
  ```python
  # 根据矩阵大小动态选择 tiling
  if m <= 64 and k <= 64 and n <= 64:
      pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
  elif m <= 2048:
      pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
  else:
      pypto.set_cube_tile_shapes([256, 256], [256, 256], [256, 256])
  ```
- **Loop 结构**: 无显式 loop，matmul API 内部处理分块计算
- **边界处理**: 3D/4D 需要额外调用 set_vec_tile_shapes

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| bias 处理 | 示例多为纯 matmul | 需支持可选 bias | 有 bias 时调用 add 或使用 extend_params |
| 转置方式 | 少数使用 b_trans | weight 必须转置 | 使用 b_trans=True 参数 |
| 动态 shape | 部分示例支持 | 必须支持 batch, seq_len 动态 | 使用 from_torch 的 dynamic_axis 参数 |

---

<!-- REQUIRED -->
## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | 所有核心 API 均可用 | — |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| bias_tensor 融合限制 | extend_params 的 bias_tensor 仅支持 2D matmul，3D+ 需分开用 add |
| 动态 shape 处理 | 需在 from_torch 时指定 dynamic_axis=[0, 1]（batch, seq_len） |
| FP32 对齐要求 | set_cube_tile_shapes 的 kL0, kL1, nL0, nL1 需 16 元素对齐（非 32 字节） |
| contiguous 输入 | 必须确保 torch.Tensor.is_contiguous() == True |

---

<!-- REQUIRED -->
## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| matmul API 文档 | `docs/api/operation/pypto-matmul.md` |
| add API 文档 | `docs/api/operation/pypto-add.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Cube Tiling | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| Vector Tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 matmul | `operators/matmul/matmul_impl.py` |
| 参考实现 FFN | `examples/02_intermediate/basic_nn/ffn/ffn_module.py` |

---

<!-- REQUIRED -->
## 9. 结论

- **可行性**: 可行
- **主要问题**: 无阻断问题
- **实现建议**:
  1. 2D 输入场景优先使用 bias_tensor 融合方案
  2. 3D/4D 输入使用 matmul + add 分开方案
  3. 使用 b_trans=True 实现 weight 转置，避免额外 transpose 操作
  4. 根据输入 shape 动态选择 tiling 配置
  5. 使用 from_torch 的 dynamic_axis 支持动态 shape

---

## 10. 典型配置映射

| spec 配置 | 实现方案 |
|-----------|----------|
| 性能_P0_2D (batch, 4096) @ (4096, 4096).T + bias | matmul + extend_params={'bias_tensor': bias} |
| 性能_P0_3D (batch, seq, 1024) @ (4096, 1024).T + bias | matmul(b_trans=True) + add (分开实现) |
| 功能_P0_no_bias (batch, 512) @ (512, 512).T | matmul(b_trans=True) |
| 功能_P1_4D (batch, heads, seq, 256) @ (512, 256).T + bias | matmul(b_trans=True) + add (分开实现) |
