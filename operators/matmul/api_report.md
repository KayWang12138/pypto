# API 探索报告

> **生成时间**: 2026-03-28T23:20:00Z

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: matmul
- **数学公式**: C = A @ B, C_ij = sum_k(A_ik * B_kj)
- **功能描述**: 矩阵乘法算子，支持 batch 维度广播的批量矩阵乘法
- **输入规格**:
  - A: [..., M, K], dtype: float32/float16
  - B: [..., K, N], dtype: float32/float16
- **输出规格**:
  - C: [..., M, N], dtype: float32/float16
- **关键特性**: batch_matmul, batch_broadcast, 2D_matmul, dynamic_axis

### 1.2 算子分类

- **类型**: Cube
- **判断依据**: 算子核心操作为矩阵乘法（matmul），属于 Cube 类型算子，需要使用 `pypto.set_cube_tile_shapes()` 配置 Tiling

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | matmul | C = A @ B | 批量矩阵乘法，支持 batch 维度广播 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | C = A @ B | `pypto.matmul(input, mat2, out_dtype, *, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=None)` | direct | ✓ |

### 3.2 Substitute 配方

N/A - 直接映射

---

## 4. 约束检查

### 4.1 入口约束 (from_torch)

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int8, torch.uint8, torch.int16, torch.uint16, torch.int32, torch.uint32, torch.int64, torch.uint64, torch.bool | float32, float16 | ✓ |
| contiguous | 必须 (tensor.is_contiguous() == True) | — | 需确保 |

### 4.2 API 约束 (pypto.matmul)

| 约束项 | 要求 | 结果 |
|--------|------|------|
| **输入矩阵维度** | 2维、3维、4维，且左右矩阵维度需保持一致 | ✓ (spec 支持 2D, 3D, 4D) |
| **输入 dtype** | DT_INT8, DT_FP16, DT_BF16, DT_FP32, DT_HF8, DT_FP8E5M2, DT_FP8E4M3 | ✓ (float32, float16) |
| **输出 dtype** | DT_FP32, DT_FP16, DT_BF16, DT_INT32 | ✓ |
| **输入格式** | TILEOP_ND, TILEOP_NZ | ✓ (默认 ND) |
| **ND 格式外轴范围** | [1, 2^31 - 1] | ✓ |
| **ND 格式内轴范围** | [1, 65535] | ✓ |
| **batch 广播** | 支持维度为 1 的轴扩展 | ✓ |
| **动态轴** | 支持 batch, M, N, K 动态变化 | ✓ |

### 4.3 Tiling 约束

#### Cube Tiling (set_cube_tile_shapes)

| 约束项 | 要求 | 说明 |
|--------|------|------|
| **对齐约束** | kL0, kL1, nL0, nL1 均需 32 字节对齐（FP32 场景 16 元素对齐） | 必须满足 |
| **mL0 范围** | 0 < mL0 <= mL1 且 mL1 % mL0 == 0 | 必须满足 |
| **kL0 范围** | 0 < kL0 <= kL1 且 kL1 % kL0 == 0 | 必须满足 |
| **nL0 范围** | 0 < nL0 <= nL1 且 nL1 % nL0 == 0 | 必须满足 |
| **L0 空间** | CeilAlign(mL0,16) * CeilAlign(kL0,16) * sizeof(aDtype) <= L0A_size | FP16/BF16/FP32 |
| **L1 空间** | CeilAlign(mL1,16) * CeilAlign(kL1,16) * sizeof(aDtype) + CeilAlign(nL1,16) * CeilAlign(kL1,16) * sizeof(bDtype) <= L1_size | FP16/BF16/FP32 |
| **3D/4D 场景** | 需额外调用 `pypto.set_vec_tile_shapes()` | 必须 |

#### Vector Tiling (set_vec_tile_shapes) - 3D/4D 场景

| 约束项 | 要求 |
|--------|------|
| 维度数量 | 每个维度必须 > 0，最多 4 个维度 |
| 默认值 | 如未设置，内部默认为 [128, 128] |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API | 备注 |
|----------|-----------|------|
| Cube | `pypto.set_cube_tile_shapes(m, k, n, enable_split_k)` | 调用 matmul 前必须设置 |
| Vector (3D/4D) | `pypto.set_vec_tile_shapes(*args)` | 3D/4D 场景必须额外设置 |

### 推荐配置

```python
# 2D 场景
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# 3D/4D 场景
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
pypto.set_vec_tile_shapes(128, 128)  # 处理 batch 维度
```

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/01_beginner/compute/matmul_ops.py` | examples | 高 | 高 | 基础 matmul、batch matmul、broadcast、transpose、bias 场景 |
| `examples/01_beginner/tiling/tiling_config.py` | examples | 高 | 高 | Tiling 配置最佳实践、性能调优示例 |
| `models/glm_v4_5/glm_attention.py` | models | 中 | 高 | 复杂场景中的 matmul 使用、动态 shape、循环结构 |

### 6.2 可复用模式

#### API 调用模式
```python
# 基础 2D matmul
pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
out[:] = pypto.matmul(a, b, pypto.DT_FP32)

# Batch matmul with broadcast
pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
out[:] = pypto.matmul(a, b, pypto.DT_FP32)

# Transpose 支持
out[:] = pypto.matmul(a, b, pypto.DT_FP32, a_trans=True, b_trans=True)

# Bias 支持
extend_params = {"bias_tensor": bias}
out[:] = pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)
```

#### Tiling 策略
- **小规模**: `[32, 32], [64, 64], [64, 64]` - 适合小矩阵
- **中规模**: `[128, 128], [128, 128], [128, 128]` - 平衡性能
- **大规模**: `[64, 64], [128, 128], [128, 128]` - 大矩阵优化

#### Loop 结构
- 3D/4D 场景需要使用 `pypto.loop()` 遍历 batch 维度
- 动态 shape 使用 `pypto.DYNAMIC` 标记

#### 边界处理
- 使用 `pypto.view()` 处理动态 shape 的有效区域
- 使用 `valid_shape` 参数指定实际计算范围

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| Batch 广播 | 示例展示了基础广播（如 [1,B,M,K] @ [B,1,K,N]） | 需支持更复杂的多维广播 | 参考 GLM attention 中的复杂广播处理 |
| 动态轴 | 示例使用固定 shape | 需支持 batch, M, N, K 全动态 | 使用 `dynamic_axis` 参数 + `pypto.DYNAMIC` 标记 |
| 性能要求 | 示例未强调性能目标 | 需达到首跑性能的 2 倍 | 参考 tiling_config.py 中的性能调优实践 |

---

## 7. 风险评估

### 7.1 阻断问题

无阻断问题。PyPTO `pypto.matmul` API 完全满足算子需求。

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| **Tiling 配置必须** | 调用 `pypto.matmul` 前必须调用 `pypto.set_cube_tile_shapes()`，否则会报错 |
| **3D/4D 需 Vector Tiling** | 矩阵维度为 3D 或 4D 时，必须额外调用 `pypto.set_vec_tile_shapes()` |
| **动态 shape 标记** | 使用 `pypto.from_torch(tensor, dynamic_axis=[...])` 标记动态维度 |
| **对齐约束** | Tiling 配置需满足 32 字节对齐（FP32 场景 16 元素对齐） |
| **多核切 K 限制** | 3D/4D 场景不支持 `enable_split_k=True` |
| **Contiguous 要求** | 输入 tensor 必须连续（`tensor.is_contiguous() == True`） |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` (第 65 行) |
| matmul API 文档 | `docs/api/operation/pypto-matmul.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Cube Tiling 文档 | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| Vector Tiling 文档 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 基础示例 | `examples/01_beginner/compute/matmul_ops.py` |
| Tiling 示例 | `examples/01_beginner/tiling/tiling_config.py` |
| 复杂场景示例 | `models/glm_v4_5/glm_attention.py` |

---

## 9. 结论

- **可行性**: **可行**
- **主要问题**: 无

### 实施建议

1. **直接使用 `pypto.matmul` API**，完全满足所有需求
2. **Tiling 配置**:
   - 2D 场景：仅调用 `pypto.set_cube_tile_shapes()`
   - 3D/4D 场景：同时调用 `pypto.set_cube_tile_shapes()` 和 `pypto.set_vec_tile_shapes()`
3. **动态 shape 处理**: 使用 `dynamic_axis` 参数标记动态维度
4. **性能优化**: 参考 `tiling_config.py` 中的不同 tiling 配置对性能的影响
5. **测试覆盖**: 参考 `matmul_ops.py` 中的测试用例（basic, batch, broadcast, trans, bias）

### 后续步骤

进入 Stage 3 - Golden 参考实现生成，基于本报告的 API 映射和约束信息生成 `{op}_golden.py`。
