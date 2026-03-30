# API 探索报告

> **生成时间**: 2026-03-29

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: batch_matmul
- **数学公式**: C[b,m,n] = sum_k(A[b,m,k] * B[b,k,n])
- **输入规格**:
  - x1: [batch, m, k], float32
  - x2: [batch, k, n], float32
- **输出规格**: y: [batch, m, n], float32
- **可选参数**: transpose_x1 (P2), transpose_x2 (P2)
- **精度要求**: rtol < 1e-5, atol < 1e-5

### 1.2 算子分类

- **类型**: Cube
- **判断依据**: 核心操作是批量矩阵乘法 (matmul)，属于 Cube 类型运算，需要使用 `pypto.set_cube_tile_shapes()` 配置

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | matmul | C[b,m,n] = A[b,m,k] @ B[b,k,n] | 批量矩阵乘法，对每个 batch 执行矩阵乘法 |
| 2 | transpose (可选) | A' = A^T 或 B' = B^T | 根据参数对输入矩阵转置 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | C = A @ B | `pypto.matmul()` | direct | ✓ |
| 2 | A' = A^T | `pypto.matmul(a_trans=True)` | direct | ✓ |
| 2 | B' = B^T | `pypto.matmul(b_trans=True)` | direct | ✓ |

### 3.2 API 详细映射

**核心 API**: `pypto.matmul(input, mat2, out_dtype, *, a_trans=False, b_trans=False, ...)`

| 参数 | 对应算子输入 | 说明 |
|------|-------------|------|
| `input` | x1 | 左矩阵 [batch, m, k] |
| `mat2` | x2 | 右矩阵 [batch, k, n] |
| `out_dtype` | DT_FP32 | 输出数据类型 |
| `a_trans` | transpose_x1 | 左矩阵转置标志 |
| `b_trans` | transpose_x2 | 右矩阵转置标志 |

**支持的数据类型**:
- 输入: DT_FP16, DT_BF16, DT_FP32, DT_INT8
- 输出: DT_FP16, DT_BF16, DT_FP32, DT_INT32

**支持的矩阵维度**: 2维、3维、4维

---

## 4. 约束检查

### 4.1 入口约束 (from_torch)

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | FP32 | ✓ |
| contiguous | 必须 | — | 需确保 |
| shape | 非空 Tensor | [b,m,k], [b,k,n] | ✓ |

### 4.2 API 约束 (pypto.matmul)

| 约束项 | 要求 | 结果 |
|--------|------|------|
| 矩阵维度 | 2维/3维/4维，左右维度一致 | ✓ (3维) |
| Format | TILEOP_ND, TILEOP_NZ | ✓ |
| 外轴范围 (ND) | [1, 2^31 - 1] | ✓ |
| 内轴范围 (ND) | [1, 65535] | ✓ |
| dtype 一致性 | 左右矩阵 dtype 一致 (除 FP8) | ✓ |

### 4.3 3维/4维矩阵额外约束

| 约束项 | 要求 |
|--------|------|
| vec_tile_shapes | 需调用 `pypto.set_vec_tile_shapes()` 设置 Vector TileShape |
| 默认值 | 若未设置，接口内部自动设置为 [128, 128] |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API | 说明 |
|----------|-----------|------|
| Cube | `pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1])` | 设置 M、N、K 轴切分大小 |
| Vector (3D/4D) | `pypto.set_vec_tile_shapes(b, m, n)` | 3维矩阵需额外设置 |

### 5.1 Cube Tiling 约束

```
buffer 空间约束 (FP16/BF16/FP32):
- L0A: CeilAlign(mL0,16) * CeilAlign(kL0,16) * sizeof(dtype) <= L0A_size
- L0B: CeilAlign(nL0,16) * CeilAlign(kL0,16) * sizeof(dtype) <= L0B_size
- L0C: CeilAlign(mL0,16) * CeilAlign(nL0,16) * sizeof(dtype) <= L0C_size
- L1: CeilAlign(mL1,16) * CeilAlign(kL1,16) * sizeof(dtype) + CeilAlign(nL1,16) * CeilAlign(kL1,16) * sizeof(dtype) <= L1_size

切分约束:
- mL0 <= mL1, kL0 <= kL1, nL0 <= nL1
- mL1 % mL0 == 0, kL1 % kL0 == 0, nL1 % nL0 == 0
```

### 5.2 推荐 Tiling 配置

| 场景 | cube_tile_shapes | vec_tile_shapes |
|------|------------------|-----------------|
| 小规模 (m,n,k <= 128) | [[32,32], [32,32], [32,32]] | (batch, m, n) |
| 中规模 (m,n,k <= 1024) | [[64,128], [128,128], [128,128]] | (batch, m, n) |
| 大规模 (m,n,k > 1024) | [[128,256], [256,256], [256,256]] | (batch, m, n) |

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/01_beginner/compute/matmul_ops.py` | examples | 高 | 高 | 批量 matmul、转置参数使用、tiling 配置 |
| `examples/01_beginner/tiling/tiling_config.py` | examples | 高 | 高 | cube/vec tile shapes 配置、性能对比 |
| `models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py` | models | 中 | 高 | 3D matmul 实际应用、动态 batch 处理 |

### 6.2 可复用模式

**API 调用模式**:
```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def batch_matmul_kernel(x1, x2, out, transpose_x1=False, transpose_x2=False):
    pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1])
    out[:] = pypto.matmul(x1, x2, pypto.DT_FP32, a_trans=transpose_x1, b_trans=transpose_x2)
```

**Tiling 策略**:
- 3D matmul 需同时设置 `set_cube_tile_shapes` 和 `set_vec_tile_shapes`
- cube_tile_shapes 控制 M/N/K 轴的 Cube 切分
- vec_tile_shapes 控制 batch 维度的 Vector 切分

**Loop 结构**:
- 对于动态 batch 场景，可使用 `pypto.loop()` 进行批处理
- 参考: `models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py` 中的 batch loop 模式

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| transpose 参数 | 通过 a_trans/b_trans 传入 | 支持 transpose_x1/x2 | 直接映射，无需调整 |
| 动态 batch | 部分示例使用静态 shape | 支持 batch 动态 | 使用 dynamic_axis 标记 |
| 输出 dtype | 可配置 out_dtype | 固定 FP32 | 显式指定 DT_FP32 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | API 直接支持所需功能 | - |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| contiguous 要求 | 输入 tensor 必须是连续的，调用前需确保 `tensor.is_contiguous() == True` |
| 3D matmul 需设置 vec_tile_shapes | 3维矩阵乘法除 cube tiling 外，还需设置 vector tiling |
| 转置参数语义 | `a_trans=True` 表示 x1 转置，`b_trans=True` 表示 x2 转置 |
| 性能调优 | tiling 配置对性能影响大，需根据实际 shape 调优 |
| 动态轴 | batch 维度支持动态，需通过 `from_torch(..., dynamic_axis=[0])` 标记 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| matmul 文档 | `docs/api/operation/pypto-matmul.md` |
| transpose 文档 | `docs/api/operation/pypto-transpose.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| cube tiling | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| vec tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 (matmul) | `examples/01_beginner/compute/matmul_ops.py` |
| 参考实现 (tiling) | `examples/01_beginner/tiling/tiling_config.py` |
| 参考实现 (model) | `models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无阻断问题
- **API 选型**: `pypto.matmul()` 直接支持批量矩阵乘法，包括转置参数
- **实现建议**:
  1. 使用 `pypto.matmul()` 作为核心 API
  2. 对于 3D 输入，需同时配置 cube_tile_shapes 和 vec_tile_shapes
  3. 通过 `a_trans` 和 `b_trans` 参数支持转置功能
  4. 动态 batch 维度通过 `from_torch(..., dynamic_axis=[0])` 实现
