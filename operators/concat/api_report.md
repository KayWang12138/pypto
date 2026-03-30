# API 探索报告

> **生成时间**: 2026-03-28

---

<!-- REQUIRED -->
## 1. 概述

### 1.1 输入摘要

- **算子名称**: concat
- **功能描述**: 将多个张量沿指定维度拼接成一个张量
- **输入**: tensors (List[Tensor]), dim (int)
- **输出**: output (Tensor)
- **关键约束**: 非拼接维度必须相同，dtype 必须一致

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: concat 是形状操作算子，不涉及矩阵乘法，仅涉及内存复制和拼接，属于 Vector 类型算子

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | shape | output[i] = tensors[i] | 沿 dim 维度拼接多个张量 |

---

<!-- REQUIRED -->
## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | concat(tensors, dim) | `pypto.concat(tensors, dim)` | direct | ✓ |

### 3.2 Substitute 配方

无需 substitute，PyPTO 直接支持 concat 操作。

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8/INT16/INT32 | float32/float16/bfloat16 | ✓ |
| contiguous | 必须 | — | 需确保 |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.concat | tensors 数量 | 2 <= len(tensors) <= 128 | ✓ |
| pypto.concat | Shape 维度 | 仅支持 2-4 维 | ✓ |
| pypto.concat | Shape Size | <= INT32_MAX (2147483647) | ✓ |
| pypto.concat | dtype 一致性 | 所有 tensor dtype 相同 | ✓ |
| pypto.concat | 非拼接维度 | 必须相同 | ✓ |
| pypto.concat | dim 范围 | -input.dim <= dim < input.dim | ✓ |
| pypto.concat | viewshape | dim 对应维度不切块 | 需注意 |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**Tiling 配置说明**:
- TileShape 维度应和输出一致
- 如输入 tensors 维度为 [m, c1, p], [m, c2, p]，输出为 [m, c1+c2, p]
- TileShape 设置为 [m1, n1, p1]，则 m1, p1 分别用于切分 m, p 轴，n1 用于切分 c1 和 c2 轴

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/01_beginner/transform/transform_ops.py` | examples | 高 | 高 | 基础 concat 用法、多张量拼接、不同维度拼接 |
| `models/glm_v4_5/glm_attention_fusion.py` | models | 中 | 高 | 复杂场景下的 concat 使用（RoPE 后拼接） |

### 6.2 可复用模式

- **API 调用模式**:
  ```python
  pypto.set_vec_tile_shapes(*tile_shapes)
  out[:] = pypto.concat([a, b], dim=dim)
  ```

- **Tiling 策略**:
  ```python
  tile_shapes = [8 for _ in range(len(a.shape))]
  pypto.set_vec_tile_shapes(*tile_shapes)
  ```

- **Loop 结构**: 基础用法无需 loop，直接调用即可

- **边界处理**: 支持不同 shape 的张量拼接（非拼接维度相同前提下）

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 动态轴 | 示例未显式处理动态轴 | 需支持 batch 和 seq_len 动态 | 使用 `pypto.DYNAMIC` 标记动态轴 |
| 张量数量 | 示例支持 2-3 个张量 | 需支持可变数量（2-128） | 按需传入 tensor 列表 |

---

<!-- REQUIRED -->
## 7. 风险评估

### 7.1 阻断问题

无阻断问题。PyPTO 直接支持 concat 操作，API 完整。

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| viewshape 约束 | 设置 viewshape 时，dim 对应维度不切块（即 viewshape 对应值 >= tensors 任一 tensor 的对应值） |
| 单张量输入 | 支持输入一个 tensor 情况，但精度暂时不保证（建议至少 2 个） |
| 空张量 | 不支持空 Tensor |
| Shape 维度限制 | 仅支持 2-4 维，需注意 spec 中 3D 场景符合要求 |

---

<!-- REQUIRED -->
## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| concat 文档 | `docs/api/operation/pypto-concat.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 1 | `examples/01_beginner/transform/transform_ops.py` |
| 参考实现 2 | `models/glm_v4_5/glm_attention_fusion.py` |

---

<!-- REQUIRED -->
## 9. 结论

- **可行性**: 可行
- **主要问题**: 无
- **API 映射**: PyPTO 原生支持 `pypto.concat`，直接映射
- **实现难度**: 低
- **建议**: 直接使用 `pypto.concat` API，注意 Tiling 配置和动态轴处理
