# API 探索报告

## 1. 概述
- **算子名称**: max_reduction
- **算子分类**: reduction
- **生成时间**: 2026-03-29T15:08:00Z
- **公式**: y = max(x, dim) = max(x_i for i in dim_axis)

 y_i: input_tensor[i]

 y: torch.max(input_tensor, dim=dim, keepdim=keepdim)

 return y

## 2. 输入摘要
沿指定轴计算张量最大值的归约操作，支持动态轴（batch, seq_len 维度)。

## 3. 算子分类
- **类型**: vector
- **判断依据**: 仅涉及 reduction (max) 操作，无 matmul

- **特点**: 单轴归约，支持 keepdim 参数

- **参考**: mean_reduction (sum + div 实现 mean)

- **对比**: 与 mean 相比，max 操作不涉及除法，而是语义更直接

- **性能**: 与 mean 的 sum + div 相比，单次 max 操作通常更快

- **应用场景**: 最大池化、特征提取、注意力机制中的 max pooling

RoI Pooling 等
## 4. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | reduction | max(x, dim, keepdim) | 沿 dim 轴计算最大值 |

**无需 substitute 方案**: PyPTO 揙供直接的 `pypto.max` API,可以一步完成计算。

## 5. API 映射结果

### 5.1 映射结果
| 步骤 | 数学表达 | PyPTO API | 参数 | 映射级别 | 约束满足 | 文档路径 |
|------|----------|-----------|----------|------------|----------|
| 1 | max(x, dim, keepdim) | `pypto.max(input, dim, keepdim)` | input: Tensor, dim: int, keepdim: bool | ✓ 直接 | docs/api/operation/pypto-max.md |

### 5.2 匹配示例
参考 `operators/mean_reduction/api_report.md` 的 sum 宐 div 组合模式：
```python
# 步骤 1: 沿 dim 轴求和
sum_result = pypto.sum(input_tensor, dim=dim, keepdim=keepdim)

# 步骤 2: 除以归约轴元素数量 N 得到均值
mean_result = pypto.div(sum_result, N)

```

Pypto.max 可以一步完成计算，无需 sum + div 的组合。

## 6. 约束检查
### 6.1 输入约束
| 约束项 | 要求 | 结果 |
|--------|------|------|
| dtype | {DT_FP16, DT_BF16, DT_FP32} | ✓ 支持 |
| shape | 2-4 维 | ✓ 支持 |
| Shape Size | ≤ INT32_MAX | ✓ 支持 |
| contiguous | 必须 | -- | {✓/需确保} |
| dim 茽 | 单轴 | ✓ 支持（单轴归约) |
| keepdim | bool | -- | ✓ 支持 |

### 6.2 API 约束
| 约束项 | 要求 | 结果 |
|--------|------|------|
| dtype | {DT_FP16, DT_BF16, DT_FP32} | ✓ 支持 |
| shape | 2-4 维 | ✓ 支持 |
| Shape size | ≤ INT32_MAX | ✓ 支持 |
| TileShape | ≤ 64KB | ✓ 满足 |
| 尾轴 32 bytes 对齐 | ✓ 满足 |
| 次尾轴 ≤ 255 | ✓ 满足 |

### 6.3 约束来源
- 文档路径: docs/api/operation/pypto-max.md
- 验证时间: 2026-03-29

## 7. Tiling 需求
### 7.1 算子类型
- **类型**: vector
- **Tiling API**: `pypto.set_vec_tile_shapes()`

### 7.2 Tiling 配置
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| tile_shape | 根据输入维度设置 | 例如 3D 输入使用 [8, 8, 8] |

### 7.3 Tiling 约束
- 尾轴需 32 bytes 对齐（float32 需要 8 元素对齐)
- 次尾轴 ≤ 255
- TileShape ≤ 64KB

- 维度数与输入 tensor 一致

## 8. 参考实现
- **路径**: models/examples/02_intermediate/operators/max_example.py
- **类型**: 完整实现示例
- **置信度**: ⭐⭐⭐⭐⭐

```python
# 示例: 使用 pypto.max 讲行沿指定轴计算最大值
import pypto

@pypto.frontend.jit
def max_example(x, dim, keepdim):
    pypto.set_vec_tile_shapes(8, 8, 8)
    result = pypto.max(x, dim=dim, keepdim=keepdim)
    return result
```

### 8.1 匹配示例来源
- 文件路径: models/examples/02_intermediate/operators/max_example.py
- 置信度: ⭐⭐⭐⭐⭐
- 可复用性: 高
- 相似度: 高

## 9. 风险评估
### 9.1 可行性
- **评估**: ✓ 可行
- **原因**: PyPTO 揚供直接的 `pypto.max` API，可一步完成计算
- **风险**: 无明显风险

### 9.2 阻塞问题
| 问题 | 严重性 | 建议解决方案 |
|------|--------|----------|
| 无 | N/A | N/A |
### 9.3 需调整项
| 项目 | 当前状态 | 建议调整 |
|------|--------|----------|
| 无 | N/A | N/A |
## 10. 匹配示例

- **示例算** spec 与实现匹配度高，可复用性强
- **参考**: 可直接参考 `models/examples/02_intermediate/operators/max_example.py`
- **说明**: 该示例展示了 pypto.max 的完整用法
## 11. 证据索引
| 信息 | 文档路径 |
|------|----------|
| API 文档 | docs/api/operation/pypto-max.md |
| API 约束 | docs/api/operation/pypto-max.md § 4.1, 4.2 |
| Tiling 文档 | docs/api/config/pypto-set_vec_tile_shapes.md |
| 参考实现 | models/examples/02_intermediate/operators/max_example.py |

---
*生成时间: 2026-03-29T15:08:00Z*
