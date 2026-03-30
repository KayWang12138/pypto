## 算子需求规范

### 1. 基础信息
- **算子名称**: hardswish
- **算子分类**: activation  <!-- element-wise / reduction / matmul / attention / custom -->
- **数学公式**: $hardswish(x) = x \cdot \frac{relu6(x + 3)}{6} = x \cdot \frac{min(max(x + 3, 0), 6)}{6}$
- **功能描述**: HardSwish激活函数，是Swish激活函数的硬件友好近似版本。广泛应用于移动端网络（如MobileNetV3）。计算过程包含三个步骤：1) x + 3；2) relu6；3) 除以6后与x相乘。

### 2. 关键特性
<!-- 复杂算子必须填写，简单算子可省略 -->
本算子为简单element-wise操作，无复杂特性。

### 3. 算法描述
<!-- 当公式无法完整表述计算流程时填写，简单算子省略此节 -->
简单算子，公式已完整描述计算逻辑，无需算法描述。

### 4. 数据流图

```
    输入 x                    输出 y
┌──────────────┐         ┌──────────────┐
│  [d1, d2, ...] │ ──────▶ │  [d1, d2, ...] │
│   float32     │hardswish│   float32     │
└──────────────┘         └──────────────┘

公式: hardswish(x) = x * relu6(x + 3) / 6
     = x * min(max(x + 3, 0), 6) / 6
     = x * hardsigmoid(x)

动态轴: 所有维度
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| x | [d1, d2, d3, d4] (1D-4D) | float32 | 所有维度 | 输入tensor，支持1D到4D |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y | [d1, d2, d3, d4] (与输入相同) | float32 | 所有维度 | 输出tensor，shape与输入一致 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 默认 |

### 7. 精度要求
- **atol**: 0.001
- **rtol**: 0.001

### 8. 动态轴说明
- **动态轴**: 所有维度（d1, d2, d3, d4）
- **轴含义**: 通用维度，可表示[batch, seq, features]等
- **取值范围**: [1, INT32_MAX]

### 9. 边界条件处理
- **零值**: 正常计算
- **极值**: 正常计算
- **NaN/Inf**: 正常计算（未特殊处理）

### 10. 性能要求
- **性能目标**: 无特殊要求

### 11. 参考信息
- **参考实现**: PyTorch `torch.nn.functional.hardswish`
- **论文**: Searching for MobileNetV3 (https://arxiv.org/abs/1905.02244)
- **类似算子**: hardsigmoid, swish, relu6

### 12. 应用场景
- **目标模型**: MobileNetV3, EfficientNet等移动端网络
- **使用位置**: 卷积层后的激活函数

**典型配置**（建议至少提供一个，用于下游 golden 验证和设计方案生成）:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | - | [1024, 1024] | [1024, 1024] | 核心性能场景，2D大tensor |
| 功能_P0_1D | 功能 | P0 | - | [1024] | [1024] | 1D功能验证 |
| 功能_P0_2D | 功能 | P0 | - | [128, 256] | [128, 256] | 2D功能验证 |
| 功能_P0_3D | 功能 | P0 | - | [16, 128, 256] | [16, 128, 256] | 3D功能验证（需reshape到2D） |
| 功能_P0_4D | 功能 | P0 | - | [4, 16, 128, 256] | [4, 16, 128, 256] | 4D功能验证（需reshape到2D） |

### 13. 实现要求

**必须遵循的实现约束**:

1. **动态轴支持**: MUST support dynamic axis - use implicit shape inference: `pypto.Tensor([], pypto.DT_FP32)`
2. **维度支持**: Support 1D-4D inputs
3. **3D+ reshape策略**: For 3D+ inputs, reshape to 2D to avoid tiling compilation issues
4. **Tiling配置**: Use `pypto.set_vec_tile_shapes(64, 128)` for 2D kernels

**实现方案建议**:

方案1（推荐）: 使用 PyPTO 内置 API
```python
# hardswish(x) = x * hardsigmoid(x)
# hardsigmoid(x) = relu6(x + 3) / 6
temp = pypto.add(x, 3.0)
temp = pypto.relu6(temp)
temp = pypto.div(temp, 6.0)
y = pypto.mul(x, temp)
```

方案2: 使用 maximum/minimum 实现 relu6
```python
# relu6(x) = min(max(x, 0), 6)
temp = pypto.add(x, 3.0)
temp = pypto.maximum(temp, 0.0)
temp = pypto.minimum(temp, 6.0)
temp = pypto.div(temp, 6.0)
y = pypto.mul(x, temp)
```

---
*生成时间: 2026-03-30T10:24:00Z*
*确认状态: 已确认*
