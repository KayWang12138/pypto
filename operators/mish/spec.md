## 算子需求规范

### 1. 基础信息
- **算子名称**: mish
- **算子分类**: element-wise (activation)
- **数学公式**: $mish(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$
- **功能描述**: Mish 是一种平滑的激活函数，通过将输入与 tanh(softplus(x)) 相乘实现非线性变换。相比 ReLU，Mish 提供了更平滑的梯度流，在深度网络中表现更好。

### 2. 关键特性
<!-- 简单逐元素算子，无需复杂特性 -->

不适用（简单逐元素激活函数）

### 3. 算法描述
<!-- 公式足以描述计算流程，无需算法描述 -->

不适用（逐元素计算，公式已完整描述计算逻辑）

### 4. 数据流图

```
    输入 x                         输出 y
┌──────────────────┐         ┌──────────────────┐
│  [1D-4D tensor]  │         │  [1D-4D tensor]  │
│    float32       │ ──────▶ │    float32       │
└──────────────────┘  mish   └──────────────────┘

计算分解:
  step1: sp = ln(1 + exp(x))     # softplus
  step2: t = tanh(sp)            # tanh
  step3: y = x * t               # element-wise multiply

动态轴: 支持所有维度动态 (使用隐式shape推断)
特殊处理: 4D输入需reshape为2D避免tiling编译问题
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| x | [1D-4D] | float32 | 所有维度 | 输入张量，支持1D到4D |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y | [与输入相同] | float32 | 所有维度 | 输出张量，shape与输入一致 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | Yes | 0.001 | 0.001 | 默认，主要支持类型 |

### 7. 精度要求
- **atol**: 0.001
- **rtol**: 0.001

### 8. 动态轴说明
- **动态轴**: 所有维度 (dim0, dim1, dim2, dim3)
- **轴含义**:
  - 1D: [N] - 元素个数
  - 2D: [M, N] - M行N列
  - 3D: [B, S, D] - Batch, Sequence, Dimension
  - 4D: [B, H, W, C] - Batch, Height, Width, Channel (或 [B, S, N, D])
- **取值范围**: [1, INT32_MAX]
- **隐式shape推断**: 使用 `pypto.Tensor([], pypto.DT_FP32)` 进行动态shape推断

### 9. 边界条件处理
- **零值**: 正常计算 (mish(0) = 0 * tanh(ln(1+1)) = 0)
- **极值**:
  - x -> +inf: mish(x) -> x * tanh(ln(exp(x))) = x * tanh(x) -> x * 1 = x
  - x -> -inf: mish(x) -> x * tanh(0) = x * 0 = 0
- **NaN/Inf**: 正常传播，不特殊处理

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的2倍
- **Tiling配置**: 使用 `pypto.set_vec_tile_shapes(64, 128)` 用于2D kernel

### 11. 参考信息
- **参考实现**:
  - PyTorch: `torch.nn.functional.mish` (PyTorch 1.9+)
  - 手动实现: `x * torch.tanh(torch.nn.functional.softplus(x))`
- **论文**: Mish: A Self Regularized Non-Monotonic Activation Function (Diganta Misra, 2019)
- **类似算子**: swish/silu, gelu, relu

### 12. 应用场景
- **目标模型**: YOLOv4, ResNet变体, EfficientNet等现代神经网络
- **使用位置**: 隐藏层激活函数，替代ReLU/GELU

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_1D | 功能 | P0 | - | [1024] | [1024] | 1D基础功能验证 |
| 功能_2D | 功能 | P0 | - | [128, 1024] | [128, 1024] | 2D基础功能验证 |
| 功能_3D | 功能 | P0 | - | [2, 128, 1024] | [2, 128, 1024] | 3D功能验证 |
| 功能_4D | 功能 | P0 | - | [2, 4, 128, 1024] | [2, 4, 128, 1024] | 4D功能验证(reshape为2D处理) |
| 性能_2D | 性能 | P0 | - | [4096, 4096] | [4096, 4096] | 核心性能场景 |

### 13. 实现约束
- **动态轴支持**: 使用隐式shape推断 `pypto.Tensor([], pypto.DT_FP32)`
- **4D输入处理**: 将4D输入reshape为2D (flatten前两维) 避免tiling编译问题
- **Tiling配置**: 2D kernel使用 `pypto.set_vec_tile_shapes(64, 128)`

---
*生成时间: 2026-03-30T09:15:00Z*
*确认状态: 自动确认（非交互模式）*
