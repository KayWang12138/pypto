# 阶段三：PyPTO算子开发详细指南

**总体开发流程首先请参考PyPTO下.agents/skills/pypto-operator-develop-workflow这个skill，下方仅作为补充参考**

## 1. 开发规范

### 数据类型处理
- BF16/F16输入 → FP32中间计算 → BF16/F16输出
- 避免精度损失累积

### Kernel签名
- 使用静态shape（PyPTO要求）
- 所有tensor参数需要在签名中声明

### Tile设置
- 使用 `pypto.set_cube_tile_shapes()` 设置cube tile
- 使用 `pypto.set_vec_tile_shapes()` 设置vector tile
- 根据数据规模选择合适的tile size

### 循环处理
- 使用 `pypto.loop()` 进行分块循环
- 合理设置循环展开因子

## 2. 可选参数处理策略

根据阶段一的分析，选择合适的实现策略：

| 策略 | 适用场景 | 实现方式 |
|-----|---------|---------|
| 统一kernel | 可选参数少，可以用默认值填充 | kernel签名包含所有参数，调用时填充默认值 |
| 多kernel | 可选参数组合多，需要不同实现 | 为不同功能组合创建独立kernel |
| 部分实现 | 某些功能PyPTO不支持 | 实现支持的功能，文档说明限制 |
| 分阶段实现 | 先实现核心功能，后续扩展 | 先P0，后P1/P2 |

## 3. 特殊算子类型处理

### Attention类算子

如果算子内部包含attention计算：

1. **优先使用Flash Attention实现**
   - 使用 `pypto.loop` 分块计算
   - 实现online softmax

2. **失败时使用原版attention**
   - Flash实现编译/运行失败
   - 算子规格不支持分块（如动态shape）

### 归约类算子

注意：
- 中间结果的精度问题
- 分块归约的正确性

### 矩阵乘法类算子

注意：
- 设置合适的cube tile shapes
- 处理非对齐的shape