# 卷积操作实现文档

## 概述
本实现参考 Cube Matmul 操作的设计，实现了一个基本的卷积操作(Conv2D)，支持两层 tile 展开：
1. **第一层展开**: 从 GM (全局内存) 往 L1 (一级缓存) 搬运数据
2. **第二层展开**: 从 L1 往 L0 (零级缓存) 搬运数据

## 文件结构

### 1. 新增文件

#### 测试文件
- **路径**: `/root/programspace/pypto/framework/tests/ut/passes/src/test_conv_operation.cpp`
- **功能**: 
  - 测试基本的 Conv2D 操作
  - 测试不同形状的输入
  - 验证 tile 展开后的 VIEW 操作是否正确标记

#### 实现文件
- **路径**: `/root/programspace/pypto/framework/src/interface/operation/conv_operation_impl.cpp`
- **功能**:
  - 实现卷积操作的核心逻辑
  - 实现两层 tile 展开
  - 添加卷积特定的属性标记

### 2. 修改文件

#### 头文件更新
- **文件**: `/root/programspace/pypto/framework/include/tilefwk/tilefwk_op.h`
- **修改**: 添加 `Conv` namespace 和 `Conv2D` 函数声明

- **文件**: `/root/programspace/pypto/framework/src/interface/operation/operation_impl.h`
- **修改**: 添加 `Conv::ConstructConvTileGraph` 函数声明

#### 操作展开注册
- **文件**: `/root/programspace/pypto/framework/src/interface/operation/operation_impl.cpp`
- **修改**: 在 `ExpandOperationInto` 函数中添加 `OP_CONV` 的处理分支

## 设计细节

### 1. 输入输出规格

- **fmap (特征图)**: 5维张量 `[N, C1, H, W, C0]`
  - N: batch size
  - C1: 通道数的高维部分
  - H: 高度
  - W: 宽度
  - C0: 通道数的低维部分 (通常为16，用于 NZ 格式对齐)

- **weight (卷积核)**: 4维张量 `[Co, Ci, Kh, Kw]`
  - Co: 输出通道数
  - Ci: 输入通道数
  - Kh: 卷积核高度
  - Kw: 卷积核宽度

- **output (输出)**: 5维张量 `[N, Co1, H_out, W_out, Co0]`

### 2. 两层 Tile 展开

#### 第一层展开：GM → L1

**Fmap 搬运**:
```cpp
// 从 GM 搬运 fmap 的一个 tile 到 L1
fmapL1Shape = {1, c1Size, fmapH, fmapW, fmapC0}
fmapL1Offset = {n, c1Idx, 0, 0, 0}
MemoryType: MEM_L1
```

**Weight 搬运**:
```cpp
// 从 GM 搬运 weight 的一个 tile 到 L1
weightL1Shape = {coSize, weightCi, weightKh, weightKw}
weightL1Offset = {coIdx, 0, 0, 0}
MemoryType: MEM_L1
```

#### 第二层展开：L1 → L0

**Fmap L1 → L0A**:
```cpp
// 从 L1 搬运到 L0A (Cube 单元的 A 侧输入)
fmapL0Shape = {1, c1Size, hSize, wSize, fmapC0}
fmapL0Offset = {0, 0, hIdx, wIdx, 0}
MemoryType: MEM_L0A
```

**Weight L1 → L0B**:
```cpp
// 从 L1 搬运到 L0B (Cube 单元的 B 侧输入)
weightL0Shape = {coSize_tile, weightCi, weightKh, weightKw}
weightL0Offset = {0, 0, 0, 0}
MemoryType: MEM_L0B
```

### 3. 卷积标记属性

所有与卷积相关的 VIEW 操作都会添加以下属性标记：

```cpp
const std::string CONV_OP_MARKER = "op_attr_is_conv_op";      // 标记是否为卷积操作
const std::string CONV_FMAP_H = "op_attr_conv_fmap_h";        // fmap 高度
const std::string CONV_FMAP_W = "op_attr_conv_fmap_w";        // fmap 宽度
const std::string CONV_KERNEL_H = "op_attr_conv_kernel_h";    // 卷积核高度
const std::string CONV_KERNEL_W = "op_attr_conv_kernel_w";    // 卷积核宽度
```

这些属性在以下位置设置：
1. **GM → L1 的 VIEW 操作**: 设置 `CONV_OP_MARKER = true`
2. **L1 → L0 的 VIEW 操作**: 设置 `CONV_OP_MARKER = true`
3. **Conv 操作节点**: 设置所有卷积相关属性

### 4. ExpandFunction 集成

在 `expand_function.cpp` 的 `ExpandOperationInto` 函数中，添加了对 `OP_CONV` 的处理：

```cpp
case Opcode::OP_CONV: {
    Conv::ConstructConvTileGraph(function, tileShape, iOperand, oOperand[0], op);
    break;
}
```

当遇到 `OP_CONV` 操作时，会调用 `ConstructConvTileGraph` 进行 tile 展开，生成多个 VIEW 操作来实现数据在不同存储层级间的搬运。

## 使用示例

```cpp
// 定义输入张量
std::vector<int64_t> fmapShape{1, 4, 56, 56, 16};     // [N, C1, H, W, C0]
std::vector<int64_t> weightShape{64, 64, 3, 3};      // [Co, Ci, Kh, Kw]

Tensor fmap(DT_FP16, fmapShape, "fmap");
Tensor weight(DT_FP16, weightShape, "weight");

// 设置 tile 形状
TileShape::Current().SetCubeTile({
    {16, 64},      // M dimension [L0, L1]
    {16, 64, 64},  // K dimension [L0, L1a, L1b]
    {16, 64}       // N dimension [L0, L1]
});

// 调用卷积操作
FUNCTION("MyConv") {
    Tensor output = npu::tile_fwk::Conv::Conv2D(DT_FP16, fmap, weight);
}
```

## 关键特性

1. **两层 Tile 展开**: 严格遵循 GM → L1 → L0 的数据流动路径
2. **使用 OP_VIEW**: 所有数据搬运操作都使用 VIEW 操作实现
3. **卷积标记**: 通过 attribute 标记区分卷积相关的 VIEW 操作
4. **自动化编译**: 通过 CMakeLists.txt 的 glob 模式自动包含新文件
5. **测试覆盖**: 提供单元测试验证实现正确性

## 后续优化方向

1. 支持更多卷积参数（stride, padding, dilation）
2. 优化 tile 大小计算策略
3. 支持更多数据格式（ND, NZ）
4. 添加性能优化（如数据复用）
5. 支持 bias 和 activation fusion
