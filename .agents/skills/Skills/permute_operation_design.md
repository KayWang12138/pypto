# Permute Operation 需求开发文档

## 一、功能概述

### 1.1 功能描述

Permute Operation 用于对张量的维度进行重排列，功能等价于 PyTorch 的 `torch.permute(input, dims)` 函数。该操作返回原始张量的一个视图，其维度按照指定的顺序重新排列。

**重要约束**：本实现要求在所有场景中仅调用一次 Permute Operation，不通过多次调用 Transpose 来实现。

### 1.2 功能规格

#### PyTorch 参考实现

`torch.permute(input, dims) → Tensor`

**参数说明**：
- `input (Tensor)`: 输入张量
- `dims (torch.Size, tuple of int or list of int)`: 期望的维度顺序

**返回值**：
- 返回输入张量的视图，维度按照 `dims` 指定的顺序重排列

### 1.3 功能要求

| 要求项 | 规格 |
|--------|------|
| 支持维度数 | 2D ~ 5D |
| 数据类型 | FP16, FP32, INT16, INT32 |
| 视图语义 | 返回视图而非拷贝（在框架层面） |
| 负索引支持 | 支持 dims 参数中的负索引 |
| 恒等变换优化 | 当 dims = [0, 1, 2, ...] 时直接返回原张量 |
| 维度验证 | 验证 dims 包含所有维度且无重复 |
| 单次调用约束 | 所有场景仅调用一次 Permute Operation |

---

## 二、核心实现算法

### 2.1 算法设计思路

Permute 操作的核心挑战在于正确处理不同维度之间的映射关系。本实现采用**单次调用 + TGATHER 为主**的策略：

1. **维度映射验证**：验证 permutation 参数的有效性
2. **恒等变换优化**：检测恒等变换并直接返回
3. **TGATHER 硬件加速**：优先使用 PTO 的 TGATHER 指令，通过预先生成的索引 tile 一次性完成数据收集
4. **TTRANS 特例优化**：对于 2D 简单转置（`dims=[1,0]`），使用 TTRANS 指令获得最佳性能
5. **模板参数传递**：使用模板参数将维度映射关系传递到编译期

**核心改进**：
- 以 TGATHER 指令为主要实现方案，充分利用硬件加速
- 预先生成索引 tile，一次性完成数据收集，避免逐个元素复制的循环开销
- 保留 TTRANS 作为 2D 简单转置的最优解
- 所有场景都保证**单次调用**，不依赖多次 Transpose

### 2.2 维度映射算法

#### 2.2.1 正向映射与逆向映射

给定输入张量形状 `[N0, N1, N2, N3, N4]` 和 permutation `[P0, P1, P2, P3, P4]`：

**正向映射**：`output[i0][i1][i2][i3][i4] = input[iP0][iP1][iP2][iP3][iP4]`

**维度映射关系表**：构建 `axisMap` 数组，其中 `axisMap[outDim] = inDim` 表示输出维度 outDim 对应输入维度 inDim

**示例**：`dims = [2, 0, 1]`（3D）
- `axisMap[0] = 2`：输出维度 0 对应输入维度 2
- `axisMap[1] = 0`：输出维度 1 对应输入维度 0
- `axisMap[2] = 1`：输出维度 2 对应输入维度 1

#### 2.2.2 Shape 推导算法

根据 permutation 参数，输出张量的 Shape 按照维度映射关系重新排列。

#### 2.2.3 Stride 计算

对于输入张量，计算各个维度的 Stride（步长），用于索引计算。

**示例**：输入 Shape = [N0, N1, N2, N3]
- `inputStrides[0] = N1 * N2 * N3`
- `inputStrides[1] = N2 * N3`
- `inputStrides[2] = N3`
- `inputStrides[3] = 1`

### 2.3 Tile 切分算法

#### 2.3.1 Tile 策略

对于大张量，需要将其切分为适合硬件计算的 Tile。Permute 操作采用**输出导向的 Tile 切分策略**：

1. 按照输出张量的 Shape 进行 Tile 切分
2. 对每个输出 Tile，根据维度映射关系计算对应的输入 Tile 位置
3. 单次完成所有维度的重排

### 2.4 TileOP 层实现算法

TileOP 层采用**TGATHER 为主，TTRANS 为辅**的混合策略：
1. **通用方案**：优先使用 PTO 的 `TGATHER` 硬件指令，通过预先生成的索引 tile 一次性完成数据收集
2. **2D 特例**：对于 2D 张量的简单转置（`dims=[1,0]`），直接调用 PTO 的 `TTRANS` 硬件指令获得最优性能
3. **回退方案**：小尺寸 Tile 或不满足 TGATHER 约束时，使用通用的直接索引计算方案
4. **所有场景都确保单次调用**，不依赖多次 Transpose

#### 2.4.1 PTO TGATHER 指令介绍

PTO 提供了专门的 Gather 指令 `TGATHER`，用于通过索引 tile 高效地从源 tile 收集数据到目标 tile。

**数学解释**：
```
dst[i,j] = src[indices[i,j]]
```

**功能描述**：
- 对于目标 tile 的每个元素位置 `(i,j)`，根据索引 tile 中 `indices[i,j]` 指定的位置，从源 tile 收集对应元素
- 支持静态或动态索引 tile
- 适用于任意维度的 permutation 操作

**约束条件**：
- **A2A3 平台**：
  - 数据类型：int16/uint16/int32/uint32/half/float
  - 索引类型：int32/uint32
  - 数据类型和索引类型必须匹配
- **A5 平台**：
  - 数据类型：int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float/float8
  - 索引类型：int16/uint16/int32/uint32
  - 数据类型和索引类型必须匹配

#### 2.4.2 维度映射参数说明

模板参数 `axis0-axis4` 表示输出维度到输入维度的映射：

| dims 参数 | axis0 | axis1 | axis2 | axis3 | axis4 | dimCount | 说明 | 实现方式 |
|-----------|-------|-------|-------|-------|-------|----------|------|----------|
| [1, 0] | 1 | 0 | -1 | -1 | -1 | 2 | 2D 简单转置 | TTRANS 指令 |
| [0, 1] | 0 | 1 | -1 | -1 | -1 | 2 | 2D 恒等变换 | 直接返回 |
| [2, 0, 1] | 2 | 0 | 1 | -1 | -1 | 3 | 3D 排列 | TGATHER 指令 |
| [1, 2, 0] | 1 | 2 | 0 | -1 | -1 | 3 | 3D 排列 | TGATHER 指令 |
| [3, 0, 2, 1] | 3 | 0 | 2 | 1 | -1 | 4 | 4D 排列 | TGATHER 指令 |
| [4, 2, 0, 1, 3] | 4 | 2 | 0 | 1 | 3 | 5 | 5D 排列 | TGATHER 指令 |

**映射规则**：
- `axisX >= 0`：输出维度 X 对应输入维度 axisX
- `axisX < 0`：该维度不存在（用于填充不足 5D 的情况）
- `dimCount`：实际维度数（2-5）

**实现选择策略**：
- 2D 且 `[1, 0]` → 使用 TTRANS 硬件指令
- 恒等变换 → 直接返回（不执行任何操作）
- 其他情况 → 使用通用索引计算方案

---

## 三、Operation 层实现要点

### 3.1 核心数据结构

- **LogicalTensor**: 表示逻辑张量，包含 Shape、DataType、Format 等信息
- **TileInfo**: 表示 Tile 的 Shape 和 Offset 信息
- **LogicalInput**: 封装 LogicalTensor 和 TileInfo

### 3.2 关键函数

- **PermuteOperationOperandCheck**: 检查输入输出操作数的有效性
- **PermuteResultShape**: 计算输出张量的 Shape
- **IsIdentityPermutation**: 判断是否为恒等变换
- **NormalizePermutation**: 处理负索引
- **ValidatePermutation**: 验证 permutation 参数的有效性
- **TiledPermuteOperation**: 递归切分 Tile 并添加 Operation

### 3.3 Operation 属性设置

每个 Permute Operation 需要设置以下属性：
- `axis0` - `axis4`: 维度映射参数
- `dimCount`: 实际维度数

---

## 四、Codegen 层实现要点

### 4.1 代码生成策略

根据不同的场景选择不同的代码生成方式：
- **TileTensor 模式**: 使用 TileTensor 对象，代码更简洁
- **动态 Shape 模式**: 运行时传入 Shape 参数
- **静态 Shape 模式**: 编译期确定所有 Shape 参数

### 4.2 关键函数

- **GenPermuteOp**: 主入口函数，生成 Permute 操作代码
- **PrintPermuteTileTensor**: TileTensor 模式代码生成
- **PrintPermuteStatic**: 静态 Shape 代码生成
- **PrintPermuteDynamicUnaligned**: 动态 Shape 代码生成

---

## 五、开发任务清单

1. 在 `opcode.h` 中添加 `OP_PERMUTE` 操作码
2. 在 `opcode.cpp` 中注册 `OP_PERMUTE` 操作
3. 实现 `permute.h` 头文件定义
4. 实现 `permute.cpp` Operation 层代码
5. 实现 Codegen 层代码生成逻辑
6. 编写单元测试验证功能正确性

---

## 六、测试用例设计

### 6.1 基本功能测试

- 2D 张量转置 `[1, 0]`
- 3D 张量排列 `[2, 0, 1]`, `[1, 2, 0]`
- 4D 张量排列
- 5D 张量排列

### 6.2 边界条件测试

- 恒等变换 `[0, 1, 2, ...]`
- 负索引支持
- 维度验证（重复维度、缺失维度）

### 6.3 数据类型测试

- FP16, FP32
- INT16, INT32

---

## 七、注意事项

1. **单次调用约束**: 所有场景必须保证只调用一次 Permute Operation
2. **恒等变换优化**: 当 permutation 为恒等变换时，直接返回原张量
3. **负索引处理**: 需要将负索引转换为正索引
4. **维度验证**: 必须验证 permutation 包含所有维度且无重复
5. **临时空间**: Permute 操作需要临时空间用于数据重排
