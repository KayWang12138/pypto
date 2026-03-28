# Permute Operation 开发总结

## 1. 概述

Permute Operation 实现了张量维度重排功能，等效于 PyTorch 的 `torch.permute`。该操作通过单次调用完成维度置换，支持 2D-5D 张量。

## 2. 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      Operation Layer                         │
│  permute.h / permute.cpp                                     │
│  - Permute() API 入口                                        │
│  - TensorPermuteOperation() 创建 LogicalTensor               │
│  - PermuteOperationTileFunc() Tile 切分                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       TileOP Layer                           │
│  tileop/vector/permute.h                                     │
│  - Tpermute<axis0, axis1, axis2, axis3, axis4, dimCount>    │
│  - 使用 TTRANS / TGATHER 指令实现                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Codegen Layer                           │
│  codegen_op_cloudnpu.cpp / codegen_vector_unary_with_tmp.cpp│
│  - PrintPermute() 代码生成                                   │
│  - PrintPermuteTileTensor() TileTensor 模式                 │
└─────────────────────────────────────────────────────────────┘
```

## 3. 核心实现逻辑

### 3.1 Operation 层

#### 3.1.1 操作码注册 (opcode.h / opcode.cpp)

```cpp
// opcode.h - 枚举定义
enum class Opcode {
    ...
    OP_TRANSPOSE_VNCHWCONV,
    OP_PERMUTE,  // 新增
    OP_LN,
    ...
};

// opcode.cpp - 操作注册
RegisterInfo(Opcode::OP_PERMUTE, OpCoreType::AIV, "PERMUTE", 
    {MemoryType::MEM_UB},                              // 输入: UB
    {MemoryType::MEM_UB, MemoryType::MEM_UB},          // 输出: 结果 + 临时缓冲区
    {"TileOp::Tpermute", PIPE_V, PIPE_V, CoreType::AIV},
    OpCalcType::OTHER, 
    {OP_ATTR_PREFIX + "axis0", OP_ATTR_PREFIX + "axis1", OP_ATTR_PREFIX + "axis2",
     OP_ATTR_PREFIX + "axis3", OP_ATTR_PREFIX + "axis4", OP_ATTR_PREFIX + "dimCount"},
    TileShapeVerifier::Verify);
```

**设计要点：**
- 输入为单个 UB 内存张量
- 输出包含结果张量和临时缓冲区（用于 TGATHER 指令的索引计算）
- 属性传递置换信息：`axis0-axis4` 表示各维度的源位置，`dimCount` 表示实际维度数

#### 3.1.2 Permute API 实现

```cpp
Tensor Permute(const Tensor &self, std::vector<int> perm) {
    // 1. 参数校验
    const int shapeSize = self.GetShape().size();
    ASSERT(perm.size() == shapeSize);  // 维度数匹配
    ASSERT(shapeSize >= 2 && shapeSize <= 5);  // 支持 2D-5D
    
    // 2. 归一化负索引
    NormalizePermutation(perm, shapeSize);  // -1 -> shapeSize-1
    
    // 3. 验证置换有效性
    ValidatePermutation(perm, shapeSize);  // 无重复、范围正确
    
    // 4. 恒等变换优化
    if (IsIdentityPermutation(perm)) {
        return self;  // 直接返回原张量
    }
    
    // 5. 2D 转置优化
    if (shapeSize == 2 && perm[0] == 1 && perm[1] == 0) {
        return Transpose(self, {0, 1});  // 使用现有 Transpose
    }
    
    // 6. 创建 Permute Operation
    return TensorPermuteOperation(function, self.GetStorage(), perm);
}
```

#### 3.1.3 Operation 创建

```cpp
LogicalTensorPtr TensorPermuteOperation(Function &function, LogicalTensorPtr self, 
                                         const std::vector<int> &perm) {
    // 1. 计算输出形状
    std::vector<int64_t> resultShape = PermuteResultShape(self->shape, perm);
    // 例: input [2,3,4], perm [1,0,2] -> output [3,2,4]
    
    // 2. 创建输出 LogicalTensor
    auto result = std::make_shared<LogicalTensor>(function, self->datatype, resultShape, ...);
    
    // 3. 添加 Operation
    auto &op = function.AddOperation(Opcode::OP_PERMUTE, {self}, {result});
    
    // 4. 设置属性
    op.SetAttribute("axis0", perm[0]);
    op.SetAttribute("axis1", perm[1]);
    op.SetAttribute("axis2", perm.size() > 2 ? perm[2] : -1);
    op.SetAttribute("axis3", perm.size() > 3 ? perm[3] : -1);
    op.SetAttribute("axis4", perm.size() > 4 ? perm[4] : -1);
    op.SetAttribute("dimCount", perm.size());
    
    return result;
}
```

### 3.2 TileOP 层

#### 3.2.1 模板函数设计

```cpp
template <int axis0, int axis1, int axis2, int axis3, int axis4, int dimCount,
          typename T0, typename T1, typename T2>
TILEOP void Tpermute(T0 dst, T1 src, T2 tmp) {
    // 模板参数在编译时确定，实现高效分支
}
```

**模板参数说明：**
- `axis0-axis4`: 输出维度 i 对应的输入维度位置
- `dimCount`: 实际维度数 (2-5)
- `T0, T1, T2`: 目标、源、临时张量类型

#### 3.2.2 硬件指令选择策略

```cpp
if constexpr (dimCount == 2 && axis0 == 1 && axis1 == 0) {
    // 情况1: 2D 标准转置 [N, M] -> [M, N]
    // 使用 TTRANS 指令，硬件优化
    pto::TTRANS(dstTile, srcTile, tmpTile);
}
else if constexpr (dimCount == 2 || dimCount == 3 || dimCount == 4) {
    // 情况2: 2D-4D 一般置换
    // 使用 TGATHER 指令，通过索引重排数据
    pto::TGATHER(dstTile, srcTile, idxTile, tmpIdxTile);
}
else if constexpr (dimCount == 5) {
    // 情况3: 5D 置换
    // 使用标量元素拷贝（循环展开）
    dstAddr[dstOffset] = srcAddr[srcOffset];
}
```

#### 3.2.3 偏移计算核心算法

```cpp
// 输出坐标 -> 输入坐标映射
// 例: perm = [1, 0, 2], 输出坐标 (i0, i1, i2)
// 输入坐标 = (i1, i0, i2)

int64_t srcIdx0 = (axis0 == 0) ? i0 : ((axis0 == 1) ? i1 : ...);
int64_t srcIdx1 = (axis1 == 0) ? i0 : ((axis1 == 1) ? i1 : ...);
// ...

// 计算内存偏移
auto srcOffset = srcIdx0 * srcStride0 + srcIdx1 * srcStride1 + ...;
auto dstOffset = i0 * dstStride0 + i1 * dstStride1 + ...;
```

### 3.3 Codegen 层

#### 3.3.1 代码生成逻辑

```cpp
std::string CodeGenOpCloudNPU::PrintPermuteTileTensor() const {
    // 1. 获取张量名称
    std::string dstTensor = QueryTileTensorNameByIdx(DST_IDX);
    std::string srcTensor = QueryTileTensorNameByIdx(SRC1_IDX);
    std::string tmpTensor = QueryTileTensorNameByIdx(SRC0_IDX);
    
    // 2. 获取属性
    int axis0 = AnyCast<int64_t>(opAttrs.at("axis0"));
    int axis1 = AnyCast<int64_t>(opAttrs.at("axis1"));
    // ...
    
    // 3. 生成代码
    // Tpermute<1, 0, 2, -1, -1, 3>(dst, src, tmp);
    oss << tileOpName << "<" << axis0 << ", " << axis1 << ", " 
        << axis2 << ", " << axis3 << ", " << axis4 << ", " << dimCount << ">";
    oss << "(" << dstTensor << ", " << srcTensor << ", " << tmpTensor << ");\n";
    
    return oss.str();
}
```

## 4. 数据流示例

### 4.1 3D 张量置换

```
输入张量: shape = [2, 3, 4], stride = [12, 4, 1]
置换参数: perm = [1, 0, 2]  (交换维度0和1)
输出张量: shape = [3, 2, 4], stride = [8, 4, 1]

坐标映射:
  输出 (i0, i1, i2) -> 输入 (i1, i0, i2)
  
示例:
  输出坐标 (1, 0, 2) -> 输入坐标 (0, 1, 2)
  输出偏移 = 1*8 + 0*4 + 2*1 = 10
  输入偏移 = 0*12 + 1*4 + 2*1 = 6
  dst[10] = src[6]
```

### 4.2 代码生成示例

```cpp
// 输入: shape=[2,3,4], perm=[1,0,2]
// 生成的代码:
Tpermute<1, 0, 2, -1, -1, 3>(dst_tensor, src_tensor, tmp_tensor);

// 展开后执行:
for (int i0 = 0; i0 < 3; ++i0) {        // 输出维度0 = 输入维度1
    for (int i1 = 0; i1 < 2; ++i1) {    // 输出维度1 = 输入维度0
        for (int i2 = 0; i2 < 4; ++i2) {// 输出维度2 = 输入维度2
            // 坐标映射: (i0, i1, i2) -> (i1, i0, i2)
            dst[i0*8 + i1*4 + i2] = src[i1*12 + i0*4 + i2];
        }
    }
}
```

## 5. 优化策略

### 5.1 恒等变换优化
```cpp
if (IsIdentityPermutation(perm)) {
    return self;  // 直接返回，避免无意义计算
}
```

### 5.2 2D 转置特殊处理
```cpp
if (shapeSize == 2 && perm[0] == 1 && perm[1] == 0) {
    return Transpose(self, {0, 1});  // 复用优化的 Transpose 操作
}
```

### 5.3 编译时分支
使用 `if constexpr` 在编译时确定执行路径，消除运行时开销。

## 6. 文件清单

| 文件路径 | 修改类型 | 说明 |
|---------|---------|------|
| `interface/operation/opcode.h` | 修改 | 添加 OP_PERMUTE 枚举 |
| `interface/operation/opcode.cpp` | 修改 | 注册 OP_PERMUTE 操作 |
| `interface/operation/vector/permute.h` | 新建 | Operation 层头文件 |
| `interface/operation/vector/permute.cpp` | 重写 | Operation 层实现 |
| `interface/tileop/vector/permute.h` | 新建 | TileOP 层实现 |
| `codegen/cloudnpu/codegen_op_cloudnpu.cpp` | 修改 | 添加代码生成映射 |
| `codegen/cloudnpu/codegen_op_cloudnpu.h` | 修改 | 添加函数声明 |
| `codegen/cloudnpu/codegen_vector_unary_with_tmp.cpp` | 修改 | 实现代码生成 |

## 7. 支持规格

- **维度**: 2D - 5D
- **数据类型**: FP16, FP32, INT16, INT32
- **内存位置**: UB (Unified Buffer)
- **特殊支持**: 负索引、恒等变换优化
