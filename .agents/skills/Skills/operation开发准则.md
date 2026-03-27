# Operation 开发准则

## 一、概述

Operation 是 PyPTO 框架的接口层，负责将 Python 前端的算子调用转换为底层的 Tile 操作。Operation 层是连接前端和后端的关键桥梁，负责张量切分、广播处理、类型检查等核心功能。

### 架构位置

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python前端层                              │
│  python/pypto/op/math.py                                        │
│  └─> pypto_impl.Add(input, other)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Operation接口层                               │
│  interface/operation/vector/binary.cpp                          │
│  └─> Add(function, tileShape, input1, input2)                   │
│      └─> TiledBinaryOperation<ADD>(...)                         │
│          └─> function.AddOperation(OP_ADD, {in1, in2}, {out})   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Function管理层                                │
│  interface/function/function.h                                  │
│  └─> operations_.push_back(op)                                  │
│  └─> CodeGen(function) 编译                                     │
└─────────────────────────────────────────────────────────────────┘
```

## 二、核心职责

### 2.1 主要功能

1. **张量切分（Tiling）**：将大张量切分为适合硬件计算的 Tile
2. **广播处理**：处理不同 Shape 张量之间的广播操作
3. **类型检查**：验证输入输出的数据类型和格式
4. **Shape 推导**：计算输出张量的 Shape
5. **Operation 创建**：创建 Operation 对象并添加到 Function

### 2.2 核心数据结构

```cpp
// Operation 类定义
class Operation : public std::enable_shared_from_this<Operation>, public AttrHolder {
public:
    LogicalTensors iOperand;    // 输入操作数
    LogicalTensors oOperand;    // 输出操作数
    LogicalTensors dependOperand; // 依赖操作数
    Opcode opCode;              // 操作码
    
    // 构造函数
    Operation(Function &cur, Opcode opcode, LogicalTensors iOperands, LogicalTensors oOperands);
    
    // 属性设置
    void SetAttribute(const std::string &key, const Any &value);
    void SetAttr(const std::string &key, const Any &value);
};
```

## 三、目录结构

```
interface/operation/
├── vector/                    # 向量运算
│   ├── binary.cpp/.h         # 二元运算
│   ├── unary.cpp/.h          # 一元运算
│   ├── compare.cpp           # 比较运算
│   ├── reduction.cpp         # 规约运算
│   ├── permute.cpp           # 维度变换
│   ├── indexing.cpp          # 索引操作
│   ├── where.cpp             # 条件选择
│   └── tensor_transformation.h # 张量变换
├── distributed/              # 分布式操作
│   ├── moe_dispatch.cpp
│   └── shmem_operation_impl.cpp
├── operation.h               # Operation 基类定义
├── operation.cpp             # Operation 实现
├── opcode.h                  # 操作码定义
├── opcode.cpp                # 操作码实现
├── attribute.h               # 属性定义
├── attr_holder.h             # 属性持有者
├── verifier.h                # 验证器
└── tile_shape.cpp            # Tile Shape 处理
```

## 四、开发规范

### 4.1 命名规范

| 类型 | 命名规则 | 示例 |
|------|----------|------|
| Operation 函数 | 操作名（首字母大写） | `Add`, `Sub`, `Mul` |
| Tile 函数 | Tiled + 操作名 + Operation | `TiledBinaryOperation` |
| 检查函数 | 操作名 + OperandCheck | `BinaryOperationOperandCheck` |
| Opcode | OP_ + 操作名（大写） | `OP_ADD`, `OP_SUB` |
| 枚举类型 | 操作类型 + OpType | `BinaryOpType`, `UnaryOpType` |

### 4.2 必须包含的头文件

```cpp
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/operator_tracer.h"
#include "interface/utils/vector_error.h"
```

## 五、开发步骤

### 5.1 定义 Opcode

在 `opcode.h` 中添加新的操作码：

```cpp
enum class Opcode {
    // Binary Vector
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MY_NEW_OP,  // 新增操作码
    // ...
};
```

### 5.2 定义操作类型枚举

```cpp
enum class BinaryOpType {
    ADD,
    SUB,
    MUL,
    DIV,
    MY_NEW_OP,  // 新增操作类型
    // ...
};
```

### 5.3 实现 Opcode 映射函数

```cpp
template <BinaryOpType T>
Opcode GetBinaryOpNameCode() {
    switch (T) {
        case BinaryOpType::ADD: return Opcode::OP_ADD;
        case BinaryOpType::SUB: return Opcode::OP_SUB;
        case BinaryOpType::MY_NEW_OP: return Opcode::OP_MY_NEW_OP;
        default: ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown binary op type";
    }
}
```

### 5.4 实现输入检查函数

```cpp
void MyNewOpOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, 
    const std::vector<LogicalTensorPtr> &oOperand) {
    // 检查输入数量
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 2) 
        << "iOperand size should be 2";
    
    // 检查输出数量
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) 
        << "oOperand size should be 1";
    
    // 检查数据类型
    auto input1 = iOperand[0];
    auto input2 = iOperand[1];
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, input1->Datatype() == input2->Datatype()) 
        << "The dtype of input tensors are not same.";
    
    // 检查 Shape 兼容性
    CheckBinOpOperandsValid(input1, input2);
}
```

### 5.5 实现 Tiled 操作函数

```cpp
template <BinaryOpType T>
void TiledMyNewOp(
    Function &function, 
    const TileShape &tileShape, 
    size_t cur, 
    LogicalInput &input1,
    LogicalInput &input2, 
    const LogicalTensorPtr &result, 
    TileInfo &resultTileInfo) {
    
    // 递归终止条件：完成所有维度的切分
    if (cur == input1.tensor->GetShape().size()) {
        // 创建 Tile 视图
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        
        // 添加 Operation 到 Function
        function.AddOperation(
            GetBinaryOpNameCode<T>(), 
            {inputTile1, inputTile2}, 
            {resultTile}
        );
        return;
    }
    
    // 递归切分每个维度
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        
        TiledMyNewOp<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}
```

### 5.6 实现顶层接口函数

```cpp
template <BinaryOpType T>
LogicalTensorPtr TensorMyNewOp(Function &function, const Tensor &operand1, const Tensor &operand2) {
    // 1. 获取存储张量
    auto oprandT1 = operand1.GetStorage();
    auto oprandT2 = operand2.GetStorage();
    
    // 2. 处理广播
    if (oprandT1->shape.size() != oprandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(oprandT1, oprandT2);
        oprandT1 = BinaryOperationBroadCast(oprandT1, broadCastShape);
        oprandT2 = BinaryOperationBroadCast(oprandT2, broadCastShape);
    }
    
    // 3. 检查输入
    auto opName = GetBinaryOpName<T>();
    CheckBinaryInputTensors(oprandT1, oprandT2, opName);
    
    // 4. 计算输出 Shape
    std::vector<int64_t> resultShape = BinaryOperationResultShape(oprandT1, oprandT2);
    
    // 5. 创建输出张量
    auto result = std::make_shared<LogicalTensor>(
        function, 
        oprandT1->Datatype(), 
        resultShape
    );
    
    // 6. 获取 TileShape
    auto tileShape = function.GetTileShape(oprandT1->shape, oprandT1->Datatype());
    
    // 7. 执行 Tiled 操作
    TiledMyNewOp<T>(function, tileShape, oprandT1, oprandT2, result);
    
    return result;
}
```

### 5.7 实现用户接口

```cpp
Tensor MyNewOp(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();
    
    RETURN_CALL(
        TensorMyNewOp<BinaryOpType::MY_NEW_OP>, 
        *Program::GetInstance().GetCurrentFunction(), 
        operand1, 
        operand2
    );
}
```

## 六、关键工具函数

### 6.1 Shape 处理

```cpp
// 计算二元操作结果 Shape
std::vector<int64_t> BinaryOperationResultShape(LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    std::vector<int64_t> resultShape(operand1->shape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        resultShape[i] = std::max(operand1->shape[i], operand2->shape[i]);
    }
    return resultShape;
}

// 广播张量
LogicalTensorPtr BinaryOperationBroadCast(const LogicalTensorPtr &operand, const std::vector<int> &broadCastShape);

// 检查操作数有效性
void CheckBinOpOperandsValid(const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2);
```

### 6.2 广播处理

```cpp
void BroadcastOperandTensor(
    LogicalTensorPtr &operand, 
    LogicalTensorPtr &other, 
    LogicalTensorPtr result,
    Function& function, 
    const TileShape& tileShape) {
    
    auto dstShape = result->shape;
    if (operand->shape == dstShape) {
        return;
    }
    
    // 创建扩展后的张量
    auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
    Expand(function, tileShape, operand, {other}, expanded);
    operand = expanded;
}
```

### 6.3 Tile Info 结构

```cpp
struct LogicalInput {
    const LogicalTensorPtr tensor;
    TileInfo tileInfo;
};

struct TileInfo {
    std::vector<int64_t> shape;
    std::vector<int64_t> offset;
};
```

## 七、属性设置

### 7.1 Operation 属性键

```cpp
class OpAttributeKey {
public:
    static const std::string aicpuCall;
    static const std::string scalar;
    static const std::string broadcastLastAxis;
    static const std::string inplaceIdx;
    static const std::string lastUse;
    static const std::string rowPad;
    // ...
};
```

### 7.2 设置属性

```cpp
// 设置 Operation 属性
auto &op = function.AddOperation(Opcode::OP_PRELU, {tile, weightTile}, {resultTile, tmpTensor});
op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

// 设置 OpAttributeKey 属性
op.SetAttr(OpAttributeKey::rowPad, dimMap);
```

## 八、特殊场景处理

### 8.1 需要临时空间的操作

```cpp
template <BinaryOpType T>
void TiledBinaryOperation(...) {
    if (withBrc) {
        // 创建临时张量
        std::vector<int64_t> tmpShape(input1.tileInfo.shape);
        auto alignSize = BLOCK_SIZE / BytesOf(input2.tensor->Datatype());
        tmpShape[input1.tileInfo.shape.size() - 1] = alignSize;
        
        auto tempTensor = std::make_shared<LogicalTensor>(function, input2.tensor->Datatype(), tmpShape);
        
        // 添加带临时空间的 Operation
        function.AddOperation(
            GetBinaryOpNameCode<T, false, true>(), 
            {inputTile1, inputTile2}, 
            {resultTile, tempTensor}
        );
    }
}
```

### 8.2 广播操作

```cpp
// 判断是否需要广播
bool CallBrcBinOp(LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    size_t shapeSize = operand1->shape.size();
    for (size_t i = 0; i < shapeSize - 1; ++i) {
        if (operand1->shape[i] != operand2->shape[i]) {
            return false;
        }
    }
    return ((operand1->shape[shapeSize - 1] != 1) && (operand2->shape[shapeSize - 1] == 1)) ||
           ((operand1->shape[shapeSize - 1] == 1) && (operand2->shape[shapeSize - 1] != 1));
}

// 判断是否使用广播操作
template <BinaryOpType T>
bool ShouldUseBrcOperation(Function &function, LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    bool isCombineAxisEnabled = 
        function.paramConfigs_.forceCombineAxis || 
        (function.paramConfigs_.combineAxis && IsOpInBrcWhitelist<T>());
    
    if (!isCombineAxisEnabled) {
        return false;
    }
    return CallBrcBinOp(operand1, operand2);
}
```

### 8.3 类型转换

```cpp
Tensor Rsqrt(const Tensor &self) {
    DECLARE_TRACER();
    
    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        // 类型转换为 FP32
        castSelf = CALL(CastOperation<CastOpType::CAST>, 
            *Program::GetInstance().GetCurrentFunction(),
            self.GetStorage(), 
            DataType::DT_FP32, 
            CastMode::CAST_NONE);
    }
    
    // 执行计算
    auto sqrtSelf = CALL(UnaryOperation<UnaryOpType::SQRT>, 
        *Program::GetInstance().GetCurrentFunction(), castSelf);
    auto ones = CALL(FullOperation, 
        *Program::GetInstance().GetCurrentFunction(), 
        Element(DataType::DT_FP32, 1.0),
        SymbolicScalar(), 
        DataType::DT_FP32, 
        self.GetShape(), 
        self.GetStorage()->GetDynValidShape());
    auto result = CALL(BinaryOperation<BinaryOpType::DIV>, 
        *Program::GetInstance().GetCurrentFunction(), ones, sqrtSelf);
    
    // 转换回原类型
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, 
            *Program::GetInstance().GetCurrentFunction(), 
            result,
            self.GetDataType(), 
            CastMode::CAST_NONE);
    }
    return result;
}
```

## 九、错误处理

### 9.1 断言宏

```cpp
// 使用 ASSERT 宏进行错误检查
ASSERT(VectorErrorCode::ERR_PARAM_INVALID, condition) << "error message";

// 常用错误码
namespace VectorErrorCode {
    constexpr int ERR_PARAM_INVALID = 1;
    constexpr int ERR_PARAM_DTYPE_UNSUPPORTED = 2;
    constexpr int ERR_PARAM_SHAPE_UNSUPPORTED = 3;
}
```

### 9.2 类型检查

```cpp
void CheckBinaryInputTensors(const LogicalTensorPtr &tensor1, const LogicalTensorPtr &tensor2, std::string &op) {
    CheckTensorShape(tensor1, op);
    CheckTensorShape(tensor2, op);
    CheckBinOpOperandsValid(tensor1, tensor2);
    
    if (tensor1->Datatype() != tensor2->Datatype()) {
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) 
            << "The dtype of input tensors are not same.";
    }
    
    if (tensor1->Format() != tensor2->Format()) {
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) 
            << "The format of input tensors are not same.";
    }
}
```

## 十、最佳实践

### 10.1 代码组织

1. **分离声明和实现**：头文件放声明，cpp 文件放实现
2. **模板函数放在头文件**：模板函数必须在头文件中实现
3. **使用命名空间**：所有代码放在 `npu::tile_fwk` 命名空间

### 10.2 性能优化

1. **避免不必要的拷贝**：使用引用和智能指针
2. **预计算 Shape**：在循环外计算 Shape 相关信息
3. **合理使用广播**：仅在必要时进行广播操作

### 10.3 可维护性

1. **添加详细注释**：说明函数功能、参数含义
2. **使用有意义的命名**：变量和函数名应清晰表达意图
3. **保持函数简洁**：每个函数只做一件事

## 十一、完整示例

```cpp
// my_new_op.h
#pragma once

#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {

enum class MyNewOpType {
    CUSTOM_OP,
};

template <MyNewOpType T>
Opcode GetMyNewOpNameCode() {
    switch (T) {
        case MyNewOpType::CUSTOM_OP: return Opcode::OP_MY_NEW_OP;
        default: ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown op type";
    }
}

void MyNewOpOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, 
    const std::vector<LogicalTensorPtr> &oOperand);

template <MyNewOpType T>
void TiledMyNewOp(
    Function &function, 
    const TileShape &tileShape, 
    size_t cur, 
    LogicalInput &input1,
    LogicalInput &input2, 
    const LogicalTensorPtr &result, 
    TileInfo &resultTileInfo);

template <MyNewOpType T>
LogicalTensorPtr TensorMyNewOp(Function &function, const Tensor &operand1, const Tensor &operand2);

Tensor MyNewOp(const Tensor &operand1, const Tensor &operand2);

} // namespace npu::tile_fwk


// my_new_op.cpp
#include "my_new_op.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/utils/vector_error.h"

namespace npu::tile_fwk {

void MyNewOpOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, 
    const std::vector<LogicalTensorPtr> &oOperand) {
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 2) 
        << "iOperand size should be 2";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) 
        << "oOperand size should be 1";
    
    auto input1 = iOperand[0];
    auto input2 = iOperand[1];
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, input1->Datatype() == input2->Datatype()) 
        << "The dtype of input tensors are not same.";
}

template <MyNewOpType T>
void TiledMyNewOp(
    Function &function, 
    const TileShape &tileShape, 
    size_t cur, 
    LogicalInput &input1,
    LogicalInput &input2, 
    const LogicalTensorPtr &result, 
    TileInfo &resultTileInfo) {
    
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        
        function.AddOperation(GetMyNewOpNameCode<T>(), {inputTile1, inputTile2}, {resultTile});
        return;
    }
    
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        TiledMyNewOp<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}

template <MyNewOpType T>
LogicalTensorPtr TensorMyNewOp(Function &function, const Tensor &operand1, const Tensor &operand2) {
    auto oprandT1 = operand1.GetStorage();
    auto oprandT2 = operand2.GetStorage();
    
    if (oprandT1->shape.size() != oprandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(oprandT1, oprandT2);
        oprandT1 = BinaryOperationBroadCast(oprandT1, broadCastShape);
        oprandT2 = BinaryOperationBroadCast(oprandT2, broadCastShape);
    }
    
    std::vector<int64_t> resultShape = BinaryOperationResultShape(oprandT1, oprandT2);
    auto result = std::make_shared<LogicalTensor>(function, oprandT1->Datatype(), resultShape);
    
    auto tileShape = function.GetTileShape(oprandT1->shape, oprandT1->Datatype());
    
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    
    auto input1 = LogicalInput{oprandT1, tileInfo1};
    auto input2 = LogicalInput{oprandT2, tileInfo2};
    
    TiledMyNewOp<T>(function, tileShape, 0, input1, input2, result, resultTileInfo);
    
    return result;
}

Tensor MyNewOp(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();
    RETURN_CALL(TensorMyNewOp<MyNewOpType::CUSTOM_OP>, 
        *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

// 显式实例化模板
template void TiledMyNewOp<MyNewOpType::CUSTOM_OP>(
    Function &, const TileShape &, size_t, LogicalInput &, LogicalInput &, 
    const LogicalTensorPtr &, TileInfo &);

template LogicalTensorPtr TensorMyNewOp<MyNewOpType::CUSTOM_OP>(
    Function &, const Tensor &, const Tensor &);

} // namespace npu::tile_fwk
```

## 十二、参考文件

- [binary.cpp](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/interface/operation/vector/binary.cpp) - 二元运算实现
- [binary.h](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/interface/operation/vector/binary.h) - 二元运算声明
- [unary.cpp](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/interface/operation/vector/unary.cpp) - 一元运算实现
- [operation.h](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/interface/operation/operation.h) - Operation 基类
- [opcode.h](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/interface/operation/opcode.h) - 操作码定义
