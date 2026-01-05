# 如何添加新的 Operation

> **适用对象：** 想要在PyPTO框架中添加新操作的开发者  
> **学习时间：** 60-90分钟  
> **前置知识：** 已阅读[Operation类技术文档](../02-core/06-operation.md)和[核心概念](../02-core/01-concepts.md)  
> **学习目标：** 理解如何添加新的Operation，包括Opcode定义、形状推断、Tiled函数注册等

## 概述

在PyPTO框架中添加新的Operation需要修改多个地方，包括Opcode定义、形状推断函数、Tiled函数实现、注册机制等。本文档详细说明添加新Operation的完整流程。

**相关文档：**
- [Operation类技术文档](../02-core/06-operation.md) - Operation类详细说明
- [核心概念](../02-core/01-concepts.md) - 基础概念和术语
- [Codegen模块](../02-core/10-codegen.md) - 代码生成机制

---

## 目录

1. [添加Operation的完整流程](#添加operation的完整流程)
2. [步骤1：定义Opcode](#步骤1定义opcode)
3. [步骤2：实现形状推断函数](#步骤2实现形状推断函数)
4. [步骤3：实现Tiled函数（可选）](#步骤3实现tiled函数可选)
5. [步骤4：实现Operator函数（可选）](#步骤4实现operator函数可选)
6. [步骤5：注册机制](#步骤5注册机制)
7. [步骤6：Python绑定（可选）](#步骤6python绑定可选)
8. [完整示例](#完整示例)
9. [验证与测试](#验证与测试)
10. [特殊Pass和模块考虑](#特殊pass和模块考虑)
11. [特殊场景考虑](#特殊场景考虑)
12. [常见问题](#常见问题)

---

## 添加Operation的完整流程

### 流程图

```
┌─────────────────────────────────────────────────────────┐
│           添加新Operation的完整流程                       │
└─────────────────────────────────────────────────────────┘

1. 定义Opcode
   └─→ framework/src/interface/operation/opcode.h
   
2. 实现形状推断函数
   └─→ framework/src/interface/operation/vector/ 或相应目录
   └─→ 注册到 InferShapeRegistry
   
3. 实现Tiled函数（可选）
   └─→ framework/src/interface/operation/vector/ 或相应目录
   └─→ 注册到 TiledFuncRegistry
   
4. 实现Operator函数（可选）
   └─→ framework/src/operator/ 相应目录
   └─→ 提供高级API
   
5. Python绑定（可选）
   └─→ python/pypto/op/ 或相应目录
   └─→ 提供Python接口
   
6. 验证与测试
   └─→ 编写测试用例
   └─→ 验证功能正确性
```

### 修改文件清单

| 步骤 | 需要修改的文件 | 说明 |
|------|--------------|------|
| **1. 定义Opcode** | `framework/src/interface/operation/opcode.h` | 在Opcode枚举中添加新操作码 |
| **2. 形状推断** | `framework/src/interface/operation/vector/*.cpp` | 实现形状推断函数并注册 |
| **3. Tiled函数** | `framework/src/interface/operation/vector/*.cpp` | 实现Tiled函数并注册（可选） |
| **4. Operator函数** | `framework/src/operator/*/*.cpp` | 实现高级API（可选） |
| **5. Python绑定** | `python/pypto/op/*.py` | 提供Python接口（可选） |
| **6. 测试** | `python/pypto/tests/` | 编写测试用例 |

---

## 步骤1：定义Opcode

### 1.1 在Opcode枚举中添加新操作码

**文件位置**：`framework/src/interface/operation/opcode.h`

**操作步骤**：

1. **确定操作码位置**：根据操作类型选择合适的分类
   - Unary Vector：一元向量操作（如`OP_EXP`、`OP_NEG`）
   - Binary Vector：二元向量操作（如`OP_ADD`、`OP_SUB`）
   - Cube：矩阵运算（如`OP_A_MUL_B`、`OP_CONV`）
   - View：视图操作（如`OP_VIEW`、`OP_RESHAPE`）
   - Move：数据移动（如`OP_COPY_IN`、`OP_COPY_OUT`）
   - Call：函数调用（如`OP_CALL`）

2. **添加操作码定义**：

```cpp
enum class Opcode {
    // ... 现有操作码 ...
    
    // 新操作码（示例：添加一个自定义的激活函数）
    OP_MY_CUSTOM_ACTIVATION,  // 新操作码
    
    // ... 其他操作码 ...
};
```

**注意事项**：
- 操作码名称使用`OP_`前缀
- 使用大写字母和下划线
- 保持与现有操作码的命名风格一致

### 1.2 在OpcodeManager中添加字符串映射

**文件位置**：`framework/src/interface/operation/opcode.h`

**操作步骤**：

在`OpcodeManager`类中添加操作码到字符串的映射：

```cpp
class OpcodeManager {
    // ... 现有代码 ...
    
    // 添加新操作码的字符串映射
    static constexpr const char* GetOpcodeStr(Opcode opcode) {
        switch (opcode) {
            // ... 现有case ...
            case Opcode::OP_MY_CUSTOM_ACTIVATION:
                return "OP_MY_CUSTOM_ACTIVATION";
            // ... 其他case ...
        }
    }
};
```

**代码位置**：`framework/src/interface/operation/opcode.h`（查找`GetOpcodeStr`函数`）

---

## 步骤2：实现形状推断函数

### 2.1 创建形状推断函数

**文件位置**：`framework/src/interface/operation/vector/` 或相应目录

**函数签名**：
```cpp
void InferShapeMyCustomActivation(
    Operation* op,
    std::vector<std::vector<SymbolicScalar>>& outValidShapes
);
```

**函数实现**：

```cpp
void InferShapeMyCustomActivation(
    Operation* op,
    std::vector<std::vector<SymbolicScalar>>& outValidShapes
) {
    // 1. 获取输入操作数
    const auto& iOperands = op->GetIOperands();
    if (iOperands.empty()) {
        ALOG_ERROR_F("MyCustomActivation operation has no input operands");
        return;
    }
    
    // 2. 获取输入形状
    const auto& inputShape = iOperands[0]->GetShape();
    
    // 3. 计算输出形状（通常与输入形状相同）
    std::vector<SymbolicScalar> outputShape;
    for (const auto& dim : inputShape) {
        outputShape.push_back(dim);
    }
    
    // 4. 设置输出有效形状
    outValidShapes.push_back(outputShape);
}
```

**关键点**：
- 输入参数：`Operation* op`（操作对象）和`outValidShapes`（输出形状列表）
- 需要处理所有输出操作数的形状
- 支持动态形状（使用`SymbolicScalar`）
- 需要错误处理

### 2.2 注册形状推断函数

**使用注册宏**：

```cpp
// 在实现文件末尾添加
REGISTER_INFER_SHAPE_FUNC(OP_MY_CUSTOM_ACTIVATION, Opcode::OP_MY_CUSTOM_ACTIVATION, InferShapeMyCustomActivation);
```

**宏定义位置**：`framework/src/interface/operation/op_infer_shape_impl.h:74`

**注册机制**：
- 使用静态对象自动注册
- 在程序启动时自动执行
- 无需手动调用

**完整示例**：

```cpp
// framework/src/interface/operation/vector/my_custom_activation.cpp

#include "interface/operation/op_infer_shape_impl.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {

void InferShapeMyCustomActivation(
    Operation* op,
    std::vector<std::vector<SymbolicScalar>>& outValidShapes
) {
    const auto& iOperands = op->GetIOperands();
    if (iOperands.empty()) {
        ALOG_ERROR_F("MyCustomActivation operation has no input operands");
        return;
    }
    
    const auto& inputShape = iOperands[0]->GetShape();
    std::vector<SymbolicScalar> outputShape;
    for (const auto& dim : inputShape) {
        outputShape.push_back(dim);
    }
    
    outValidShapes.push_back(outputShape);
}

// 注册形状推断函数
REGISTER_INFER_SHAPE_FUNC(OP_MY_CUSTOM_ACTIVATION, Opcode::OP_MY_CUSTOM_ACTIVATION, InferShapeMyCustomActivation);

} // namespace npu::tile_fwk
```

---

## 步骤3：实现Tiled函数（可选）

### 3.1 什么是Tiled函数

**Tiled函数**是操作在Tile级别的实现，用于代码生成阶段。如果操作可以通过组合现有操作实现，可以跳过此步骤。

### 3.2 创建Tiled函数

**文件位置**：`framework/src/interface/operation/vector/` 或相应目录

**函数签名**：
```cpp
void MyCustomActivationTileFunc(
    Function &function,
    const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op
);
```

**函数实现**：

```cpp
void MyCustomActivationTileFunc(
    Function &function,
    const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op
) {
    // 1. 获取操作属性（如果有）
    // auto attr = std::static_pointer_cast<MyCustomActivationAttribute>(op.GetOpAttribute());
    
    // 2. 生成Tile级别的操作
    // 这里需要根据具体操作实现Tile级别的代码生成逻辑
    // 通常涉及：
    // - 创建中间Tensor
    // - 添加Tile级别的Operation
    // - 处理内存布局
    
    // 示例：简单的逐元素操作
    // function.AddOperation(Opcode::OP_EXP, {iOperand[0]}, {oOperand[0]});
}
```

**关键点**：
- 在Tile级别实现操作
- 需要考虑内存布局和Tile形状
- 可能需要创建中间Tensor
- 需要处理动态形状

### 3.3 注册Tiled函数

**使用注册宏**：

```cpp
// 在实现文件末尾添加
REGISTER_OPERATION_TILED_FUNC(OP_MY_CUSTOM_ACTIVATION, Opcode::OP_MY_CUSTOM_ACTIVATION, MyCustomActivationTileFunc);
```

**宏定义位置**：`framework/src/interface/operation/operation_common.h:91`

**完整示例**：

```cpp
// framework/src/interface/operation/vector/my_custom_activation.cpp

#include "interface/operation/operation_common.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {

void MyCustomActivationTileFunc(
    Function &function,
    const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op
) {
    // Tile级别的实现
    // ...
}

// 注册Tiled函数
REGISTER_OPERATION_TILED_FUNC(OP_MY_CUSTOM_ACTIVATION, Opcode::OP_MY_CUSTOM_ACTIVATION, MyCustomActivationTileFunc);

} // namespace npu::tile_fwk
```

---

## 步骤4：实现Operator函数（可选）

### 4.1 什么是Operator函数

**Operator函数**是操作的高级API，提供更友好的接口。如果操作可以通过组合现有操作实现，可以跳过此步骤。

### 4.2 创建Operator函数

**文件位置**：`framework/src/operator/` 相应目录

**函数签名**：
```cpp
Tensor MyCustomActivation(const Tensor &operand);
```

**函数实现**：

```cpp
// framework/src/operator/activation/my_custom_activation.cpp

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {

Tensor MyCustomActivation(const Tensor &operand) {
    // 使用FUNCTION宏定义函数
    FUNCTION("MY_CUSTOM_ACTIVATION", {operand}, {result}) {
        // 实现计算逻辑
        // 可以使用现有的操作组合
        // 例如：result = Exp(operand);
    }
    return result;
}

} // namespace npu::tile_fwk
```

**关键点**：
- 使用`FUNCTION`宏定义函数
- 可以使用现有操作组合实现
- 支持动态形状
- 需要处理数据类型转换

### 4.3 在CMakeLists.txt中添加源文件

**文件位置**：`framework/src/operator/CMakeLists.txt`

**操作步骤**：

```cmake
# 添加新源文件
set(OPERATOR_SOURCES
    # ... 现有文件 ...
    operator/activation/my_custom_activation.cpp
    # ... 其他文件 ...
)
```

---

## 步骤5：注册机制

### 5.1 形状推断注册

**已在前面的步骤2.2中完成**，使用`REGISTER_INFER_SHAPE_FUNC`宏。

### 5.2 Tiled函数注册

**已在前面的步骤3.3中完成**，使用`REGISTER_OPERATION_TILED_FUNC`宏。

### 5.3 确保文件被编译

**检查CMakeLists.txt**：

确保实现文件被包含在编译目标中：

```cmake
# framework/src/interface/operation/CMakeLists.txt 或相应文件
set(OPERATION_SOURCES
    # ... 现有文件 ...
    interface/operation/vector/my_custom_activation.cpp
    # ... 其他文件 ...
)
```

---

## 步骤6：Python绑定（可选）

### 6.1 创建Python接口

**文件位置**：`python/pypto/op/` 或相应目录

**Python接口实现**：

```python
# python/pypto/op/activation.py

import pypto
from pypto import pypto_impl

def my_custom_activation(x):
    """自定义激活函数
    
    Args:
        x: 输入Tensor
        
    Returns:
        输出Tensor
    """
    # 调用C++实现
    return pypto_impl.MyCustomActivation(x)
```

### 6.2 C++绑定

**文件位置**：`python/src/bindings/` 相应文件

**绑定代码**：

```cpp
// python/src/bindings/operator.cpp

#include <pybind11/pybind11.h>
#include "operator/activation/my_custom_activation.h"

namespace py = pybind11;

PYBIND11_MODULE(pypto_impl, m) {
    // ... 现有绑定 ...
    
    // 绑定新操作
    m.def("MyCustomActivation", &npu::tile_fwk::MyCustomActivation,
          "My custom activation function");
}
```

---

## 完整示例

### 示例：添加一个简单的激活函数

假设我们要添加一个名为`MyReLU`的激活函数（实际上PyPTO可能已有ReLU，这里仅作示例）。

#### 1. 定义Opcode

**文件**：`framework/src/interface/operation/opcode.h`

```cpp
enum class Opcode {
    // ... 现有操作码 ...
    OP_MY_RELU,  // 新操作码
    // ... 其他操作码 ...
};
```

#### 2. 实现形状推断

**文件**：`framework/src/interface/operation/vector/my_relu.cpp`

```cpp
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {

void InferShapeMyReLU(
    Operation* op,
    std::vector<std::vector<SymbolicScalar>>& outValidShapes
) {
    const auto& iOperands = op->GetIOperands();
    if (iOperands.empty()) {
        ALOG_ERROR_F("MyReLU operation has no input operands");
        return;
    }
    
    const auto& inputShape = iOperands[0]->GetShape();
    std::vector<SymbolicScalar> outputShape;
    for (const auto& dim : inputShape) {
        outputShape.push_back(dim);
    }
    
    outValidShapes.push_back(outputShape);
}

REGISTER_INFER_SHAPE_FUNC(OP_MY_RELU, Opcode::OP_MY_RELU, InferShapeMyReLU);

} // namespace npu::tile_fwk
```

#### 3. 实现Operator函数

**文件**：`framework/src/operator/activation/my_relu.cpp`

```cpp
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {

Tensor MyReLU(const Tensor &operand) {
    Tensor result;
    FUNCTION("MY_RELU", {operand}, {result}) {
        // ReLU = max(x, 0)
        auto zero = Fill(operand.GetShape(), 0.0f, operand.GetStorage()->Datatype());
        result = Maximum(operand, zero);
    }
    return result;
}

} // namespace npu::tile_fwk
```

#### 4. 更新CMakeLists.txt

**文件**：`framework/src/operator/CMakeLists.txt`

```cmake
set(OPERATOR_SOURCES
    # ... 现有文件 ...
    operator/activation/my_relu.cpp
)
```

**文件**：`framework/src/interface/operation/CMakeLists.txt`

```cmake
set(OPERATION_SOURCES
    # ... 现有文件 ...
    interface/operation/vector/my_relu.cpp
)
```

---

## 验证与测试

### 1. 编译验证

**检查编译错误**：
```bash
cd /root/pypto
python build_ci.py --build_type Debug
```

**常见编译错误**：
- 未定义的Opcode：检查opcode.h中是否正确定义
- 未注册的形状推断函数：检查REGISTER_INFER_SHAPE_FUNC宏
- 链接错误：检查CMakeLists.txt是否包含源文件

### 2. 功能测试

**编写测试用例**：

```python
# python/pypto/tests/test_my_relu.py

import pypto
import torch

def test_my_relu():
    # 创建输入
    x_torch = torch.randn(2, 3, 4)
    x_pto = pypto.from_torch(x_torch)
    
    # 调用新操作
    result = pypto.op.activation.my_relu(x_pto)
    
    # 验证结果
    result_torch = pypto.to_torch(result)
    expected = torch.relu(x_torch)
    
    assert torch.allclose(result_torch, expected)

if __name__ == "__main__":
    test_my_relu()
```

### 3. 调试技巧

**启用详细日志**：
```bash
export GLOBAL_LOG_LEVEL=0  # DEBUG级别
```

**检查形状推断**：
- 查看`run.log`中的形状推断日志
- 验证输出形状是否正确

**检查代码生成**：
- 查看生成的CCE代码
- 验证操作是否正确生成

---

## 常见问题

### Q1: 如何确定操作应该放在哪个目录？

**A**: 根据操作类型选择目录：
- **向量操作**：`framework/src/interface/operation/vector/`
- **矩阵操作**：`framework/src/interface/operation/cube/`
- **视图操作**：`framework/src/interface/operation/view/`
- **移动操作**：`framework/src/interface/operation/move/`
- **自定义操作**：创建新目录或放在`operator/`下

### Q2: 什么时候需要实现Tiled函数？

**A**: 以下情况需要实现Tiled函数：
- 操作无法通过组合现有操作实现
- 需要特殊的Tile级别优化
- 需要直接生成CCE代码

如果操作可以通过组合现有操作实现，可以跳过Tiled函数实现。

### Q3: 形状推断函数什么时候被调用？

**A**: 形状推断函数在以下时机被调用：
- Pass优化阶段：验证和推导形状
- 图转换阶段：确定输出形状
- 代码生成阶段：生成形状相关的代码

### Q4: 如何支持动态形状？

**A**: 在形状推断函数中使用`SymbolicScalar`：
```cpp
// 获取动态维度
SymbolicScalar dim = inputShape[0];  // 可能是符号值

// 在输出形状中使用
outputShape.push_back(dim);
```

### Q5: 如何添加操作属性？

**A**: 
1. 定义属性类（继承`OpAttribute`）
2. 在操作创建时设置属性
3. 在Tiled函数中读取属性

**示例**：
```cpp
class MyCustomActivationAttribute : public OpAttribute {
public:
    float alpha = 1.0f;  // 属性值
};

// 设置属性
op.SetOpAttribute(std::make_shared<MyCustomActivationAttribute>());

// 读取属性
auto attr = std::static_pointer_cast<MyCustomActivationAttribute>(op.GetOpAttribute());
float alpha = attr->alpha;
```

---

## 特殊Pass和模块考虑

### 7.1 OpcodeManager配置

**关键点**：新Operation需要在`OpcodeManager`中配置元信息

**文件位置**：`framework/src/interface/operation/opcode.cpp`（OpcodeManager构造函数）

**必须配置的属性**：

| 属性 | 说明 | 影响 |
|------|------|------|
| **coreType** | 核心类型（AIC/AIV/AICPU/HUB等） | 影响图分区、调度优化 |
| **calcType** | 计算类型（ELMWISE/REDUCE/MOVE_IN等） | 影响Pass优化策略 |
| **inputsMemType** | 输入内存类型列表 | 影响内存分配和优化 |
| **outputsMemType** | 输出内存类型列表 | 影响内存分配和优化 |
| **tileOpCfg** | Tile操作配置 | 影响代码生成 |
| **attrs** | 支持的属性列表 | 影响属性验证 |

**配置示例**：

```cpp
// framework/src/interface/operation/opcode.cpp
// 在OpcodeManager构造函数中添加

OpcodeManager::OpcodeManager() {
    // ... 现有配置 ...
    
    // 配置新Operation
    auto &info = opcodeInfos_[static_cast<int>(Opcode::OP_MY_CUSTOM_ACTIVATION)];
    info.opcode = Opcode::OP_MY_CUSTOM_ACTIVATION;
    info.coreType = OpCoreType::AIV;  // 向量操作使用AIV
    info.calcType = OpCalcType::ELMWISE;  // 逐元素操作
    info.str = "OP_MY_CUSTOM_ACTIVATION";
    info.inputsMemType = {MemoryType::MEM_UB};  // 输入在UB
    info.outputsMemType = {MemoryType::MEM_UB};  // 输出在UB
    info.tileOpCfg = TileOpCfg("my_custom_activation", PIPE_S, PIPE_S, CoreType::AIV);
    info.attrs = {};  // 支持的属性列表
}
```

**影响分析**：

1. **coreType影响**：
   - `AIC`：矩阵运算，影响图分区（`GraphPartition`）
   - `AIV`：向量运算，影响调度优化
   - `AICPU`：CPU操作，影响执行路径
   - `HUB`：特殊操作，影响同步

2. **calcType影响**：
   - `MOVE_IN`/`MOVE_OUT`：边界操作，影响`CommonOperationEliminate` Pass
   - `REDUCE`：归约操作，影响内存优化
   - `ELMWISE`：逐元素操作，影响融合优化

### 7.2 Tensor Graph Pass影响

#### InferMemoryConflict Pass

**作用**：推断内存冲突，插入COPY操作

**影响**：
- 如果Operation涉及内存类型转换，需要确保内存类型配置正确
- 如果Operation是边界操作（`MOVE_IN`/`MOVE_OUT`），会被特殊处理

**检查点**：
```cpp
// 检查是否为边界操作
bool isInBoundary = OpcodeManager::Inst().IsBoundaryIn(op->GetOpcode());
bool isOutBoundary = OpcodeManager::Inst().IsBoundaryOut(op->GetOpcode());
```

#### RemoveRedundantReshape Pass

**作用**：移除冗余的RESHAPE操作

**影响**：
- 如果新Operation类似RESHAPE（形状不变），可能需要特殊处理
- 如果Operation会改变形状，需要确保形状推断正确

#### AutoCast Pass

**作用**：自动类型转换

**影响**：
- 如果Operation对数据类型敏感，需要确保类型转换正确

### 7.3 Tile Graph Pass影响

#### GraphPartition Pass

**作用**：图分区，将Tile Graph分割为子图

**影响**：
- **coreType配置**：影响分区策略
  ```cpp
  // 代码位置：framework/src/passes/tile_graph_pass/graph_partition/n_buffer_merge.cpp:240
  if (OpcodeManager::Inst().GetCoreType(opOriList[opIdx].GetOpcode()) == OpCoreType::AIC) {
      // AIC操作的特殊处理
  }
  ```
- **calcType配置**：影响分区边界
  ```cpp
  // 代码位置：framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp:525
  if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_OUT) {
      // MOVE_OUT操作的特殊处理
  }
  ```

#### CommonOperationEliminate Pass

**作用**：消除公共操作

**影响**：
- **内存类型配置**：影响操作消除策略
  ```cpp
  // 代码位置：framework/src/passes/tile_graph_pass/graph_partition/common_operation_eliminate.cpp:48
  auto &inputsMemType = OpcodeManager::Inst().GetInputsMemType(operation->GetOpcode());
  auto &outputsMemType = OpcodeManager::Inst().GetOutputsMemType(operation->GetOpcode());
  OpCalcType opCalcType = OpcodeManager::Inst().GetOpCalcType(operation->GetOpcode());
  ```
- **特殊操作处理**：某些操作（如`OP_VIEW`）不会被消除
  ```cpp
  if (operation->GetOpcode() == Opcode::OP_VIEW) {
      return nullptr;  // VIEW操作不消除
  }
  ```

#### SubgraphToFunction Pass

**作用**：将子图转换为函数

**影响**：
- **COPY操作处理**：如果Operation涉及数据移动，需要确保COPY操作正确处理
  ```cpp
  // 代码位置：framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:607
  ASSERT(copyout->GetOpcode() == Opcode::OP_COPY_OUT) 
      << "Expect Opcode OP_COPY_OUT";
  ```
- **VIEW操作处理**：VIEW操作会被转换为COPY_IN
  ```cpp
  // 代码位置：framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:813
  op.SetOpCode(Opcode::OP_COPY_IN);
  ```

### 7.4 Block Graph Pass影响

#### GlobalMemoryReuse Pass

**作用**：全局内存重用优化

**影响**：
- 如果Operation涉及内存分配，需要确保内存重用正确
- 如果Operation是边界操作，需要特殊处理

#### OoOSchedule Pass

**作用**：乱序调度优化

**影响**：
- **coreType配置**：影响调度策略
- **calcType配置**：影响依赖关系

#### InsertSync Pass

**作用**：插入同步操作

**影响**：
- 如果Operation需要同步，需要确保同步操作正确插入

### 7.5 Codegen模块影响

#### CodeGenOp类

**作用**：代码生成，将Operation转换为CCE代码

**影响**：
- **SKIP_OPCODE列表**：某些操作会被跳过代码生成
  ```cpp
  // 代码位置：framework/src/codegen/codegen_op.h:36
  const std::unordered_set<Opcode> SKIP_OPCODE = {
      Opcode::OP_VIEW,
      Opcode::OP_ASSEMBLE,
      Opcode::OP_RESHAPE,
      // ...
  };
  ```
- **GenOpCode()方法**：需要为每个Operation实现代码生成逻辑
  ```cpp
  // 需要在相应的CodeGenOp子类中实现
  virtual std::string GenOpCode() const = 0;
  ```

#### 代码生成特殊处理

**分布式操作**：
- 如果Operation涉及分布式通信，需要在`codegen_distributed.cpp`中特殊处理
  ```cpp
  // 代码位置：framework/src/codegen/cloudnpu/codegen_distributed.cpp
  case Opcode::OP_SHMEM_PUT:
  case Opcode::OP_SHMEM_GET: {
      // 分布式操作的特殊处理
  }
  ```

### 7.6 Function类影响

#### 边界操作检查

**作用**：Function类会检查操作的边界属性

**影响**：
```cpp
// 代码位置：framework/src/interface/function/function.cpp:2971
bool isInBoundary = OpcodeManager::Inst().IsBoundaryIn(op->GetOpcode());
bool isOutBoundary = OpcodeManager::Inst().IsBoundaryOut(op->GetOpcode());
```

#### 属性验证

**作用**：Function类会验证操作的属性

**影响**：
```cpp
// 代码位置：framework/src/interface/function/function.cpp:2993
for (const auto &attr : OpcodeManager::Inst().GetAttrs(op->GetOpcode())) {
    // 验证属性
}
```

---

## 特殊场景考虑

### 8.1 内存类型敏感的操作

**场景**：Operation对输入/输出内存类型有特定要求

**需要考虑的点**：

1. **配置内存类型**：
   ```cpp
   // 在OpcodeManager中配置
   info.inputsMemType = {MemoryType::MEM_L1};  // 输入必须在L1
   info.outputsMemType = {MemoryType::MEM_UB};  // 输出在UB
   ```

2. **Pass影响**：
   - `InferMemoryConflict` Pass会检查内存冲突
   - `CommonOperationEliminate` Pass会根据内存类型决定是否消除

3. **代码生成**：
   - 需要确保代码生成时正确处理内存类型

### 8.2 边界操作（MOVE_IN/MOVE_OUT）

**场景**：Operation是数据移动操作（如COPY_IN、COPY_OUT）

**需要考虑的点**：

1. **calcType配置**：
   ```cpp
   info.calcType = OpCalcType::MOVE_IN;  // 或 MOVE_OUT
   ```

2. **Pass影响**：
   - `CommonOperationEliminate` Pass会特殊处理
   - `SubgraphToFunction` Pass会检查COPY操作

3. **Function类影响**：
   - `IsBoundaryIn()`/`IsBoundaryOut()`会返回true
   - 影响函数边界检查

### 8.3 视图操作（VIEW/ASSEMBLE）

**场景**：Operation是视图操作，不实际移动数据

**需要考虑的点**：

1. **SKIP_OPCODE**：
   - 如果不需要生成代码，需要添加到`SKIP_OPCODE`列表

2. **Pass影响**：
   - `CommonOperationEliminate` Pass不会消除VIEW操作
   - `SubgraphToFunction` Pass会将VIEW转换为COPY_IN

3. **内存优化**：
   - 视图操作不涉及实际内存分配，需要特殊处理

### 8.4 分布式操作

**场景**：Operation涉及多设备通信

**需要考虑的点**：

1. **calcType配置**：
   ```cpp
   info.calcType = OpCalcType::DISTRIBUTED;
   ```

2. **代码生成**：
   - 需要在`codegen_distributed.cpp`中实现特殊处理
   - 需要处理同步和通信

3. **Pass影响**：
   - `InsertSync` Pass会插入同步操作
   - `GraphPartition` Pass会考虑分布式边界

### 8.5 归约操作（REDUCE）

**场景**：Operation是归约操作（如SUM、MAX）

**需要考虑的点**：

1. **calcType配置**：
   ```cpp
   info.calcType = OpCalcType::REDUCE;
   ```

2. **内存优化**：
   - 归约操作可能需要临时内存
   - `GlobalMemoryReuse` Pass需要特殊处理

3. **代码生成**：
   - 归约操作需要特殊的CCE代码生成逻辑

### 8.6 矩阵运算（MATMUL）

**场景**：Operation是矩阵运算

**需要考虑的点**：

1. **coreType配置**：
   ```cpp
   info.coreType = OpCoreType::AIC;  // 矩阵运算使用AIC
   ```

2. **图分区**：
   - AIC操作会影响图分区策略
   - `GraphPartition` Pass会特殊处理

3. **调度优化**：
   - 矩阵运算需要特殊的调度策略
   - `OoOSchedule` Pass会考虑AIC操作

### 8.7 动态形状支持

**场景**：Operation需要支持动态形状

**需要考虑的点**：

1. **形状推断**：
   - 形状推断函数必须使用`SymbolicScalar`
   - 需要正确处理动态维度

2. **代码生成**：
   - 需要支持动态形状的代码生成
   - `CodeGenOp`需要处理动态形状

3. **Pass影响**：
   - 某些Pass可能不支持动态形状
   - 需要测试动态形状场景

### 8.8 属性支持

**场景**：Operation需要支持特定属性

**需要考虑的点**：

1. **属性注册**：
   ```cpp
   // 在OpcodeManager中注册支持的属性
   info.attrs = {"alpha", "beta", "axis"};
   ```

2. **属性验证**：
   - Function类会验证属性
   - 需要确保属性类型正确

3. **代码生成**：
   - 属性需要转换为代码生成参数
   - `CodeGenOp::GenOpAttr()`需要处理属性

---

## 总结

添加新Operation的完整流程：

1. **定义Opcode**：在`opcode.h`中添加新操作码
2. **实现形状推断**：创建形状推断函数并注册
3. **实现Tiled函数**（可选）：创建Tile级别实现并注册
4. **实现Operator函数**（可选）：创建高级API
5. **配置OpcodeManager**：**必须**在`opcode.cpp`中配置元信息（coreType、calcType、内存类型等）
6. **考虑Pass影响**：检查相关Pass是否需要特殊处理
7. **实现代码生成**：实现CodeGenOp或确保在SKIP_OPCODE列表中
8. **Python绑定**（可选）：提供Python接口
9. **验证测试**：编写测试用例验证功能

**关键要点**：
- **必须实现形状推断函数**：所有Operation都需要形状推断
- **必须在OpcodeManager中配置元信息**：coreType、calcType、内存类型等影响整个编译流程
- **Tiled函数是可选的**：如果可以通过组合现有操作实现，可以跳过
- **考虑Pass影响**：
  - Tensor Graph Pass：`InferMemoryConflict`、`RemoveRedundantReshape`等
  - Tile Graph Pass：`GraphPartition`、`CommonOperationEliminate`、`SubgraphToFunction`等
  - Block Graph Pass：`GlobalMemoryReuse`、`OoOSchedule`、`InsertSync`等
- **考虑代码生成**：需要实现CodeGenOp或确保在SKIP_OPCODE列表中
- **特殊场景处理**：内存类型敏感、边界操作、视图操作、分布式操作等
- **确保所有文件被正确编译**：检查CMakeLists.txt
- **编写测试用例验证功能**：包括正常场景和边界场景

**相关文档**：
- [Operation类技术文档](../02-core/06-operation.md)
- [核心概念](../02-core/01-concepts.md)
- [Codegen模块](../02-core/10-codegen.md)

