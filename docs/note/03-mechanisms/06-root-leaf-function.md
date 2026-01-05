# PyPTO root_function 和 leaf_function 机制详解

> **适用对象：** 想要深入理解PyPTO框架函数层级关系的开发者  
> **学习时间：** 50-70分钟  
> **前置知识：** 已阅读[Function类技术文档](../02-core/05-function.md)和[核心概念](../02-core/01-concepts.md)  
> **学习目标：** 理解root_function和leaf_function的概念、关系、作用以及在编译和执行中的角色

## 概述

PyPTO框架采用函数层级结构来组织和管理计算图。在这个层级结构中，`root_function`（根函数）和`leaf_function`（叶子函数）是两个核心概念，它们共同构成了PyPTO的函数调用和执行体系。

**核心概念**：
- **root_function**：函数层级结构中的顶层函数，管理整个函数调用树
- **leaf_function**：函数层级结构中的叶子节点函数，不包含子函数，是实际执行的计算单元

**相关文档：**
- [关键机制列表](01-key-mechanisms-list.md) - 所有机制总览
- [Function类技术文档](../02-core/05-function.md) - Function类详细说明
- [Passes模块](../02-core/09-passes.md) - 图优化Pass，包含SubgraphToFunction

---

## 目录

1. [基本概念](#基本概念)
2. [函数层级结构](#函数层级结构)
3. [root_function机制详解](#root_function机制详解)
4. [leaf_function机制详解](#leaf_function机制详解)
5. [关系与交互](#关系与交互)
6. [在编译流程中的角色](#在编译流程中的角色)
7. [在运行时中的角色](#在运行时中的角色)
8. [实际应用场景](#实际应用场景)

---

## 基本概念

### root_function（根函数）

**定义**：函数层级结构中的顶层函数，是函数调用树的根节点。

**核心特征**：
- 没有父函数（`parent_ == nullptr`）
- 管理整个函数调用树的结构
- 持有拓扑信息（`topoInfo_`）
- 管理所有叶子函数（`programs_`）
- 通常是`EXECUTE_GRAPH`类型的函数

**代码位置**：
- 头文件：`framework/src/interface/function/function.h:550`
- 实现文件：`framework/src/interface/function/function.cpp`

**关键API**：
```cpp
Function *GetRootFunction() const { return rootFunc_; }
```

### leaf_function（叶子函数）

**定义**：函数层级结构中的叶子节点函数，不包含子函数，是实际执行的计算单元。

**核心特征**：
- 不包含`OP_CALL`操作（或只包含对已编译函数的调用）
- 是实际的计算单元，包含具体的操作序列
- 通常是`BLOCK_GRAPH`类型的函数
- 可以被多个调用点复用（通过函数缓存）

**代码位置**：
- 创建位置：`framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:432`
- 属性管理：`framework/src/interface/function/function.h`

**关键特征**：
- 通过`SetProgramOp()`设置操作序列
- 通过`SetLeafFuncAttribute()`设置叶子函数属性
- 包含核心类型信息（`CoreType`）

---

## 函数层级结构

### 层级关系图

```
┌─────────────────────────────────────────────────────────┐
│              函数层级结构示意图                            │
└─────────────────────────────────────────────────────────┘

                    root_function
                    (EXECUTE_GRAPH)
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    leaf_func_0      leaf_func_1      leaf_func_2
    (BLOCK_GRAPH)   (BLOCK_GRAPH)   (BLOCK_GRAPH)
        │                │                │
    [Operations]    [Operations]    [Operations]
```

### 数据结构关系

**root_function的数据结构**：
```cpp
class Function {
    Function *rootFunc_ = nullptr;  // 指向根函数（如果是叶子函数）
    Function *parent_ = nullptr;     // 指向父函数
    
    // root_function特有的成员
    SubfuncTopologyInfoTy topoInfo_;  // 拓扑信息（root function持有）
    std::map<uint64_t, Function*> programs_;  // 子图ID到leaf function的映射
};
```

**leaf_function的数据结构**：
```cpp
class Function {
    Function *rootFunc_ = nullptr;  // 指向根函数
    Function *parent_ = nullptr;     // 指向父函数（通常是root_function）
    
    // leaf_function特有的成员
    std::vector<OperationPtr> programOp_;  // 操作序列
    std::shared_ptr<LeafFuncAttribute> leafFuncAttr_;  // 叶子函数属性
};
```

### 层级关系表

| 函数类型 | parent_ | rootFunc_ | 说明 |
|---------|---------|-----------|------|
| **root_function** | `nullptr` | `nullptr` | 顶层函数，没有父函数 |
| **leaf_function** | `root_function*` | `root_function*` | 叶子函数，父函数是root_function |
| **中间函数** | `非nullptr` | `root_function*` | 中间层函数（较少见） |

---

## root_function机制详解

### 定义与特征

**root_function**是函数层级结构中的顶层函数，负责管理整个函数调用树的结构和执行顺序。

**核心职责**：
1. **管理函数调用树**：维护所有子函数的调用关系
2. **持有拓扑信息**：存储子图之间的依赖关系和执行顺序
3. **管理叶子函数映射**：通过`programs_`映射子图ID到叶子函数
4. **协调执行**：在运行时协调各个叶子函数的执行

### 创建时机

**root_function在以下时机创建**：

1. **SubgraphToFunction Pass阶段**
   - 当Tile Graph转换为Execute Graph时
   - 代码位置：`framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:515-520`

```cpp
// 创建root function
auto rootName = Function::CreateRootRawName(function.GetRawName());
Program::GetInstance().BeginFunction(rootName, function.GetFunctionType(), GraphType::EXECUTE_GRAPH);
auto rootFunc = Program::GetInstance().GetCurrentFunction();
InitializeRootFunction(function, *rootFunc);
```

2. **函数类型**
   - 通常是`EXECUTE_GRAPH`类型
   - 继承原函数的`FunctionType`（STATIC或DYNAMIC）

### 关键数据结构

#### 1. topoInfo_（拓扑信息）

**定义**：存储子图之间的依赖关系和执行顺序

**作用**：
- 记录子图之间的依赖关系
- 确定执行顺序
- 支持并行执行优化

**代码位置**：`framework/src/interface/function/function.h:480`

```cpp
SubfuncTopologyInfoTy topoInfo_;  // root function持有，对应1.0的SubgraphTopologyInfoTy
```

#### 2. programs_（叶子函数映射）

**定义**：子图ID到叶子函数的映射表

**作用**：
- 管理所有异构的叶子函数
- 支持子图到函数的查找
- 支持函数复用和缓存

**代码位置**：`framework/src/interface/function/function.h:481`

```cpp
std::map<uint64_t, Function*> programs_;  // root function持有，所有异构的leaf function
```

**映射关系**：
- Key：子图ID（`subgraphID`）
- Value：对应的叶子函数指针

### 关键方法

#### GetRootFunction()

**定义**：获取根函数指针

**代码位置**：`framework/src/interface/function/function.h:550`

```cpp
Function *GetRootFunction() const { return rootFunc_; }
```

**使用场景**：
- 叶子函数查找根函数
- 编译时确定函数层级
- 运行时访问拓扑信息

**示例**：
```cpp
// 在叶子函数中访问根函数
Function *root = leafFunc->GetRootFunction();
if (root != nullptr) {
    // 访问根函数的拓扑信息
    auto &topoInfo = root->topoInfo_;
}
```

### 在编译流程中的角色

#### 1. 图转换阶段

**SubgraphToFunction Pass**：
- 将Tile Graph分割为多个子图
- 为每个子图创建对应的leaf_function
- 创建root_function管理所有leaf_function

**流程**：
```
Tile Graph (BLOCK_GRAPH)
    ↓
SubgraphToFunction Pass
    ↓
┌─────────────────────┐
│  root_function      │  (EXECUTE_GRAPH)
│  (管理调用关系)      │
└─────────────────────┘
    ├─→ leaf_func_0
    ├─→ leaf_func_1
    └─→ leaf_func_2
```

#### 2. 代码生成阶段

**root_function的作用**：
- 生成执行调度代码
- 管理函数调用序列
- 协调各个leaf_function的执行

**代码位置**：`framework/src/machine/host/backend.cpp:208`

```cpp
Function *root = func->GetRootFunction();
if (root != nullptr) {
    // 使用root function进行代码生成
}
```

### 在运行时中的角色

#### 1. 执行调度

**root_function负责**：
- 确定执行顺序
- 管理依赖关系
- 协调并行执行

**执行流程**：
```
运行时执行
    ↓
root_function::Run()
    ↓
遍历topoInfo_确定执行顺序
    ↓
依次调用leaf_function
    ↓
等待所有leaf_function完成
```

#### 2. 函数查找

**通过programs_映射查找**：
- 根据子图ID查找对应的leaf_function
- 支持函数复用
- 优化执行效率

---

## leaf_function机制详解

### 定义与特征

**leaf_function**是函数层级结构中的叶子节点函数，是实际执行的计算单元，包含具体的操作序列。

**核心特征**：
1. **不包含子函数**：不包含`OP_CALL`操作（或只包含对已编译函数的调用）
2. **包含操作序列**：通过`programOp_`存储具体的操作
3. **可独立编译**：每个leaf_function可以独立编译和优化
4. **可复用**：相同的leaf_function可以被多个调用点复用

### 创建时机

**leaf_function在以下时机创建**：

1. **SubgraphToFunction Pass阶段**
   - 当Tile Graph分割为子图时
   - 为每个子图创建一个leaf_function
   - 代码位置：`framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:426-451`

```cpp
Status SubgraphToFunction::ProcessSubgraph(
    Function &function, size_t i, size_t &programIdx, std::vector<Function *> &outputFuncList) {
    auto subgraph = nLIST[i];
    auto leafName = function.GetRawName() + "_leaf" + std::to_string(i);
    
    // 创建leaf function
    Program::GetInstance().BeginFunction(leafName, FunctionType::STATIC, GraphType::BLOCK_GRAPH);
    auto leafFunc = Program::GetInstance().GetCurrentFunction();
    
    // 设置操作序列
    leafFunc->SetProgramOp(subgraph);
    
    // 设置叶子函数属性
    leafFunc->SetLeafFuncAttribute(std::make_shared<LeafFuncAttribute>());
    
    // 插入参数
    InsertParameter(i, *leafFunc);
    
    // 结束函数，生成CallOp
    auto result = Program::GetInstance().EndFunction(leafName);
    // ...
}
```

2. **函数类型**
   - 通常是`BLOCK_GRAPH`类型
   - 通常是`STATIC`类型（静态形状）

### 关键数据结构

#### 1. programOp_（操作序列）

**定义**：存储leaf_function的操作序列

**作用**：
- 存储实际的计算操作
- 支持独立编译
- 支持操作序列管理

**代码位置**：`framework/src/interface/function/function.h:503`

```cpp
std::vector<OperationPtr> &GetProgramOp();
void SetProgramOp(const std::vector<OperationPtr> &operations);
```

**设置时机**：
- 在`ProcessSubgraph()`中设置
- 从子图的操作列表中提取

#### 2. leafFuncAttr_（叶子函数属性）

**定义**：叶子函数的特殊属性

**作用**：
- 存储核心类型信息（`CoreType`）
- 存储其他叶子函数特有的属性
- 支持运行时优化

**代码位置**：`framework/src/interface/function/function.h`

```cpp
std::shared_ptr<LeafFuncAttribute> leafFuncAttr_;
```

**属性内容**：
- `coreType`：核心类型（AIC、AIV、AICPU等）
- 其他运行时属性

### 关键方法

#### SetProgramOp()

**定义**：设置leaf_function的操作序列

**代码位置**：`framework/src/interface/function/function.h:504`

```cpp
void SetProgramOp(const std::vector<OperationPtr> &operations);
```

**使用场景**：
- 创建leaf_function时设置操作序列
- 从子图提取操作时设置

**示例**：
```cpp
// 从子图提取操作序列
auto subgraph = nLIST[i];
leafFunc->SetProgramOp(subgraph);
```

#### SetLeafFuncAttribute()

**定义**：设置叶子函数属性

**代码位置**：`framework/src/interface/function/function.h`

```cpp
void SetLeafFuncAttribute(std::shared_ptr<LeafFuncAttribute> attr);
```

**使用场景**：
- 创建leaf_function时设置属性
- 设置核心类型等信息

### 在编译流程中的角色

#### 1. 图分割阶段

**SubgraphToFunction Pass**：
- 将Tile Graph分割为多个子图
- 为每个子图创建leaf_function
- 设置操作序列和属性

**流程**：
```
Tile Graph
    ↓
图分割（GraphPartition）
    ↓
子图列表（nLIST）
    ↓
为每个子图创建leaf_function
    ↓
设置操作序列和属性
```

#### 2. 代码生成阶段

**leaf_function的作用**：
- 生成具体的计算代码
- 优化操作序列
- 生成可执行的kernel

**代码生成**：
- 每个leaf_function独立生成代码
- 支持并行编译
- 支持代码复用

### 在运行时中的角色

#### 1. 实际执行

**leaf_function负责**：
- 执行具体的计算操作
- 处理输入输出数据
- 管理局部内存

**执行流程**：
```
运行时调用
    ↓
leaf_function::Execute()
    ↓
执行操作序列（programOp_）
    ↓
处理输入输出
    ↓
返回结果
```

#### 2. 函数复用

**通过函数缓存复用**：
- 相同的leaf_function可以被多个调用点复用
- 通过函数哈希匹配
- 减少重复编译

---

## 关系与交互

### root_function与leaf_function的关系

#### 1. 层级关系

**关系图**：
```
root_function (EXECUTE_GRAPH)
    │
    ├─→ leaf_func_0 (BLOCK_GRAPH)
    │   └─→ [Operations]
    │
    ├─→ leaf_func_1 (BLOCK_GRAPH)
    │   └─→ [Operations]
    │
    └─→ leaf_func_2 (BLOCK_GRAPH)
        └─→ [Operations]
```

**关系说明**：
- root_function是父节点
- leaf_function是子节点
- 一个root_function可以管理多个leaf_function

#### 2. 数据关系

**root_function管理leaf_function**：
```cpp
// root_function中
std::map<uint64_t, Function*> programs_;  // 子图ID → leaf_function

// leaf_function中
Function *rootFunc_ = nullptr;  // 指向root_function
Function *parent_ = nullptr;     // 指向root_function（通常是）
```

**关系建立**：
- 在`SubgraphToFunction` Pass中建立
- root_function的`programs_`存储leaf_function
- leaf_function的`rootFunc_`指向root_function

#### 3. 调用关系

**调用流程**：
```
运行时
    ↓
root_function::Run()
    ↓
根据topoInfo_确定执行顺序
    ↓
调用leaf_function_0::Execute()
    ↓
调用leaf_function_1::Execute()
    ↓
调用leaf_function_2::Execute()
    ↓
等待所有leaf_function完成
```

### 交互机制

#### 1. 编译时交互

**SubgraphToFunction Pass**：
1. 创建root_function
2. 为每个子图创建leaf_function
3. 建立root_function和leaf_function的关系
4. 设置leaf_function的`rootFunc_`指针

**代码位置**：`framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:536`

```cpp
// 建立关系
function.rootFunc_ = rootFunc;

// root_function管理leaf_function
rootFunc->programs_[subgraphID] = leafFunc;
```

#### 2. 运行时交互

**执行交互**：
- root_function根据拓扑信息确定执行顺序
- root_function调用leaf_function执行
- leaf_function通过`rootFunc_`访问根函数信息

**代码位置**：`framework/src/machine/host/backend.cpp:208`

```cpp
Function *root = func->GetRootFunction();
if (root != nullptr) {
    // 使用root function进行执行调度
}
```

---

## 在编译流程中的角色

### 编译流程概览

```
用户代码
    ↓
前端解析 → Function (TENSOR_GRAPH)
    ↓
Tensor Graph Pass
    ↓
Tile Graph Pass → Function (TILE_GRAPH)
    ↓
Block Graph Pass → Function (BLOCK_GRAPH)
    ↓
SubgraphToFunction Pass
    ↓
┌─────────────────────────┐
│  root_function          │  (EXECUTE_GRAPH)
│  └─→ leaf_func_0        │  (BLOCK_GRAPH)
│  └─→ leaf_func_1        │  (BLOCK_GRAPH)
│  └─→ leaf_func_2        │  (BLOCK_GRAPH)
└─────────────────────────┘
    ↓
代码生成
    ↓
可执行代码
```

### SubgraphToFunction Pass详解

**Pass作用**：将Block Graph转换为Execute Graph，创建root_function和leaf_function

**关键步骤**：

#### 1. 图分割

**目的**：将Block Graph分割为多个子图

**方法**：
- 使用图分割算法（如`GraphPartition`）
- 每个子图对应一个leaf_function
- 子图之间可能有依赖关系

**结果**：
- `nLIST`：子图列表，每个子图是一个操作序列

#### 2. 创建root_function

**代码位置**：`framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:515-520`

```cpp
// 创建root function
auto rootName = Function::CreateRootRawName(function.GetRawName());
Program::GetInstance().BeginFunction(rootName, function.GetFunctionType(), GraphType::EXECUTE_GRAPH);
auto rootFunc = Program::GetInstance().GetCurrentFunction();
InitializeRootFunction(function, *rootFunc);
```

**关键操作**：
- 创建`EXECUTE_GRAPH`类型的函数
- 初始化拓扑信息
- 设置函数类型

#### 3. 创建leaf_function

**代码位置**：`framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:426-451`

```cpp
Status SubgraphToFunction::ProcessSubgraph(
    Function &function, size_t i, size_t &programIdx, std::vector<Function *> &outputFuncList) {
    auto subgraph = nLIST[i];
    auto leafName = function.GetRawName() + "_leaf" + std::to_string(i);
    
    // 创建leaf function
    Program::GetInstance().BeginFunction(leafName, FunctionType::STATIC, GraphType::BLOCK_GRAPH);
    auto leafFunc = Program::GetInstance().GetCurrentFunction();
    
    // 设置操作序列
    leafFunc->SetProgramOp(subgraph);
    
    // 设置叶子函数属性
    leafFunc->SetLeafFuncAttribute(std::make_shared<LeafFuncAttribute>());
    
    // 插入参数
    InsertParameter(i, *leafFunc);
    
    // 结束函数，生成CallOp
    auto result = Program::GetInstance().EndFunction(leafName);
    // ...
}
```

**关键操作**：
- 创建`BLOCK_GRAPH`类型的函数
- 设置操作序列（`SetProgramOp`）
- 设置叶子函数属性
- 插入输入输出参数
- 生成`OP_CALL`操作（在root_function中）

#### 4. 建立关系

**代码位置**：`framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:536`

```cpp
// 建立root_function和原函数的关系
function.rootFunc_ = rootFunc;

// root_function管理leaf_function
// 在ProcessSubgraph中通过programs_映射管理
```

**关系建立**：
- 原函数的`rootFunc_`指向新创建的root_function
- root_function的`programs_`存储leaf_function映射
- leaf_function的`rootFunc_`指向root_function

### 代码生成阶段

#### root_function的代码生成

**作用**：
- 生成执行调度代码
- 管理函数调用序列
- 协调各个leaf_function的执行

**代码位置**：`framework/src/machine/host/backend.cpp:208`

```cpp
Function *root = func->GetRootFunction();
if (root != nullptr) {
    // 使用root function进行代码生成
    // 生成执行调度逻辑
}
```

#### leaf_function的代码生成

**作用**：
- 生成具体的计算代码
- 优化操作序列
- 生成可执行的kernel

**特点**：
- 每个leaf_function独立生成代码
- 支持并行编译
- 支持代码复用

---

## 在运行时中的角色

### 执行流程

#### 1. root_function的执行

**执行流程**：
```
运行时启动
    ↓
root_function::Run()
    ↓
读取topoInfo_（拓扑信息）
    ↓
确定执行顺序
    ↓
依次调用leaf_function
    ↓
等待所有leaf_function完成
    ↓
返回结果
```

**关键操作**：
- 读取拓扑信息确定执行顺序
- 管理依赖关系
- 协调并行执行
- 等待所有leaf_function完成

#### 2. leaf_function的执行

**执行流程**：
```
被root_function调用
    ↓
leaf_function::Execute()
    ↓
读取输入数据
    ↓
执行操作序列（programOp_）
    ↓
处理输出数据
    ↓
返回结果
```

**关键操作**：
- 执行具体的计算操作
- 处理输入输出数据
- 管理局部内存
- 返回计算结果

### 函数查找与复用

#### 1. 函数查找

**通过programs_映射查找**：
```cpp
// 在root_function中
auto leafFunc = rootFunc->programs_[subgraphID];
if (leafFunc != nullptr) {
    // 找到对应的leaf_function
}
```

**查找场景**：
- 根据子图ID查找leaf_function
- 运行时确定调用哪个leaf_function
- 支持函数复用

#### 2. 函数复用

**通过函数缓存复用**：
- 相同的leaf_function可以被多个调用点复用
- 通过函数哈希匹配
- 减少重复编译和执行

**代码位置**：`framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:463`

```cpp
auto cacheValue = Program::GetInstance().TryHitCahce(callAttr->GetCalleeHash());
if (cacheValue) {
    // 复用已编译的leaf_function
    callAttr->SetCalleeMagicName(cacheValue->cacheFunction->GetMagicName());
}
```

---

## 实际应用场景

### 场景1：图分割优化

**场景描述**：将大型计算图分割为多个子图，每个子图对应一个leaf_function

**优势**：
- 支持并行执行
- 优化内存使用
- 提高执行效率

**实现**：
- 使用`GraphPartition` Pass分割图
- 使用`SubgraphToFunction` Pass创建leaf_function
- root_function管理执行顺序

### 场景2：函数复用

**场景描述**：相同的子图复用同一个leaf_function

**优势**：
- 减少重复编译
- 节省内存空间
- 提高执行效率

**实现**：
- 通过函数哈希匹配
- 使用函数缓存机制
- root_function管理复用关系

### 场景3：动态执行

**场景描述**：根据运行时条件动态选择执行哪个leaf_function

**优势**：
- 支持动态形状
- 支持条件分支
- 提高灵活性

**实现**：
- root_function根据条件选择leaf_function
- 通过`programs_`映射查找
- 动态调用leaf_function

---

## 代码示例

### 示例1：创建root_function和leaf_function

**代码位置**：`framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:514-536`

```cpp
Status SubgraphToFunction::IslandToFunction(Function &function) {
    // 1. 创建root function
    auto rootName = Function::CreateRootRawName(function.GetRawName());
    Program::GetInstance().BeginFunction(rootName, function.GetFunctionType(), GraphType::EXECUTE_GRAPH);
    auto rootFunc = Program::GetInstance().GetCurrentFunction();
    InitializeRootFunction(function, *rootFunc);

    // 2. 为每个子图创建leaf function
    size_t programIdx = 0;
    for (size_t i = 0; i < nLIST.size(); i++) {
        Status status = ProcessSubgraph(function, i, programIdx, mergedFuncList);
        if (status != SUCCESS) {
            return status;
        }
    }

    // 3. 建立关系
    function.rootFunc_ = rootFunc;
    
    return SUCCESS;
}
```

### 示例2：访问root_function

**代码位置**：`framework/src/machine/host/backend.cpp:208`

```cpp
Function *root = func->GetRootFunction();
if (root != nullptr) {
    // 使用root function进行代码生成或执行
    // 访问拓扑信息
    auto &topoInfo = root->topoInfo_;
    // 访问leaf function映射
    auto &programs = root->programs_;
}
```

### 示例3：查找leaf_function

**代码位置**：`framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:728`

```cpp
auto root = function.GetRootFunction();
if (root != nullptr) {
    // 通过子图ID查找leaf function
    for (const auto &[psgId, leaf] : root->programs_) {
        // 处理leaf function
        auto iodescDict = leaf->GetTensorDataForLeafGraph();
        // ...
    }
}
```

---

## 关键设计要点

### 1. 层级关系管理

**设计要点**：
- root_function管理整个函数调用树
- leaf_function通过`rootFunc_`访问根函数
- 支持多层级函数嵌套（较少见）

**优势**：
- 清晰的层级关系
- 便于管理和查找
- 支持复杂函数结构

### 2. 函数复用机制

**设计要点**：
- 通过函数哈希匹配
- 使用函数缓存机制
- root_function管理复用关系

**优势**：
- 减少重复编译
- 节省内存空间
- 提高执行效率

### 3. 执行调度机制

**设计要点**：
- root_function负责执行调度
- 根据拓扑信息确定执行顺序
- 支持并行执行优化

**优势**：
- 灵活的调度策略
- 支持并行执行
- 优化执行效率

---

## 总结

### 核心要点

1. **root_function**：
   - 函数层级结构中的顶层函数
   - 管理整个函数调用树
   - 持有拓扑信息和leaf_function映射
   - 负责执行调度

2. **leaf_function**：
   - 函数层级结构中的叶子节点函数
   - 包含具体的操作序列
   - 是实际执行的计算单元
   - 可以被复用

3. **关系**：
   - root_function管理leaf_function
   - leaf_function通过`rootFunc_`访问根函数
   - 一个root_function可以管理多个leaf_function

4. **作用**：
   - 支持图分割优化
   - 支持函数复用
   - 支持并行执行
   - 优化内存使用

### 实际应用

在实际使用中：
- **开发调试**：通过root_function和leaf_function理解函数结构
- **性能优化**：通过函数复用和并行执行优化性能
- **问题排查**：通过函数层级关系定位问题
- **架构理解**：理解PyPTO的函数组织方式

### 最佳实践

1. **理解层级关系**：清楚root_function和leaf_function的关系
2. **利用函数复用**：通过函数缓存机制减少重复编译
3. **优化执行顺序**：通过拓扑信息优化执行顺序
4. **调试技巧**：通过`GetRootFunction()`访问根函数信息

---

## 相关文档索引

- [关键机制列表](01-key-mechanisms-list.md) - 所有机制总览
- [Function类技术文档](../02-core/05-function.md) - Function类详细说明
- [Passes模块](../02-core/09-passes.md) - 图优化Pass，包含SubgraphToFunction
- [核心概念](../02-core/01-concepts.md) - 基础概念和术语

---

## 附录

### 关键代码位置

| 功能 | 文件位置 | 说明 |
|------|---------|------|
| **GetRootFunction()** | `framework/src/interface/function/function.h:550` | 获取根函数 |
| **创建root_function** | `framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:515` | 创建根函数 |
| **创建leaf_function** | `framework/src/passes/tile_graph_pass/subgraph_to_function.cpp:432` | 创建叶子函数 |
| **设置操作序列** | `framework/src/interface/function/function.h:504` | SetProgramOp |
| **访问root_function** | `framework/src/machine/host/backend.cpp:208` | 运行时访问 |

### 数据结构说明

| 成员变量 | 类型 | 说明 |
|---------|------|------|
| **rootFunc_** | `Function*` | 指向根函数（叶子函数持有） |
| **parent_** | `Function*` | 指向父函数 |
| **topoInfo_** | `SubfuncTopologyInfoTy` | 拓扑信息（root function持有） |
| **programs_** | `std::map<uint64_t, Function*>` | 子图ID到leaf function的映射（root function持有） |
| **programOp_** | `std::vector<OperationPtr>` | 操作序列（leaf function持有） |
| **leafFuncAttr_** | `std::shared_ptr<LeafFuncAttribute>` | 叶子函数属性（leaf function持有） |

### Python API

**Python侧访问**：
```python
# 获取root function
root_func = func.root_function

# 检查是否有parent
has_parent = func.has_parent

# 访问函数属性
incast = func.incast
outcast = func.outcast
```

**代码位置**：`python/pypto/functions.py:163-169`

