# PyPTO 核心概念详解

> **适用对象：** 想要深入理解PyPTO核心概念的开发者  
> **学习时间：** 30-45分钟  
> **前置知识：** 已完成[上手指南](../00-getting-started/00-quick-start.md)  
> **学习目标：** 掌握Tensor、Tile、IR、Pass等核心概念，理解PTO编程范式

## 概述

本文档详细解释 PyPTO 框架中的核心概念和术语，帮助开发者深入理解框架的设计理念和实现原理。

**为什么需要阅读本文档：**
- **理解PyPTO的核心设计理念**（PTO编程范式）
- **掌握多层级IR系统的转换过程**
- **理解Tile如何影响性能**
- **为深入学习各模块打下基础**

**相关文档：**
- [框架总览](00-overview.md) - PyPTO 框架技术文档总览
- [Function 类](03-function.md) - Function 类技术文档
- [Operation 类](04-operation.md) - Operation 类技术文档
- [Passes 模块](07-passes.md) - Passes 模块技术文档

---

## 目录

- [框架基础概念](#框架基础概念)
- [多层级 IR 系统](#多层级-ir-系统)
- [编译流程概念](#编译流程概念)
- [执行调度概念](#执行调度概念)
- [内存管理概念](#内存管理概念)
- [控制流概念](#控制流概念)
- [数据类型和格式](#数据类型和格式)
- [优化和 Pass](#优化和-pass)

---

## 框架基础概念

### PyPTO

**定义**：PyPTO（发音：pai p-t-o）是一款面向 AI 加速器的高性能编程框架，旨在简化复杂融合算子乃至整个模型网络的开发流程，同时保持高性能计算能力。

**核心特性**：
- 基于 Tile 的编程模型
- 多层级 IR 系统
- 自动化代码生成
- MPMD 执行调度

**相关文档**：[框架总览](00-overview.md)

### PTO（Parallel Tensor/Tile Operation）

**定义**：PTO 是一种基于 Tensor 的编程范式，核心思想是将 Tensor 作为数据的基本表达方式，通过一系列对 Tensor 的基本运算来描述并组装完整的计算流程（或计算图）。

**设计理念**：
- **Tensor 级别抽象**：以 Tensor 而非单个元素描述计算，贴近算法设计者的数学表达式
- **声明式编程**：开发者只需描述"做什么"，框架自动处理"怎么做"
- **基于 Tile 的计算**：所有计算最终都基于 Tile（硬件感知的数据块）进行，充分利用硬件并行计算能力
- **计算图驱动**：通过构建计算图，框架可以自动进行优化、调度和执行

**相关文档**：[框架总览](00-overview.md#pypto-架构总览)

### Operation（操作）

**定义**：PyPTO 中描述计算图中基本运算的单元。每个 Operation 定义了一个具体的计算逻辑，能够处理输入 Tensor 并生成输出 Tensor。

**特点**：
- 每个 Operation 有唯一的操作码（Opcode）
- 包含输入输出操作数（Operands）
- 可以包含属性（Attributes）用于配置行为
- 支持形状推断和类型检查

**相关文档**：[Operation 类](04-operation.md)

### Function（函数）

**定义**：Function 是 PyPTO 中函数级 IR 的核心抽象，管理操作序列、张量映射等。

**职责**：
- 管理操作序列（`Operation` 列表）
- 管理张量映射（`TensorMap`）
- 支持多层级 IR 转换
- 提供函数哈希和缓存

**相关文档**：[Function 类](03-function.md)

---

## 多层级 IR 系统

### IR（Intermediate Representation，中间表示）

**定义**：IR 是编译过程中用于表示程序结构的中间形式，介于源代码和机器代码之间。

**PyPTO 的 IR 特点**：
- 多层级设计，从高层次到低层次逐步转换
- 每一层 IR 都有特定的优化目标
- 支持增量优化和转换

### Tensor Graph（张量图）

**定义**：Tensor Graph 由 Tensor 和 Operation 节点构成，用于描述用户定义的计算流程。该图不涉及 Tile 展开与内存层级等底层语义，仅作为高层计算逻辑的表达。

**特点**：
- **抽象层次**：算法抽象，贴近数学表达式
- **核心组件**：`Function`、`Operation`、`LogicalTensor`
- **主要优化**：图级优化、内存冲突推断、冗余节点消除、常量折叠等
- **优化阶段**：Tensor Graph Pass 阶段

**转换过程**：
```
用户代码 → Tensor Graph → Tile Graph
```

**相关文档**：
- [Function 类](03-function.md)
- [Operation 类](04-operation.md)
- [Passes 模块 - Tensor Graph Pass](07-passes.md#tensor-graph-pass-详细说明)

### Tile Graph（分块图）

**定义**：Tile Graph 由 Tile 和 TileOp 构成。Tensor Graph 根据 TileShape 展开，将 Tensor 分解为 Tile，将 Operation 分解为 TileOp。Tile Graph 依据 TileOp 信息以及目标硬件的内存层级，自动推导 Tile 的存储位置，并在必要时插入内存搬运节点，确保数据在不同内存层级之间正确传输。

**特点**：
- **抽象层次**：硬件感知，体现内存层次结构
- **核心组件**：Tile、TileOp、内存层级信息
- **主要优化**：Tile 展开、内存类型分配、内存搬运优化等
- **优化阶段**：Tile Graph Pass 阶段

**转换过程**：
```
Tensor Graph → Tile Graph → Block Graph
```

**关键概念**：
- **Tile**：硬件感知的数据块，是计算的基本单位
- **TileShape**：定义张量分块（Tiling）形状的参数，决定了操作在硬件上的并行粒度与数据划分模式
- **内存层级**：L0（寄存器）、L1（片上缓存）、L2（全局内存）等

**相关文档**：
- [Passes 模块 - Tile Graph Pass](07-passes.md#tile-graph-pass-详细说明)

### Block Graph（块图）

**定义**：Block Graph 通过将 Tile Graph 切分为多个子图，使得每个子图可以调度运行在单个 AI Core 上。Block Graph 用于硬件相关的优化，包括指令编排、片上内存分配、同步操作插入等，从而提升硬件执行效率。

**特点**：
- **抽象层次**：并行执行，体现硬件并行性
- **核心组件**：子图分区、资源管理、同步操作
- **主要优化**：内存重用、乱序调度、同步插入等
- **优化阶段**：Block Graph Pass 阶段

**转换过程**：
```
Tile Graph → Block Graph → Execute Graph
```

**关键概念**：
- **子图（Subgraph）**：可以独立在单个 AI Core 上执行的图片段
- **内存重用**：优化内存分配，减少内存占用
- **乱序调度（OoO）**：允许不相关的操作并行执行

**相关文档**：
- [Passes 模块 - Block Graph Pass](07-passes.md#block-graph-pass-详细说明)

### Execute Graph（执行图）

**定义**：Execute Graph 是编译流程的最终产物，整合了所有优化结果，精确描述各 Block Graph 之间的依赖关系，用于设备调度器的调度执行。

**特点**：
- **抽象层次**：执行调度，体现运行时行为
- **核心组件**：依赖关系、调度信息、同步点
- **主要优化**：同步插入、调度优化、依赖分析等
- **最终产物**：用于代码生成和执行调度

**转换过程**：
```
Block Graph → Execute Graph → CCE 代码
```

**相关文档**：
- [Passes 模块 - Execute Graph Pass](07-passes.md#execute-graph-pass)
- [Machine 模块](06-machine.md)

---

## 编译流程概念

### Pass（优化阶段）

**定义**：PyPTO 中用于编译优化计算图的阶段，以提升计算图的执行效率和硬件利用率。每个 Pass 可以对计算图进行特定的优化操作，例如图简化、算子融合、调度优化等。

**Pass 分类**：
- **Tensor Graph Pass**：在 Tensor Graph 层级进行优化
- **Tile Graph Pass**：在 Tile Graph 层级进行优化
- **Block Graph Pass**：在 Block Graph 层级进行优化

**Pass 执行流程**：
```
PreRun() → PreCheck() → RunOnFunction() → PostCheck() → PostRun()
```

**相关文档**：
- [Passes 模块](07-passes.md)
- [Pass 框架](07-passes.md#pass-框架)

### Lowering（降级）

**定义**：将高层次的 IR 转换为低层次的 IR 的过程。

**PyPTO 中的 Lowering**：
- **Tensor Graph → Tile Graph**：将 Tensor 展开为 Tile，Operation 展开为 TileOp
- **Tile Graph → Block Graph**：将 Tile Graph 切分为多个子图
- **Block Graph → Execute Graph**：添加调度信息和依赖关系

**相关文档**：
- [Passes 模块](07-passes.md)

### Codegen（代码生成）

**定义**：将优化后的 IR 转换为可执行代码的过程。

**PyPTO 中的 Codegen**：
- **符号管理**：管理变量和符号表达式
- **操作代码生成**：为每个操作生成 CCE 代码
- **函数体生成**：生成完整的函数实现
- **编译优化**：编译器级别的优化

**相关文档**：
- [Codegen 模块](08-codegen.md)

---

## 执行调度概念

### MPMD（Multiple Program Multiple Data）

**定义**：MPMD 是一种并行执行模式，多个程序（Program）可以同时在不同的数据上执行。

**PyPTO 中的应用**：
- 多个 AI Core 可以同时执行不同的子图
- 支持数据并行和模型并行
- 通过调度器协调执行

**相关文档**：
- [Machine 模块](06-machine.md)

### 调度器（Scheduler）

**定义**：负责管理和协调任务执行的组件。

**PyPTO 中的调度器**：
- **主机端调度器**：管理主机端的任务调度
- **设备端调度器**：管理设备端的任务执行
- **乱序调度（OoO）**：允许不相关的任务并行执行

**相关文档**：
- [Machine 模块](06-machine.md)
- [Passes 模块 - OoOSchedule](07-passes.md#关键-pass-详解)

### 同步（Synchronization）

**定义**：确保多个任务按照正确的顺序执行的机制。

**PyPTO 中的同步**：
- **同步点（Sync Point）**：任务之间的同步点
- **同步操作插入**：在编译时自动插入同步操作
- **依赖关系**：通过依赖关系控制执行顺序

**相关文档**：
- [Passes 模块 - InsertSync](07-passes.md#关键-pass-详解)

---

## 内存管理概念

### 内存层级（Memory Hierarchy）

**定义**：PyPTO 中定义的不同层次的内存，从高速到低速依次为：

| 内存层级 | 说明 | 特点 |
|---------|------|------|
| **L0（寄存器）** | 最快的存储，容量最小 | 用于临时数据存储 |
| **L1（片上缓存）** | 片上高速缓存 | 用于频繁访问的数据 |
| **L2（全局内存）** | 设备全局内存 | 容量大，速度较慢 |
| **Host 内存** | 主机端内存 | 用于主机和设备之间的数据传输 |

**相关文档**：
- [Passes 模块 - AssignMemoryType](07-passes.md#关键-pass-详解)

### 内存重用（Memory Reuse）

**定义**：通过分析张量的生命周期，让多个张量共享同一块内存，从而减少内存占用。

**优化策略**：
- **全局内存重用**：跨子图的内存重用
- **局部内存重用**：子图内部的内存重用
- **缓冲区合并**：合并相邻的缓冲区

**相关文档**：
- [Passes 模块 - GlobalMemoryReuse](07-passes.md#关键-pass-详解)

### 内存冲突（Memory Conflict）

**定义**：多个操作同时访问同一块内存可能导致的数据竞争问题。

**推断方法**：
- 分析张量的生命周期
- 检查操作的执行顺序
- 识别潜在的内存冲突

**相关文档**：
- [Passes 模块 - InferMemoryConflict](07-passes.md#关键-pass-详解)

---

## 控制流概念

### 动态函数（Dynamic Function）

**定义**：包含运行时才能确定的控制结构的函数，如循环、条件分支等。

**特点**：
- 需要生成控制流代码
- 支持动态形状
- 运行时执行控制逻辑

**相关文档**：
- [控制流编译逻辑和日志分析](../03-mechanisms/06-controlflow.md)

### 控制流编译（Control Flow Compilation）

**定义**：为动态函数生成可执行的控制流代码的过程。

**编译流程**：
- **Host 端编译**：生成主机端控制流代码（x86_64）
- **Device 端编译**：生成设备端控制流代码（ARM64/NPU）
- **表达式收集**：收集运行时表达式
- **代码生成**：生成 `ControlFlowEntry` 函数

**相关文档**：
- [控制流编译逻辑和日志分析](../03-mechanisms/06-controlflow.md)

### RUNTIME_SetExpr

**定义**：运行时设置符号表达式的宏/函数，用于在运行时计算动态形状和符号值。

**用途**：
- 设置动态形状表达式
- 设置符号变量值
- 支持运行时计算

**相关文档**：
- [控制流编译逻辑和日志分析 - RUNTIME_SetExpr 拆分优化](../03-mechanisms/00-overview.md#runtime_setexpr-拆分优化说明)

---

## 数据类型和格式

### 数据类型（DataType）

**定义**：PyPTO 中支持的数据类型。

| 类型值 | 类型名称 | 说明 |
|--------|---------|------|
| 7 | FP32 | 32 位浮点数 |
| 6 | FP16 | 16 位浮点数 |
| 5 | INT32 | 32 位整数 |
| 4 | INT16 | 16 位整数 |
| 3 | INT8 | 8 位整数 |

### 数据格式（Format）

**定义**：张量数据的存储格式，如 NCHW、NHWC 等。

**相关文档**：
- [Operation 类](04-operation.md)

---

## 优化和 Pass

### 图优化（Graph Optimization）

**定义**：对计算图进行结构优化，如节点消除、融合、拆分等。

**常见优化**：
- **冗余节点消除**：移除不必要的操作
- **算子融合**：将多个操作融合为一个
- **图拆分**：将大图拆分为多个子图

**相关文档**：
- [Passes 模块](07-passes.md)

### 内存优化（Memory Optimization）

**定义**：优化内存使用，减少内存占用和提高访问效率。

**优化策略**：
- 内存重用
- 内存类型分配
- 缓冲区合并

**相关文档**：
- [Passes 模块 - 内存优化 Pass](07-passes.md#pass-分类详解)

### 调度优化（Scheduling Optimization）

**定义**：优化任务的执行顺序，提高并行度和执行效率。

**优化策略**：
- 乱序调度（OoO）
- 依赖分析
- 同步优化

**相关文档**：
- [Passes 模块 - 调度优化 Pass](07-passes.md#pass-分类详解)

---

## 相关概念索引

### 按模块分类

- **Interface 模块**：[Interface 模块](02-interface.md)
- **Passes 模块**：[Passes 模块](07-passes.md)
- **Codegen 模块**：[Codegen 模块](08-codegen.md)
- **Machine 模块**：[Machine 模块](06-machine.md)

### 按主题分类

- **编译流程**：[框架总览 - 完整编译执行流程](00-overview.md#完整编译执行流程)
- **控制流**：[控制流编译逻辑和日志分析](../03-mechanisms/00-overview.md)
- **调试**：[完整调试指南](../05-debugging/00-complete-guide.md)
- **测试**：[测试和验证方法](../06-testing/00-methodology.md)

---

## 更新日志

- 2025-12-30: 创建概念解释文档

