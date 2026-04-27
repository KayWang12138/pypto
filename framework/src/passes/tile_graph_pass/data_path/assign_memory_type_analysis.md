# assign_memory_type.cpp 代码逻辑详细分析报告

## 一、文件概述

**文件路径**: `/mnt/workspace/gitCode/ym-HW/pypto/framework/src/passes/tile_graph_pass/data_path/assign_memory_type.cpp`

**功能定位**: NPU Tile 框架中的编译优化 Pass，负责为计算图中的所有张量分配合适的内存存储类型。

**核心职责**:
1. 根据 Opcode 定义推导张量内存类型
2. 处理特殊 Op (View/Assemble/Reshape/NOP等) 的内存类型
3. 处理内存对齐、容量阈值等约束
4. 插入必要的内存转换 Op (Convert)

---

## 二、类成员变量说明

```cpp
// 推断的辅助工具类（未在本文件中定义，推测功能）
AssignMemoryTypeInserter inserter;  // 负责插入 Convert Op 和维护 tobeMap
AssignMemoryTypeChecker checker;    // 前置/后置检查器

// 常量阈值
static constexpr float UB_THRESHOLD = ?;  // UB 缓冲区阈值比例
static constexpr float L1_THRESHOLD = ?;  // L1 缓冲区阈值比例
```

---

## 三、函数详细分析

### 3.1 SetIncastOutcastMemtype

**位置**: Line 30-53

**功能**: 设置输入转换节点 (INCAST) 和输出转换节点 (OUTCAST) 的内存类型。

**逻辑流程**:
```
┌─────────────────────────────────────────────────────────┐
│                    INCAST 处理                          │
├─────────────────────────────────────────────────────────┤
│ 1. 遍历 function.inCasts_                               │
│ 2. 设置 INCAST 内存类型为 MEM_DEVICE_DDR                │
│ 3. 遍历 INCAST 的每个消费者 Op                          │
│ 4. 将 (INCAST, consumerOp, DDR) 加入 tobeMap           │
└─────────────────────────────────────────────────────────┘

数据流示例:
         /--> op1 --> tensor1 -->
INCAST  ---> op2 --> tensor2 -->
         \--> op3 --> tensor3 -->

tobeMap 记录: {INCAST: {op1: DDR, op2: DDR, op3: DDR}}
```

```
┌─────────────────────────────────────────────────────────┐
│                    OUTCAST 处理                         │
├─────────────────────────────────────────────────────────┤
│ 1. 遍历 function.outCasts_                             │
│ 2. 设置 OUTCAST 内存类型为 MEM_DEVICE_DDR               │
│ 3. OUTCAST 没有消费者，tobeMap 为空                     │
└─────────────────────────────────────────────────────────┘

数据流示例:
op --> tensor --> op1 -->\
op --> tensor --> op2 ---> OUTCAST
op --> tensor --> op3 -->/
```

**关键点**:
- INCAST/OUTCAST 是图的边界节点，代表与外部 DDR 的交互
- `SetMemoryTypeBoth()` 同时设置 original 和 tobe 类型
- INCAST 的 tobeMap 记录其输出张量对各消费者所需的内存类型

---

### 3.2 RunOnFunction (主入口)

**位置**: Line 55-92

**功能**: Pass 的主执行函数，协调整个内存类型分配流程。

**执行阶段**:

| 阶段 | 函数 | 作用 |
|------|------|------|
| Step 1 | SetIncastOutcastMemtype | 初始化边界节点内存类型 |
| Step 1 | RunOnOperation (遍历) | 根据 Opcode 定义推导基础内存类型 |
| Step 2 | AssignMoveOp (遍历) | 处理连接型 Op (Assemble/View) 的内存传递 |
| Step 2 | AssignMemUnknown | 处理未定义内存类型的张量 |
| Step 3 | AssignSpecialOpMemtype | 处理 Reshape/ViewType/NOP/ReduceACC 等特殊 Op |
| Step 4 | ProcesSmallTileToLargeTile | Cube 级联场景：小 Tile 搬运到大 Tile |
| Step 4 | ProcessLargeTileToSamllTile | Cube 级联场景：大 Tile 搬运到小 Tile |
| Step 5 | DoInsertion | 根据内存类型差异插入 Convert Op |

**流程图**:
```
┌──────────────────────────────────────────────────────────────┐
│                    RunOnFunction                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Phase 1: 基础内存类型推导                            │    │
│  │  - SetIncastOutcastMemtype()                        │    │
│  │  - for each op: RunOnOperation()                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Phase 2: 连接 Op 内存类型传递                        │    │
│  │  - for each op: AssignMoveOp()                      │    │
│  │  - AssignMemUnknown()                               │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Phase 3: 特殊 Op 处理                               │    │
│  │  - for each op: AssignSpecialOpMemtype()           │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Phase 4: Cube 级联约束处理                          │    │
│  │  - ProcesSmallTileToLargeTile()                     │    │
│  │  - ProcessLargeTileToSamllTile()                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Phase 5: 插入 Convert Op                            │    │
│  │  - inserter.DoInsertion()                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

### 3.3 PreCheck / PostCheck

**位置**: Line 94-96

**功能**: 前置和后置检查函数，委托给 `checker` 执行具体检查逻辑。

**职责推测**:
- **PreCheck**: 检查图的合法性、Op 连接正确性
- **PostCheck**: 检查内存类型分配结果的一致性、完整性

---

### 3.4 RunOnOperation

**位置**: Line 98-153

**功能**: 为单个 Operation 设置其输入/输出张量的内存类型。

**核心逻辑**:

```
┌──────────────────────────────────────────────────────────────┐
│                   输入张量处理                                 │
├──────────────────────────────────────────────────────────────┤
│ for each input tensor (i = 0 to iOperand.size()):            │
│                                                              │
│   1. 获取 OpcodeManager 定义的输入内存类型 inputsMemType[i]  │
│                                                              │
│   2. 如果索引越界:                                            │
│      - 记录日志 "mem original is NOT Defined in opcode.cpp"  │
│      - continue (跳过此输入)                                  │
│                                                              │
│   3. 特殊处理: 如果 Op 是 MATMUL 类型:                        │
│      - 调用 ProcessAmulBInput()                              │
│      - continue                                              │
│                                                              │
│   4. 常规处理:                                                │
│      - tensor->SetMemoryTypeOriginal(inputsMemType[i])      │
│      - inserter.UpdateTensorTobeMap(tensor, op, memType)    │
└──────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────┐
│                   输出张量处理                                 │
├──────────────────────────────────────────────────────────────┤
│ for each output tensor (i = 0 to oOperand.size()):           │
│                                                              │
│   1. 如果 outputsMemType 非空:                                │
│      - tensor->SetMemoryTypeOriginal(outputsMemType[i])      │
│      - 为每个消费者更新 tobeMap                               │
│      - continue                                              │
│                                                              │
│   2. 如果 outputsMemType 为空 (未定义):                       │
│      - tensor->SetMemoryTypeOriginal(MEM_UNKNOWN)           │
│      - 为每个消费者设置 tobeMap 为 MEM_UNKNOWN               │
└──────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────┐
│                   特殊 Op 后处理                               │
├──────────────────────────────────────────────────────────────┤
│ if (opcode == OP_VIEW):                                      │
│     ProcessViewwithSpecificMem(op)                           │
│                                                              │
│ if (opcode == OP_ASSEMBLE):                                  │
│     ProcessAssemblewithSpecificMem(op)                       │
└──────────────────────────────────────────────────────────────┘
```

**关键概念**:

| 概念 | 说明 |
|------|------|
| MemoryTypeOriginal | 张量的原始内存类型（生产者决定） |
| MemoryTypeToBe | 张量对特定消费者需要的内存类型 |
| tobeMap | 记录张量对各消费者的目标内存类型映射 |

---

### 3.5 ProcessAmulBInput

**位置**: Line 154-191

**功能**: 处理矩阵乘法 Op (A_MUL_B / A_MULACC_B) 的输入张量内存类型。

**场景分析**:

```
矩阵乘法输入可能来自不同类型的生产者:

┌─────────────────────────────────────────────────────────────────┐
│ Case 1: 生产者是另一个 MATMUL                                   │
├─────────────────────────────────────────────────────────────────┤
│   Producer: MATMUL (输出在 L0C)                                 │
│   → tensor->SetMemoryTypeOriginal(MEM_L0C)                      │
│   → tobeMap: {tensor: {currentOp: L0C}}                        │
│   → 级联矩阵乘法场景，输出直接作为下一个矩阵乘法的输入            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Case 2: 生产者是 VIEW Op                                        │
├─────────────────────────────────────────────────────────────────┤
│   Producer: VIEW (带有 ToAttr 属性)                             │
│   → 从 ViewOpAttribute 获取目标类型                             │
│   → tensor->SetMemoryTypeOriginal(attrToType)                  │
│   → tobeMap: {tensor: {currentOp: attrToType}}                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Case 3: 生产者是 MOVE_LOCAL (L1→L0A)                            │
├─────────────────────────────────────────────────────────────────┤
│   Producer: MOVE_LOCAL (输入 L1, 输出 L0A)                       │
│   → tensor->SetMemoryTypeOriginal(MEM_L0A)                     │
│   → tobeMap: {tensor: {currentOp: L0A}}                        │
│   → 矩阵 A 从 L1 加载到 L0A                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Case 4: 生产者是 MOVE_LOCAL (L1→L0B)                            │
├─────────────────────────────────────────────────────────────────┤
│   Producer: MOVE_LOCAL (输入 L1, 输出 L0B)                       │
│   → tensor->SetMemoryTypeOriginal(MEM_L0B)                     │
│   → tobeMap: {tensor: {currentOp: L0B}}                        │
│   → 矩阵 B 从 L1 加载到 L0B                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Case 5: 其他生产者                                              │
├─────────────────────────────────────────────────────────────────┤
│   → tobeMap: {tensor: {currentOp: MEM_DEVICE_DDR}}             │
│   → 默认从 DDR 读取                                             │
└─────────────────────────────────────────────────────────────────┘
```

**NPU 内存层级关系**:
```
DDR (全局内存)
  ↓
L1 (L1 Buffer)
  ↓
┌─────────┬─────────┐
│   L0A   │   L0B   │  (Cube 单元输入 Buffer)
└────┬────┴────┬────┘
     │         │
     └────┬────┘
          ↓
         L0C      (Cube 单元输出 Buffer)
```

---

### 3.6 ProcessViewwithSpecificMem

**位置**: Line 193-226

**功能**: 处理 View Op 的内存类型，支持 L0C→L1 特殊通路。

**逻辑流程**:

```
┌──────────────────────────────────────────────────────────────┐
│                    View Op 结构图                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│     input_tensor ──→ VIEW ──→ output_tensor                  │
│         ↑                        ↑                           │
│    original type            original type                     │
│    (from producer)          (from consumer)                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Step 1: 检查 L0C→L1 通路适配**

```cpp
if (in->GetMemoryTypeOriginal() == MEM_L0C &&
    (out->GetMemoryTypeOriginal() == MEM_L1 || attrToType == MEM_L1)) {
    
    if (inserter.FitL0C2L1(operation)) {
        // 支持 L0C→L1 直接搬运
        inserter.UpdateTensorTobeMap(in, operation, MEM_L0C);
    } else {
        // 不支持，需要绕道 DDR
        inserter.UpdateTensorTobeMap(in, operation, MEM_DEVICE_DDR);
    }
}
```

**Step 2: 处理显式内存类型属性**

```cpp
if (attrToType == MEM_UNKNOWN) {
    // 前端未指定内存类型，跳过
    return;
}

// 将 View 输出的 original 和 tobe 类型设置为 attrToType
out->SetMemoryTypeOriginal(attrToType, true);
for (auto& consumerOp : out->GetConsumers()) {
    inserter.UpdateTensorTobeMap(out, *consumerOp, attrToType);
}
```

**Step 3: 处理 L1 大包搬运场景**

```cpp
if (attrToType == MEM_L1) {
    // 如果前序也是 View，将输入也设为 L1
    for (const auto& producerOp : producerOps) {
        if (producerOp->GetOpcode() == OP_VIEW) {
            in->SetMemoryTypeOriginal(attrToType, true);
            inserter.UpdateTensorTobeMap(in, operation, attrToType);
        }
    }
}
```

**场景示例**:
```
场景: L0C→L1 通路

MatMul (输出 L0C) → View (To=L1) → 后续 Op
     ↓                    ↓
   L0C                  L1

检查 FitL0C2L1():
  - 支持 → 直接搬运，输入 tobe=L0C
  - 不支持 → 绕道 DDR，输入 tobe=DDR
```

---

### 3.7 ProcessAssemblewithSpecificMem

**位置**: Line 228-261

**功能**: 处理 Assemble Op 的内存类型，适配 L0C→L1 通路。

**前置条件检查**:

```cpp
// 条件 1: 输入必须是 L0C
if (input->GetMemoryTypeOriginal() != MEM_L0C) {
    return;
}

// 条件 2: 必须支持 L0C→L1 通路
if (!inserter.FitL0C2L1(operation)) {
    return;
}
```

**消费者检查**:

```cpp
for (const auto& consumerOp : output->GetConsumers()) {
    // 消费者必须是 View 且 To=L1，或者消费者需要 L1 输入
    auto consumerOpAttribute = dynamic_pointer_cast<ViewOpAttribute>(...);
    
    if (consumerOpAttribute && consumerOpAttribute->GetTo() != MEM_UNKNOWN) {
        // 大包搬运场景：View 的 To 属性必须是 L1
        if (consumerOpAttribute->GetTo() != MEM_L1) {
            return;  // 不满足条件，退出
        }
    } else {
        // 常规场景：消费者的输入类型必须是 L1
        if (inputsMemType[0] != MEM_L1) {
            return;  // 不满足条件，退出
        }
    }
}
```

**满足条件后的设置**:

```cpp
// 输出设为 L1
output->SetMemoryTypeOriginal(MEM_L1, true);

// 输入 tobe 设为 L0C
inserter.UpdateTensorTobeMap(input, operation, MEM_L0C);

// 输出 tobe 设为 L1
for (const auto& consumerOp : output->GetConsumers()) {
    inserter.UpdateTensorTobeMap(output, *consumerOp, MEM_L1);
}
```

**数据流示例**:
```
优化前 (不支持 L0C→L1):
L0C → Assemble → DDR → Convert → L1 → 后续Op

优化后 (支持 L0C→L1):
L0C → Assemble → L1 → 后续Op
```

---

### 3.8 AssignMemtypeForSplitReshape

**位置**: Line 263-285

**功能**: 为 Split Reshape 场景分配内存类型。

**识别模式**:
```
Assemble → View → Reshape → View → ...
                ↑
            检测这个 Reshape
```

**逻辑**:

```cpp
void AssignMemtypeForSplitReshape(Operation& op, ...) {
    // 计算阈值
    const int UB_SIZE_THRESHOLD = GetUBLimit() * UB_THRESHOLD;
    
    // 获取生产者和消费者
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();
    Operation* producer = *producers.begin();
    Operation* consumer = *consumers.begin();
    
    // 模式识别: Assemble → ? → Reshape → View
    if (producer != nullptr && 
        consumer != nullptr &&
        producer->GetOpcode() == OP_ASSEMBLE &&
        consumer->GetOpcode() == OP_VIEW) {
        
        // 检查内存类型和大小约束
        if (input->GetMemoryTypeOriginal() == MEM_UB &&
            output->GetMemoryTypeOriginal() == MEM_UB &&
            input->GetDataSize() <= UB_SIZE_THRESHOLD) {
            
            // 更新 tobeMap
            inserter.UpdateTensorTobeMap(input, op, MEM_UB);
            
            // 级联更新后续 View 的输出
            for (const auto& consumerOp : output->GetConsumers()) {
                if (consumerOp->oOperand.front()->GetMemoryTypeOriginal() == MEM_UB) {
                    inserter.UpdateTensorTobeMap(output, *consumerOp, MEM_UB);
                }
            }
        }
    }
}
```

**条件总结**:
1. 生产者是 Assemble
2. 消费者是 View
3. 输入/输出都是 UB 类型
4. 数据大小不超过 UB 阈值

---

### 3.9 AssignOpReshapeMemtype

**位置**: Line 287-302

**功能**: 处理 Reshape Op 的内存类型。

**逻辑**:

```cpp
void AssignOpReshapeMemtype(Operation& op) {
    if (op.GetOpcode() != OP_RESHAPE) {
        return;
    }
    
    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();
    
    // 尝试 Split Reshape 特殊处理
    AssignMemtypeForSplitReshape(op, input, output);
    
    // 获取输入的 tobe 类型
    auto inputMemType = inserter.GetMemoryTypeFromTensorTobeMap(input, op);
    
    // 如果输入 tobe 与输出 original 不一致
    if (inputMemType != output->GetMemoryTypeOriginal()) {
        // 降级为 DDR
        inserter.UpdateTensorTobeMap(input, op, MEM_DEVICE_DDR);
        output->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    }
}
```

**处理策略**:
```
Reshape Op 的内存类型传递:

情况 1: 输入 tobe == 输出 original
  → 保持，无需额外处理

情况 2: 输入 tobe != 输出 original
  → 无法在原地 Reshape
  → 强制使用 DDR
  → 可能需要插入 Convert
```

---

### 3.10 AssignOpViewTypeMemtype

**位置**: Line 304-325

**功能**: 处理 ViewType Op 的内存类型。

**逻辑**:

```cpp
void AssignOpViewTypeMemtype(Operation& op) {
    if (op.GetOpcode() != OP_VIEW_TYPE) {
        return;
    }
    
    auto& viewTypeIn = op.iOperand.front();
    auto& viewTypeOut = op.oOperand.front();
    
    // 获取输入 tobe 和输出默认 tobe
    auto inputMemType = inserter.GetMemoryTypeFromTensorTobeMap(viewTypeIn, op);
    auto outTobeMem = inserter.GetTobeDefault(viewTypeOut);
    
    // 获取生产者
    auto prod = *(viewTypeIn->GetProducers().begin());
    
    // 情况 1: 前序是 View → 内存复用
    if (prod->GetOpcode() == OP_VIEW) {
        viewTypeIn->SetMemoryTypeOriginal(viewTypeOut->GetMemoryTypeOriginal(), true);
        inserter.UpdateTensorTobeMap(viewTypeIn, op, viewTypeOut->GetMemoryTypeOriginal());
        return;
    }
    
    // 情况 2: 输入 tobe != 输出 original → 降级 DDR
    if (inputMemType != viewTypeOut->GetMemoryTypeOriginal()) {
        inserter.UpdateTensorTobeMap(viewTypeIn, op, MEM_DEVICE_DDR);
        viewTypeOut->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    }
}
```

**与 Reshape 处理的异同**:
- 相同: 内存类型不一致时降级 DDR
- 不同: ViewType 有额外的前序 View 内存复用逻辑

---

### 3.11 AssignOpNopMemtype

**位置**: Line 327-341

**功能**: 处理 NOP (No Operation) 的内存类型。

**逻辑**:

```cpp
void AssignOpNopMemtype(Operation& op) {
    if (op.GetOpcode() != OP_NOP) {
        return;
    }
    
    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();
    
    // 情况 1: 输入 tobe != 输出 original
    if (input->GetMemoryTypeToBe() != output->GetMemoryTypeOriginal()) {
        // 强制使用 DDR
        input->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
        output->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    }
    
    // 情况 2: 输出 original != 输出 tobe
    // NOP 不改变数据类型，输出必须与输入一致
    if (output->GetMemoryTypeOriginal() != output->GetMemoryTypeToBe()) {
        output->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
    }
}
```

**NOP 的特殊性**:
- NOP 是占位操作，不改变数据
- 因此输入和输出必须在同一内存区域
- 如果消费者需要不同内存类型，必须在 NOP 之后插入 Convert

---

### 3.12 AssignSpecialOpMemtype

**位置**: Line 343-380

**功能**: 处理多种特殊 Op 的内存类型。

**处理的 Op 类型**:

| Op 类型 | 处理方式 |
|---------|----------|
| RESHAPE | 调用 AssignOpReshapeMemtype |
| VIEW_TYPE | 调用 AssignOpViewTypeMemtype |
| NOP | 调用 AssignOpNopMemtype |
| REDUCE_ACC | 所有输入设为 DDR |
| SHMEM_WAIT_UNTIL | 所有输出设为 DDR |
| ASSEMBLE | 检查超尺寸缓冲区 |

**REDUCE_ACC 处理**:
```cpp
if (op.GetOpcode() == OP_REDUCE_ACC) {
    // 输入数量不确定，由 Ksplit 决定
    // 每个输入都设为 DDR
    for (auto& input : op.GetIOperands()) {
        inserter.UpdateTensorTobeMap(input, op, MEM_DEVICE_DDR);
    }
}
```

**SHMEM_WAIT_UNTIL 处理**:
```cpp
if (op.GetOpcode() == OP_SHMEM_WAIT_UNTIL) {
    // 所有输出设为 DDR
    for (auto& output : op.GetOOperands()) {
        output->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    }
}
```

**ASSEMBLE 处理**:
```cpp
if (op.GetOpcode() == OP_ASSEMBLE) {
    // 如果输出是 L1 但 tobe 是 DDR，保持 original
    auto& output = op.oOperand.front();
    if (output->GetMemoryTypeOriginal() == MEM_L1 &&
        output->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
        output->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
    }
    
    // 更新超尺寸缓冲区
    UpdateOverSizedLocalBuffer(op);
    infoBufferSize = true;
}
```

---

### 3.13 UpdateOverSizedLocalBuffer

**位置**: Line 382-398

**功能**: 检查并处理超尺寸的本地缓冲区。

**逻辑**:

```cpp
void UpdateOverSizedLocalBuffer(Operation& operation) {
    // 获取平台阈值
    const int UB_SIZE_THRESHOLD = GetUBLimit() * UB_THRESHOLD;
    const int L1_SIZE_THRESHOLD = GetL1Limit() * L1_THRESHOLD;
    
    auto assembleOut = operation.GetOOperands().front();
    auto memType = assembleOut->GetMemoryTypeOriginal();
    
    // 检查是否超过阈值
    bool isOverSized = 
        (memType == MEM_UB && assembleOut->GetDataSize() > UB_SIZE_THRESHOLD) ||
        (memType == MEM_L1 && assembleOut->GetDataSize() > L1_SIZE_THRESHOLD);
    
    if (isOverSized) {
        // 降级为 DDR
        assembleOut->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
        APASS_LOG_INFO_F("output is oversized, set as MEM_DEVICE_DDR");
    }
}
```

**阈值逻辑**:
```
UB 容量 = 平台 UB 限制 * UB_THRESHOLD
L1 容量 = 平台 L1 限制 * L1_THRESHOLD

如果 Assemble 输出:
  - 类型是 UB 且大小 > UB 容量 → 降级 DDR
  - 类型是 L1 且大小 > L1 容量 → 降级 DDR
```

---

### 3.14 PrintTensorMem

**位置**: Line 400-407

**功能**: 调试辅助函数，打印张量的内存类型信息。

**输出格式**:
```
tensor magic: <id> original: <type>, tobe: <type>
```

---

### 3.15 AssignMoveOp

**位置**: Line 409-424

**功能**: 分发函数，根据 Opcode 调用对应的处理函数。

**处理映射**:
```cpp
switch (opcode) {
    case OP_ASSEMBLE:
        AssignMoveOpForAssemble(operation);
        break;
    case OP_VIEW:
        AssignMoveOpForView(operation);
        break;
    default:
        break;
}
```

---

### 3.16 CalcLineOffset

**位置**: Line 426-443

**功能**: 根据 Shape 和 Offset 计算线性偏移量。

**算法**:

```cpp
int64_t CalcLineOffset(const Shape& shape, const Offset& offset) {
    // 参数校验
    if (shape.size() != offset.size()) return -1;
    if (shape.size() == 0) return 0;
    
    int64_t lineOffset = 0;
    int64_t stride = 1;
    
    // 从最低维到最高维计算
    // 例如: shape = [N, C, H, W], offset = [n, c, h, w]
    // lineOffset = w + h*W + c*H*W + n*C*H*W
    for (size_t i = shape.size(); i > 0; --i) {
        lineOffset += offset[i - 1] * stride;
        stride *= shape[i - 1];
    }
    
    return lineOffset;
}
```

**示例**:
```
Shape: [2, 3, 4]  (N=2, C=3, H=4)
Offset: [1, 2, 3] (n=1, c=2, h=3)

计算过程:
  i=3: lineOffset += 3 * 1 = 3,    stride = 4
  i=2: lineOffset += 2 * 4 = 11,   stride = 12
  i=1: lineOffset += 1 * 12 = 23, stride = 24

结果: lineOffset = 23
```

---

### 3.17 AssignMoveOpForAssemble

**位置**: Line 445-513

**功能**: 处理 Assemble Op 的内存类型传递和对齐检查。

**核心逻辑**:

```
┌──────────────────────────────────────────────────────────────┐
│              Assemble Op 内存类型传递                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Op1 ──→ tensor1 ──┐                                         │
│  Op2 ──→ tensor2 ──┼──→ Assemble ──→ output_tensor           │
│  Op3 ──→ tensor3 ──┘                                         │
│                                                              │
│  输入类型: 由各个生产者决定                                    │
│  输出类型: 传递自输入 + 对齐检查                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Step 1: 确定输入内存类型**
```cpp
// 从 tobeMap 获取第一个输入的内存类型
MemoryType fromType = inserter.GetMemoryTypeFromTensorTobeMap(
    operation.iOperand.front(), operation);

// 检查所有生产者的输入是否一致
for (const auto& outputProducer : tensor->GetProducers()) {
    if (fromType != MEM_DEVICE_DDR && 
        outputProducer->iOperand.front()->GetMemoryTypeOriginal() != fromType) {
        // 类型不一致，降级 DDR
        fromType = MEM_DEVICE_DDR;
    }
}
```

**Step 2: 对齐检查**
```cpp
static constexpr int UB_ALIGN_BYTES = 32;

for (const auto& outputProducer : tensor->GetProducers()) {
    // 获取 Assemble 的偏移属性
    auto opAttr = dynamic_pointer_cast<AssembleOpAttribute>(...);
    auto offset = opAttr->GetToOffset();
    
    // 计算字节偏移
    int64_t lineOffset = CalcLineOffset(shape, offset);
    int64_t byteOffset = BytesOf(tensor->Datatype()) * lineOffset;
    
    // 32B 对齐检查
    if (byteOffset % UB_ALIGN_BYTES != 0) {
        hasDdr = true;
        break;
    }
}
```

**Step 3: 应用内存类型**
```cpp
if (hasDdr) {
    fromType = MEM_DEVICE_DDR;
}

// 特殊情况: L0C→L1 跳过
if (input->GetMemoryTypeOriginal() == MEM_L0C &&
    tensor->GetMemoryTypeOriginal() == MEM_L1) {
    continue;
}

// 设置输出内存类型
tensor->SetMemoryTypeOriginal(fromType, true);

// 更新 Op 属性
assembleOpAttribute->SetFromType(fromType);
```

**对齐检查原理**:
```
UB 内存访问要求 32 字节对齐。

示例:
  Assemble 输出: shape = [128], dtype = float32 (4 bytes)
  Offset = [8]
  
  byteOffset = 8 * 4 = 32  ✓ 对齐 (32 % 32 == 0)
  
  Offset = [7]
  byteOffset = 7 * 4 = 28  ✗ 不对齐 (28 % 32 != 0) → 降级 DDR
```

---

### 3.18 AssignMoveOpForView

**位置**: Line 514-566

**功能**: 处理 View Op 的内存类型传递。

**核心逻辑**:

```
┌──────────────────────────────────────────────────────────────┐
│                  View Op 内存类型传递                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  tensor1 ──→ View ──→ tensor2 ──→ Op                         │
│     ↑               ↑                                        │
│  original       original (由消费者决定)                        │
│  需要更新 tobe                                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Step 1: 显式内存类型处理**
```cpp
MemoryType attrToType = viewOpAttribute->GetTo();
bool isExplicitMemType = (attrToType != MEM_UNKNOWN);

if (isExplicitMemType) {
    // 大包搬运场景: To=L1
    if (attrToType == MEM_L1 && 
        inserter.CrossCore(input->GetMemoryTypeOriginal(), attrToType)) {
        inserter.UpdateTensorTobeMap(input, operation, attrToType);
    }
    return;
}
```

**Step 2: 对齐检查**
```cpp
auto viewOffset = viewOpAttribute->GetFromOffset();
bool unaligned = ((BytesOf(outputTensor->Datatype()) * viewOffset.back()) % 32 != 0);
```

**Step 3: 内存复用逻辑**
```cpp
for (auto& tensor : operation.iOperand) {
    MemoryType toType = outputTensor->GetMemoryTypeOriginal();
    auto originalMemType = tensor->GetMemoryTypeOriginal();
    
    // 复用条件:
    // 1. 输出 original 未知但输入 original 已知
    // 2. 输入不是 L0C (L0C→L0C 无意义)
    // 3. 未对齐
    bool memTypeSupportReuse = 
        toType == MEM_UNKNOWN && 
        originalMemType != MEM_UNKNOWN &&
        originalMemType != MEM_L0C;
    
    if (!unaligned && memTypeSupportReuse) {
        // 内存复用: 输出继承输入类型
        outputTensor->SetMemoryTypeOriginal(originalMemType);
        viewOpAttribute->SetToType(originalMemType);
        continue;
    }
    
    // L0C→L1 特殊处理
    if (originalMemType == MEM_L0C && 
        outputTensor->GetMemoryTypeOriginal() == MEM_L1 &&
        inserter.FitL0C2L1(operation)) {
        inserter.UpdateTensorTobeMap(tensor, operation, MEM_L0C);
        continue;
    }
    
    // 默认: 输入 tobe 设为输出 original
    inserter.UpdateTensorTobeMap(tensor, operation, toType);
    viewOpAttribute->SetToType(toType);
}
```

**Step 4: 未对齐处理**
```cpp
if (unaligned) {
    auto inputTensor = operation.GetIOperands().front();
    inserter.UpdateTensorTobeMap(inputTensor, operation, MEM_DEVICE_DDR);
}
```

**内存复用场景**:
```
优化前:
  UB → View (输出类型未知) → Op (需要 DDR)
  → 需要插入 Convert

优化后 (内存复用):
  UB → View (输出=UB) → Op
  → 可能不需要 Convert
```

---

### 3.19 AssignMemUnknown

**位置**: Line 568-620

**功能**: 处理所有内存类型为 UNKNOWN 的张量。

**逻辑**:

```
┌──────────────────────────────────────────────────────────────┐
│                UNKNOWN 内存类型处理策略                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  情况 1: 张量的 tobeMap 只有一个类型且不是 UNKNOWN             │
│    → 使用该类型                                              │
│                                                              │
│  情况 2: 其他情况                                            │
│    → 使用 DDR                                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**输入张量处理**:
```cpp
for (auto& i : op.iOperand) {
    if (i->GetMemoryTypeOriginal() == MEM_UNKNOWN) {
        MemoryType fromType = MEM_DEVICE_DDR;
        
        // 获取该张量的所有消费者需求
        auto localTobeMap = inserter.GetRequiredTobe(i);
        
        // 如果只有一个需求且不是 UNKNOWN，使用该类型
        if (localTobeMap.size() == 1 && 
            localTobeMap.begin()->first != MEM_UNKNOWN) {
            fromType = localTobeMap.begin()->first;
        }
        
        i->SetMemoryTypeOriginal(fromType);
        inserter.UpdateTensorTobeMap(i, op, fromType);
    }
    
    // 更新 tobeMap 中的 UNKNOWN
    inserter.UpdateTensorTobeMapUnknown(i, i->GetMemoryTypeOriginal());
}
```

**输出张量处理**:
```cpp
for (auto& o : op.oOperand) {
    if (o->GetMemoryTypeOriginal() == MEM_UNKNOWN) {
        MemoryType fromType = MEM_DEVICE_DDR;
        
        auto localTobeMap = inserter.GetRequiredTobe(o);
        
        if (localTobeMap.size() == 1 && 
            localTobeMap.begin()->first != MEM_UNKNOWN) {
            fromType = localTobeMap.begin()->first;
        }
        
        o->SetMemoryTypeOriginal(fromType);
        for (const auto& consumerOp : o->GetConsumers()) {
            inserter.UpdateTensorTobeMap(o, *consumerOp, fromType);
        }
    }
    
    inserter.UpdateTensorTobeMapUnknown(o, o->GetMemoryTypeOriginal());
}
```

**UNKNOWN 出现的场景**:
1. OP_VIEW: 输出类型由消费者决定
2. OP_COPY_IN: 特殊复制操作
3. OP_RESHAPE: 输出类型不确定
4. 输出数量超过 Opcode 定义 (如 OP_REDUCE_ACC)

---

### 3.20 ProcesSmallTileToLargeTile

**位置**: Line 622-667

**功能**: 处理 Cube 级联场景下小 Tile 搬运到大 Tile 的约束。

**场景**:
```
Cube 级联: 多个 Cube 操作的输出需要合并

L0C (小) ──→ Assemble ──→ L1 (大)
  ↑
多个小 Tile
```

**逻辑**:

```cpp
for (auto& op : function.Operations()) {
    if (op.GetOpcode() != OP_ASSEMBLE) continue;
    
    auto oOperand = op.GetOOperands().front();
    auto iOperand = op.GetIOperands().front();
    
    // 条件 1: 输入必须是 L0C
    if (iOperand->GetMemoryTypeOriginal() != MEM_L0C) continue;
    
    // 条件 2: 所有消费者都要求 L1
    bool isToL1 = true;
    auto toBeMap = inserter.GetMemoryTypeFromTensorTobeMap(oOperand);
    for (const auto& pair : toBeMap) {
        if (pair.second != MEM_L1) {
            isToL1 = false;
            break;
        }
    }
    
    // 条件 3: 维度倍数检查
    // 输出维度必须是输入维度的正整数倍
    bool isConsumerOutputMultiple = true;
    for (auto& consumerOp : oOperand->GetConsumers()) {
        if (consumerOp->GetOpcode() == OP_VIEW &&
            !IsDimMultiple(consumerOp->GetOOperands().front()->GetShape(), 
                          iOperand->GetShape())) {
            isConsumerOutputMultiple = false;
            break;
        }
    }
    
    // 条件 4: 输出维度是输入维度的倍数
    if (!isToL1 || 
        !IsDimMultiple(oOperand->GetShape(), iOperand->GetShape()) ||
        !isConsumerOutputMultiple) {
        
        // 不满足条件，降级 DDR
        oOperand->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
        
        // 更新 tobeMap
        for (const auto& [consumerOp, memoryType] : tensorToBeMap) {
            if (memoryType == MEM_L0C) {
                inserter.UpdateTensorTobeMap(oOperand, *consumerOp, MEM_DEVICE_DDR);
            }
        }
    }
}
```

**维度倍数检查示例**:
```
输入: [16, 32]
输出: [32, 64]

检查: 32 % 16 == 0 && 64 % 32 == 0 → 通过

输入: [16, 32]
输出: [24, 64]

检查: 24 % 16 != 0 → 不通过 → 降级 DDR
```

---

### 3.21 ProcessLargeTileToSamllTile

**位置**: Line 668-692

**功能**: 处理 Cube 级联场景下大 Tile 搬运到小 Tile 的约束。

**场景**:
```
Cube 级联: 大 Tile 需要拆分给多个消费者

L1 (大) ──→ View ──→ L0C (小)
            ↓
        多个小 Tile
```

**逻辑**:

```cpp
for (auto& op : function.Operations()) {
    if (op.GetOpcode() != OP_VIEW) continue;
    
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(...);
    MemoryType attrToType = viewOpAttribute->GetTo();
    
    // 只处理 To=L1 的 View
    if (attrToType != MEM_L1) continue;
    
    auto iOperand = op.GetIOperands().front();
    auto oOperand = op.GetOOperands().front();
    
    // 情况 1: L0C → 小 Tile
    if (iOperand->GetMemoryTypeOriginal() == MEM_L0C &&
        !IsDimMultiple(iOperand->GetShape(), oOperand->GetShape())) {
        
        // 输入维度不是输出维度的倍数 → 降级 DDR
        inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
        continue;
    }
    
    // 情况 2: UB → 小 Tile
    if (iOperand->GetMemoryTypeOriginal() == MEM_UB &&
        oOperand->shape != iOperand->shape) {
        
        // 形状不匹配 → 降级 DDR
        inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
        continue;
    }
}
```

**约束说明**:
```
小 → 大 (ProcesSmallTileToLargeTile):
  输出维度必须是输入维度的倍数
  例: [16, 32] → [32, 64] ✓

大 → 小 (ProcessLargeTileToSamllTile):
  输入维度必须是输出维度的倍数
  例: [32, 64] → [16, 32] ✓

违反约束时降级 DDR，避免硬件不支持的操作。
```

---

### 3.22 IsDimMultiple

**位置**: Line 698-709

**功能**: 检查两个 Shape 是否满足整倍数关系。

**实现**:

```cpp
bool IsDimMultiple(const Shape& shape1, const Shape& shape2) {
    // 维度数量必须相同
    if (shape1.size() != shape2.size()) {
        return false;
    }
    
    // 每个维度必须是正整倍数
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] <= 0 || shape2[i] <= 0) {
            return false;
        }
        if (shape1[i] % shape2[i] != 0) {
            return false;
        }
    }
    
    return true;
}
```

**示例**:
```
shape1 = [32, 64, 128]
shape2 = [16, 32, 64]

检查:
  32 % 16 == 0 ✓
  64 % 32 == 0 ✓
  128 % 64 == 0 ✓
  
结果: true
```

---

## 四、核心数据结构

### 4.1 内存类型 (MemoryType)

```cpp
enum MemoryType {
    MEM_DEVICE_DDR,    // 全局内存 (大容量，慢速)
    MEM_L1,            // L1 Buffer (中等容量，中速)
    MEM_UB,            // Unified Buffer (小容量，快速)
    MEM_L0A,           // Cube 输入 Buffer A (矩阵 A)
    MEM_L0B,           // Cube 输入 Buffer B (矩阵 B)
    MEM_L0C,           // Cube 输出 Buffer C (矩阵乘结果)
    MEM_UNKNOWN,       // 未定义
};
```

### 4.2 张量内存类型属性

```cpp
class LogicalTensor {
    MemoryType original_;  // 生产者决定的内存类型
    MemoryType tobe_;      // 消费者需要的内存类型
    
    // tobeMap: 张量对不同消费者的内存类型需求
    // map<Operation*, MemoryType> tobeMap_;
};
```

### 4.3 关键数据流

```
┌─────────────────────────────────────────────────────────────┐
│                      内存类型推导流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Opcode 定义 (opcode.cpp)                                │
│     └─→ 输入/输出的默认内存类型                              │
│                                                             │
│  2. 生产者设置                                              │
│     └─→ tensor.original = producer.output.memType          │
│                                                             │
│  3. 消费者更新 tobeMap                                      │
│     └─→ tensor.tobeMap[consumer] = consumer.input.memType  │
│                                                             │
│  4. 冲突检测                                                │
│     └─→ if (original != tobe) → 需要插入 Convert           │
│                                                             │
│  5. 特殊处理                                                │
│     └─→ View/Assemble/Reshape/NOP 等                        │
│                                                             │
│  6. 约束检查                                                │
│     └─→ 对齐、容量、维度倍数                                │
│                                                             │
│  7. 插入 Convert                                            │
│     └─→ inserter.DoInsertion()                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、关键约束总结

### 5.1 对齐约束

| 场景 | 约束 | 违反后果 |
|------|------|----------|
| Assemble 输出 | 32 字节对齐 | 降级 DDR |
| View 偏移 | 32 字节对齐 | 输入降级 DDR |

### 5.2 容量约束

| 内存类型 | 阈值 | 超限后果 |
|----------|------|----------|
| UB | UB_LIMIT × UB_THRESHOLD | 降级 DDR |
| L1 | L1_LIMIT × L1_THRESHOLD | 降级 DDR |

### 5.3 维度约束

| 场景 | 约束 |
|------|------|
| 小→大 Tile | 输出维度 = k × 输入维度 (k 为正整数) |
| 大→小 Tile | 输入维度 = k × 输出维度 (k 为正整数) |

### 5.4 特殊通路

| 通路 | 条件 |
|------|------|
| L0C→L1 | FitL0C2L1() 返回 true，Assemble 消费者需要 L1 |

---

## 六、函数调用关系图

```
RunOnFunction (主入口)
│
├── PreCheck
│
├── SetIncastOutcastMemtype
│
├── for each Operation:
│   └── RunOnOperation
│       ├── ProcessAmulBInput (if MATMUL)
│       ├── ProcessViewwithSpecificMem (if VIEW)
│       └── ProcessAssemblewithSpecificMem (if ASSEMBLE)
│
├── for each Operation:
│   └── AssignMoveOp
│       ├── AssignMoveOpForAssemble
│       │   └── CalcLineOffset
│       └── AssignMoveOpForView
│
├── AssignMemUnknown
│
├── for each Operation:
│   └── AssignSpecialOpMemtype
│       ├── AssignOpReshapeMemtype
│       │   └── AssignMemtypeForSplitReshape
│       ├── AssignOpViewTypeMemtype
│       ├── AssignOpNopMemtype
│       └── UpdateOverSizedLocalBuffer
│
├── ProcesSmallTileToLargeTile
│   └── IsDimMultiple
│
├── ProcessLargeTileToSamllTile
│   └── IsDimMultiple
│
├── DoInsertion
│
└── PostCheck
```

---

## 七、总结

`assign_memory_type.cpp` 是 NPU Tile 框架中的核心编译优化 Pass，负责为计算图中的张量分配合适的内存存储类型。

**核心设计原则**:

1. **类型传播**: 从 Opcode 定义和生产者向消费者传播内存类型
2. **冲突检测**: 检测 original 和 tobe 不匹配的情况
3. **约束处理**: 处理对齐、容量、维度等硬件约束
4. **优化插入**: 在必要位置插入 Convert Op 实现内存转换

**关键优化点**:

1. **内存复用**: View/Assemble 的内存类型传递优化
2. **特殊通路**: L0C→L1 直接搬运路径
3. **级联优化**: Cube 级联场景的维度约束处理
4. **降级策略**: 超限情况下安全降级到 DDR

**处理流程总结**:

| 阶段 | 目的 | 关键操作 |
|------|------|----------|
| Phase 1 | 初始化边界类型 | INCAST/OUTCAST 设 DDR |
| Phase 2 | 基础类型推导 | 根据 Opcode 定义设置 |
| Phase 3 | 连接 Op 处理 | View/Assemble 类型传递 |
| Phase 4 | 特殊 Op 处理 | Reshape/NOP/Reduce 等 |
| Phase 5 | 约束检查 | 对齐/容量/维度 |
| Phase 6 | Convert 插入 | 解决类型冲突 |

---

## 八、附录：代码行号索引

| 函数名 | 行号范围 |
|--------|----------|
| SetIncastOutcastMemtype | 30-53 |
| RunOnFunction | 55-92 |
| PreCheck | 94 |
| PostCheck | 96 |
| RunOnOperation | 98-153 |
| ProcessAmulBInput | 154-191 |
| ProcessViewwithSpecificMem | 193-226 |
| ProcessAssemblewithSpecificMem | 228-261 |
| AssignMemtypeForSplitReshape | 263-285 |
| AssignOpReshapeMemtype | 287-302 |
| AssignOpViewTypeMemtype | 304-325 |
| AssignOpNopMemtype | 327-341 |
| AssignSpecialOpMemtype | 343-380 |
| UpdateOverSizedLocalBuffer | 382-398 |
| PrintTensorMem | 400-407 |
| AssignMoveOp | 409-424 |
| CalcLineOffset | 426-443 |
| AssignMoveOpForAssemble | 445-513 |
| AssignMoveOpForView | 514-566 |
| AssignMemUnknown | 568-620 |
| ProcesSmallTileToLargeTile | 622-667 |
| ProcessLargeTileToSamllTile | 668-692 |
| IsDimMultiple | 698-709 |