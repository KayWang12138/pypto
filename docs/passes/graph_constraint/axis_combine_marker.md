# AxisCombineMarker Pass

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

`AxisCombineMarker` 是一个图优化 Pass，用于标记计算图中哪些张量支持**轴组合优化**（Axis Combine Optimization）。

### 核心作用

轴组合优化旨在对广播算子的输入进行维度对齐，以提高 NPU 计算效率。`AxisCombineMarker` 作为优化的前期分析阶段，需要回答：

> 哪些张量的尾轴可以安全地进行轴组合优化？

该 Pass 通过正向和反向的图遍历，为每个张量打上标记，指导后续的 `AxisCombine` Pass 选择合适的优化策略：
- **ENABLE**：张量支持轴组合，使用高效的广播扩展（OP_BRCB）
- **DISABLE**：张量不支持轴组合，使用通用的扩展操作（OP_EXPAND）
- **UNKNOWN**：张量不涉及轴组合优化

### 设计原则

1. **尾轴为 1 是前提条件**：只有尾轴（最后一个维度）为 1 的张量才可能支持轴组合
2. **尾轴不能被修改**：如果某个操作在尾轴上进行了修改（如 Assemble/Expand），则该路径不支持优化
3. **禁止状态冲突**：一旦某个节点被标记为 DISABLE，整条路径都失效
4. ** UNKNOWN 的填充**：正向遍历可能遗漏某些节点，通过反向遍历补全

---

## 执行流程

`AxisCombineMarker::Run()` 方法包含三个主要阶段：

```cpp
void AxisCombineMarker::Run(Function &function) {
    Init(function);      // 阶段1：初始化
    ForwardVisit();      // 阶段2：正向遍历
    BackwardVisit();    // 阶段3：反向遍历
}
```

### 阶段 1：Init - 初始化

**位置**：`axis_combine_marker.cpp:32-51`

**功能**：构建算子依赖图（DAG）

```cpp
- 收集所有算子列表
- 为每个算子分配索引
- 记录前驱和后继关系
- 构建邻接表（opInGraph_, opOutGraph_）
```

### 阶段 2：ForwardVisit - 正向遍历

**位置**：`axis_combine_marker.cpp:244-266`

**功能**：拓扑排序 + 前向状态推导

**算法**：BFS + 拓扑排序

```python
1. 从入度为 0 的算子开始（输入算子）
2. 对每个算子调用 UpdateOpACEnableForward()
3. 将输出算子的入度减 1
4. 入度为 0 时加入队列，继续处理
5. 直到所有算子处理完毕
```

**状态推导规则**：针对不同算子类型，根据输入状态和输出形状推导输出状态（详见下一节）

### 阶段 3：BackwardVisit - 反向遍历

**位置**：`axis_combine_marker.cpp:268-290`

**功能**：拓扑排序 + 反向状态传播

**算法**：BFS + 拓扑排序（从出度为 0 的算子开始）

```python
1. 从出度为 0 的算子开始（输出算子）
2. 对每个算子调用 UpdateOpACEnableBackward()
3. 将输入算子的出度减 1
4. 出度为 0 时加入队列，继续处理
5. 直到所有算子处理完毕
```

**作用**：
- 从输出向输入传播 `DISABLE` 状态
- 避免在无效路径上浪费时间
- 填充正向遍历遗漏的 `UNKNOWN` 节点

---

## 各算子状态更新逻辑

### 状态枚举

```cpp
enum class AxisReorderStatus {
    ENABLE,   // 支持轴组合
    DISABLE,  // 不支持轴组合
    UNKNOWN   // 未知/不涉及轴组合
};
```

---

### 1. CopyIn 算子

**位置**：`axis_combine_marker.cpp:53-66`

**状态推导逻辑**：

| 条件 | 输出状态 |
|------|----------|
| 输出尾轴 ≠ 1 | `UNKNOWN` |
| 输出尾轴 == 输入尾轴 | `ENABLE` |
| 其他 | `DISABLE` |

**说明**：
- CopyIn 将数据从内存复制到计算单元
- 如果输出尾轴不为 1，不涉及轴组合优化
- 如果尾轴保持不变且为 1，支持轴组合

---

### 2. View 算子

**位置**：`axis_combine_marker.cpp:68-90`

**状态推导逻辑**：

| 条件 | 输出状态 |
|------|----------|
| 输入尾轴 ≠ 输出尾轴 | `DISABLE` |
| 输出尾轴 ≠ 1 | `UNKNOWN` |
| 输入已有状态 | 继承输入状态 |
| 输入尾轴=1 && 输出尾轴=1 | 两者都 `ENABLE` |

**说明**：
- View 操作改变张量的视图（如切片、转置）
- 如果尾轴发生变化，不支持轴组合
- 否则继承输入状态

---

### 3. Assemble 算子

**位置**：`axis_combine_marker.cpp:92-109`

**状态推导逻辑**：

| 条件 | 输出状态 |
|------|----------|
| 输入 `ENABLE` 且输入尾轴 ≠ 输出尾轴 | 输入/输出都 `DISABLE` |
| 输入 `ENABLE` 且输入尾轴 == 输出尾轴 | `ENABLE` |
| 输入非 `ENABLE` | `DISABLE` |

**说明**：
- Assemble 操作在特定维度上合并数据
- **关键约束**：如果尾轴发生 assemble，则不支持轴组合（尾轴不能被修改）
- 这是为了保证轴组合优化的正确性

---

### 4. Expand 算子

**位置**：`axis_combine_marker.cpp:111-135`

**状态推导逻辑**：

| 条件 | 输出状态 |
|------|----------|
| 输入 `ENABLE` 且 expand轴 < 尾轴索引 | `ENABLE` |
| 输入 `ENABLE` 且 expand轴在尾轴 | 输入 `DISABLE`，输出 `UNKNOWN` |
| 输入非 `ENABLE` 且输出尾轴 == 1 | `DISABLE` |
| 输入非 `ENABLE` 且输出尾轴 ≠ 1 | `UNKNOWN` |

**说明**：
- Expand 操作在指定维度上广播张量
- 如果广播轴在尾轴，则不支持轴组合（尾轴不能被广播扩展）
- 否则可以保留轴组合能力

---

### 5. Reduce 算子

**位置**：`axis_combine_marker.cpp:137-154`

**状态推导逻辑**：

| 条件 | 输出状态 |
|------|----------|
| reduce轴 < 最后两轴 | 继承输入状态 |
| reduce倒数第二轴 | `DISABLE` |
| reduce尾轴 | `ENABLE` |

**说明**：
- Reduce 操作在指定维度上归约张量
- 如果在倒数第二轴归约，当前不支持轴组合优化
- 如果在尾轴归约，默认可以支持（因为归约后尾轴会消失）

---

### 6. Elementwise / Broadcast 算子

**位置**：`axis_combine_marker.cpp:156-174`

**状态推导逻辑**：

```python
# 正向遍历
if 任一输入为 UNKNOWN 且尾轴为 1:
    设置输入为 ENABLE

if 输出尾轴 ≠ 1:
    输出状态为 UNKNOWN

if 任一输入为 DISABLE:
    输出状态为 DISABLE

其他情况:
    输出状态为 ENABLE
```

**说明**：
- Elementwise/Broadcast 是二元运算
- 如果输出尾轴为 1，且所有输入都支持轴组合，则输出支持
- 如果任一输入不支持，则输出不支持

---

### 反向遍历逻辑

**位置**：`axis_combine_marker.cpp:211-242`

**涵盖算子**：Elementwise、Broadcast、部分 View/Assemble

```python
# 反向遍历
if 输出为 DISABLE:
    所有输入都设为 DISABLE

if 任一输入为 DISABLE:
    所有输入都设为 DISABLE

if 输入为 UNKNOWN:
    输入继承输出状态
```

**说明**：
- 从输出向输入传播 DISABLE 状态
- 避免在无效路径上浪费时间
- 填充正向遍历遗漏的 UNKNOWN 节点

---

## 应用场景

### 与 AxisCombine Pass 的协作

`AxisCombineMarker` 的输出（tensorStatus_）被 `AxisCombine` Pass 使用：

```cpp
// axis_combine.cpp
if (axisCombineMarker.IsTensorEnableAxisCombine(srcTensor)) {
    // 使用 OP_BRCB 进行广播
    brcb.SetOpCode(Opcode::OP_BRCB);
} else {
    // 使用 OP_EXPAND + 设置 EXPANDDIM 属性
    brcb.SetOpCode(Opcode::OP_EXPAND);
    brcb.SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", ...);
}
```

**含义**：
- **ENABLE**：张量支持轴组合，使用高效的广播扩展（OP_BRCB）
- **DISABLE / UNKNOWN**：张量不支持轴组合，使用通用的扩展操作（OP_EXPAND）

---

## 典型示例

### 示例 1：简单的广播加法

```
Input A: [N, 1]
Input B: [N, 8]
  ↓
  C = A + B (Elementwise)
  ↓
Output C: [N, 8]
```

**标记过程**：
1. Input A: 尾轴为 1，初始状态 `ENABLE`
2. Input B: 尾轴为 8，初始状态 `UNKNOWN`
3. Elementwise: 输出尾轴 ≠ 1，状态 `UNKNOWN`
4. 反向遍历: Output `UNKNOWN` 不影响输入

**优化决策**：
- Input A 的状态为 `ENABLE`
- AxisCombine Pass 会使用 `OP_BRCB` 对 Input A 进行广播扩展
- 将维度从 [N, 1] 填充对齐到 [N, 8]

---

### 示例 2：带 Assemble 的场景

```
Input: [N, 1]
  ↓
  Assemble(dim=-1)  # 在尾轴上合并
  ↓
Output: [N, 8]
```

**标记过程**：
1. Input: 尾轴为 1，初始状态 `ENABLE`
2. Assemble: 输出尾轴 (8) ≠ 输入尾轴 (1)
3. 触发规则：尾轴发生 assemble，两者都 `DISABLE`

**优化决策**：
- Input 的状态最终为 `DISABLE`
- AxisCombine Pass 会使用 `OP_EXPAND` 对 Input 进行扩展
- 设置 `EXPANDDIM` 属性，指定扩展维度

---

### 示例 3：Reduce 场景

```
Input: [N, M, 1]
  ↓
  Reduce(dim=-2)  # 在倒数第二轴上归约
  ↓
Output: [N, 1]
```

**标记过程**：
1. Input: 尾轴为 1，初始状态 `ENABLE`
2. Reduce: reduce轴 (dim=-2) = 倒数第二轴
3. 触发规则：reduce倒数第二轴，输出 `DISABLE`

**优化决策**：
- Output 的状态为 `DISABLE`
- 不支持轴组合优化

---

## 约束说明

1. **调用时机**：必须在 `AxisCombine` Pass 之前执行
2. **架构限制**：DAV_3510 架构不支持此优化（`axis_combine.cpp:85`）
3. **性能考虑**：只处理尾轴为 1 的张量，避免不必要的分析
4. **状态一致性**：正向和反向遍历的结果应该一致

---

## 相关文件

- **实现文件**：`framework/src/passes/tile_graph_pass/graph_constraint/axis_combine_marker.cpp`
- **头文件**：`framework/src/passes/tile_graph_pass/graph_constraint/axis_combine_marker.h`
- **协作 Pass**：`framework/src/passes/tile_graph_pass/graph_constraint/axis_combine.cpp`
