# sg_set_scope 合图实现机制

## 概述

`sg_set_scope` 是 PyPTO 中用于手动控制计算图合并的 Pass 选项。通过给 operation 打上 scopeId 标签，强制将相同 scopeId 的节点合并成一个子图。

本文档详细分析了 `sg_set_scope` 的完整实现机制，包括数据传递、合图策略和约束检查。

---

## 数据结构

### Operation 类中的 Scope 信息

```cpp
// framework/src/interface/operation/operation.h
class Operation {
    int scopeId_{-1};  // 单个 scope ID（commit之前）

    void SetScopeId(int scopeId) {scopeId_ = scopeId; };
    int GetScopeId() const { return scopeId_; };
};
```

### NodeGraphInfo 中的 Scope 信息

```cpp
// framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp
class NodeGraphInfo {
    std::vector<int32_t> nodeScope_;  // 每个 supernode 的 scopeId

    // 在 Build 方法中保存 scope 信息
    for (size_t nodeIdx = 0; nodeIdx < node2Op_.size(); nodeIdx++) {
        for (size_t opNodeIdx = 0; opNodeIdx < node2Op_[nodeIdx].size(); opNodeIdx++) {
            int32_t opIdx = node2Op_[nodeIdx][opNodeIdx];
            int32_t scopeId = operationGraphInfo->opList_[opIdx]->GetScopeId();
            if (scopeId != -1) {
                nodeScope_[nodeIdx] = scopeId;
            }
        }
    }
};
```

### SubGraph 类中的 Scope 信息

```cpp
// framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.h
class SubGraph {
    int32_t scopeId_{-1};  // 子图的 scopeId
};
```

---

## 完整工作流程

### 1. Python 层设置 Pass Option

```python
# python/pypto/config.py
pypto.set_pass_options(sg_set_scope=48)
```

### 2. Pass Option 传递到 Operation

```cpp
// framework/src/interface/function/function.cpp:1492
Operation &Function::AddOperation(...)
{
    auto &op = operations_.emplace_back(
        std::make_shared<Operation>(*this, opCode, iOperands, oOperands, updateTensorMap)
    );
    opPosition_.emplace(op.get(), operations_.size() - 1);

    // 🔑 将当前 Pass Option 的 SG_SET_SCOPE 值设置到新创建的 operation
    operations_.back()->SetScopeId(config::GetPassOption<int>(SG_SET_SCOPE));

    return *operations_.back();
}
```

### 3. Pass Option 传递到下一层（ExpandFunction）

```cpp
// framework/src/passes/tensor_graph_pass/expand_function.cpp
Status ExpandFunction::ExpandOperation(Function &function, Operation &op) const
{
    int scopeIdx = op.GetScopeId();

    if (scopeIdx >= 0) {
        // CV 分离平台检查：禁止将 Cube 和 Vector 操作混合到同一个 scope
        scopeMap_[scopeIdx].insert(op.GetCoreType());
        if (!GraphUtils::IsCVMixPlatform() &&
            scopeMap_[scopeIdx].find(CoreType::AIC) != scopeMap_[scopeIdx].end() &&
            scopeMap_[scopeIdx].find(CoreType::AIV) != scopeMap_[scopeIdx].end()) {
            APASS_LOG_ERROR_F(Elements::Function,
                "Cannot mix cube and vector op on a CV seperate platform in function: %s, please check your setting: sg_set_scope=%d",
                function.GetRawName().c_str(), scopeIdx);
            return FAILED;
        }
    }

    // 🔑 将 scope 传递到下一层 Pass
    config::SetPassOption(SG_SET_SCOPE, scopeIdx);
    ExpandOperationInto(function, op.GetTileShape(), op.GetOpcode(), op.GetIOperands(), op.GetOOperands(), op);
    config::SetPassOption(SG_SET_SCOPE, -1);  // 重置为 -1

    return SUCCESS;
}
```

### 4. Graph Partition 阶段的合图

#### 阶段 4.1：Supernode 构建（BuildSuperNodeGraph）

**作用**：将相同 scope 的 operation 合并成 supernode

```cpp
// framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp
Status SuperNodeGraphBuilder::BuildSuperNodeGraph()
{
    std::vector<Operation*> &opList = operationInfo_->opList_;
    std::vector<std::pair<int32_t, int32_t>> mergePair;

    // 更新 ASSEMBLE 和 VIEW 操作的 scope 信息
    UpdateScopeId(opList);

    // 🔑 核心：基于 scope 合并 operation
    for (size_t i = 0; i < opList.size(); i++) {
        auto targetScope = opList[i]->GetScopeId();
        if (targetScope == -1) {
            continue;  // 没有 scope 的跳过
        }

        // 将 scope 相同的前后节点合并
        for (auto outputNode : operationInfo_->outGraph_[i]) {
            if (opList[outputNode]->GetScopeId() == targetScope) {
                mergePair.emplace_back(outputNode, i);
            }
        }
    }

    // 其他类型的合图（CONVERT, COPYIN, COPYOUT, MULACC 等）
    // ...
}
```

**Scope 信息更新**：

```cpp
// framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp
inline void UpdateScopeId(std::vector<Operation*> &opList)
{
    for (size_t i = 0; i < opList.size(); i++) {
        int targetScope = opList[i]->GetScopeId();
        if (targetScope == DEFAULT_SCOPE_ID) {
            continue;
        }

        // 将 scope 传播到相邻的 ASSEMBLE 和 VIEW 操作
        for (auto &consumer : opList[i]->ConsumerOps()) {
            if (consumer->GetScopeId() == -1 && consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
                consumer->SetScopeId(targetScope);
            }
        }
        for (auto &producer : opList[i]->ProducerOps()) {
            if (producer->GetScopeId() == -1 && producer->GetOpcode() == Opcode::OP_VIEW) {
                producer->SetScopeId(targetScope);
            }
        }
    }
}
```

#### 阶段 4.2：Supernode 构建（NodeGraphInfo::Build）

```cpp
// framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp
Status NodeGraphInfo::Build(
    const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
    const std::vector<std::pair<int32_t, int32_t>> &mergePair,
    bool markIsCube)
{
    // ... 并查集处理 mergePair ...

    op2Node_.resize(opList.size());
    nodeCycles_.resize(opList.size());
    std::vector<int32_t> nodeScopeTmp(node2Op_.size(), -1);
    nodeScope_.swap(nodeScopeTmp);

    // 🔑 保存 scope 信息到 supernode
    for (size_t nodeIdx = 0; nodeIdx < node2Op_.size(); nodeIdx++) {
        nodeCycles_[nodeIdx] = 0;
        for (size_t opNodeIdx = 0; opNodeIdx < node2Op_[nodeIdx].size(); opNodeIdx++) {
            int32_t opIdx = node2Op_[nodeIdx][opNodeIdx];
            op2Node_[opIdx] = nodeIdx;
            nodeCycles_[nodeIdx] += operationGraphInfo->opList_[opIdx]->GetLatency();

            // 保存 scope 信息
            int32_t scopeId = operationGraphInfo->opList_[opIdx]->GetScopeId();
            if (scopeId != -1) {
                nodeScope_[nodeIdx] = scopeId;
            }
        }
    }

    BuildInOutGraph(operationGraphInfo, markIsCube);
    return SUCCESS;
}
```

#### 阶段 4.3：Isomorphism Graph 扩展（IsLegalIsoGraphExtender）

**作用**：检查扩展候选节点时，scope 必须相同

```cpp
// framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.cpp
bool IsomorphismGraphGroup::IsLegalIsoGraphExtender(
    std::vector<int32_t> &expandCandidate,
    std::unordered_set<int32_t> &currentNodeSet,
    std::vector<int32_t> &idxInLinkNum,
    int32_t pgUpperBound)
{
    // ... 其他检查 ...

    // 🔑 关键检查：scope 必须相同才能扩展
    for (size_t i = 0; i < expandCandidate.size(); i++) {
        int origScopeId = isoGraphs_[i]->scopeId_;
        int mergeScopeId = superNodeInfo_->nodeScope_[expandCandidate[i]];

        if (origScopeId != mergeScopeId) {
            APASS_LOG_INFO_F(Elements::Operation,
                "Cannot merge supernodes with different scopeId %d and %d.",
                origScopeId, mergeScopeId);
            return false;  // ❌ scope 不同，不允许扩展
        }
    }

    // ... 检查 coreType 兼容性 ...
}
```

#### 阶段 4.4：SubGraph 合并（SuitableForMergeCheck）

**作用**：如果 subgraph 有 scope，则不允许与其他 subgraph 合并

```cpp
// framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.cpp
bool IsoPartitioner::SuitableForMergeCheck(
    int32_t currColor, int32_t mergeColor, bool nonIsoGraphsMerge) const
{
    // 🔑 关键检查：如果任意 group 有 scopeId，则不允许合并
    for (auto graphPtr : isoSubGroups_[currColor]->isoGraphs_) {
        if (graphPtr->scopeId_ != -1) {
            return false;  // ❌ 有 scope，不能与其他 group 合并
        }
    }
    for (auto graphPtr : isoSubGroups_[mergeColor]->isoGraphs_) {
        if (graphPtr->scopeId_ != -1) {
            return false;  // ❌ 有 scope，不能与其他 group 合并
        }
    }

    // ... coreType、cycle 检查 ...
}
```

---

## 数据流图

```
Python层              C++层 (Pass Option层)        C++层 (Operation层)         C++层 (Graph Partition层)
    │                          │                          │                          │
    │  set_pass_options()      │                          │                          │
    │  sg_set_scope=48       │                          │                          │
    ├──────────────────────▶│  SG_SET_SCOPE = 48        │                          │
    │                          │                          │                          │
    │                          │  GetPassOption<int>()    │                          │
    │                          ├──────────────────────────▶│  op.SetScopeId(48)        │
    │                          │                          │                          │
    │                          │                          │  op.GetScopeId()           │
    │                          │                          ├──────────────────────────▶│
    │                          │  SetPassOption(48)      │                          │
    │                          │◀──────────────────────────┤                          │
    │                          │                          │                          │
    │                          │                          │     Supernode构建         │
    │                          │                          │     (相同scope合并)          │
    │                          │                          │                          │
    │                          │                          │     Isomorphism扩展       │
    │                          │                          │     (scope必须相同)          │
    │                          │                          │                          │
    │                          │                          │     SubGraph合并         │
    │                          │                          │     (有scope则不合并)        │
```

---

## 三阶段合图策略

### 阶段 1：Supernode 构建（BuildSuperNodeGraph）

**目标**：将相同 scope 的 operation 合并成 supernode

**机制**：
1. 遍历所有 operation
2. 对于有 scope 的 operation，检查其后继节点
3. 如果后继节点有相同的 scope，则添加到 mergePair
4. 使用并查集将 mergePair 中的节点合并成 supernode

**Scope 传播**：
- 将 scope 传播到相邻的 ASSEMBLE 和 VIEW 操作
- 这些操作本身没有 scope，但会继承上游/下游操作的 scope

### 阶段 2：Isomorphism Graph 扩展（IsLegalIsoGraphExtender）

**目标**：构建同构子图时，检查 scope 约束

**机制**：
1. 尝试扩展同构子图时，检查候选节点的 scope
2. 如果候选节点的 scope 与当前子图的 scope 不同，则拒绝扩展
3. 确保同构子图中的所有 supernode 具有相同的 scope

**约束**：同构子图内的所有 supernode 必须具有相同的 scopeId

### 阶段 3：SubGraph 合并（SuitableForMergeCheck）

**目标**：在最终合并 subgraph 时，检查 scope 约束

**机制**：
1. 检查待合并的两个 subgraph 是否有 scope
2. 如果任意一个 subgraph 有 scope，则不允许合并
3. 确保有 scope 的 subgraph 保持独立，不与其他 subgraph 合并

**约束**：有 scope 的 subgraph 不能与其他 subgraph 合并

---

## 合图示例

### 示例 1：基本合图

假设有如下计算图，设置 `sg_set_scope=1`：

```
    ┌───┐     ┌───┐     ┌───┐     ┌───┐
    │ A │────▶│ B │────▶│ C │────▶│ D │
    │ 1 │     │ 2 │     │ 1 │     │ -1│
    └───┘     └───┘     └───┘     └───┘
```

**合图过程**：

1. **Supernode 阶段**：
   - A 和 C（scope=1）**不会**合并，因为它们之间隔着 B（scope=2）
   - B（scope=2）是独立的 supernode
   - D（scope=-1）没有 scope，不受约束

2. **Isomorphism 阶段**：
   - A 和 C scope 不同，不能形成同构子图
   - 每个节点各自形成独立的 subgraph

3. **SubGraph 合并阶段**：
   - A 有 scope，不能与其他 subgraph 合并
   - C 有 scope，不能与其他 subgraph 合并
   - B 有 scope，不能与其他 subgraph 合并

**最终结果**：4 个独立的 subgraph

### 示例 2：串行合图

设置 `sg_set_scope=1`：

```
    ┌───┐     ┌───┐     ┌───┐
    │ A │────▶│ B │────▶│ C │
    │ 1 │     │ 1 │     │ -1│
    └───┘     └───┘     └───┘
```

**合图过程**：

1. **Supernode 阶段**：
   - A 和 B（scope=1）合并成一个 supernode
   - C（scope=-1）不受约束

2. **Isomorphism 阶段**：
   - A 和 B 形成的 supernode（scope=1）与 C（scope=-1）scope 不同
   - 不能形成同构子图

3. **SubGraph 合并阶段**：
   - supernode（scope=1）有 scope，不能与 C 合并

**最终结果**：2 个独立的 subgraph

### 示例 3：Scope 传播

设置 `sg_set_scope=1`：

```
┌───┐     ┌─────────┐     ┌───┐
│ X │────▶│  ASMB   │────▶│ Y │
│ 1 │     │ (view)  │     │ -1│
└───┘     └─────────┘     └───┘
```

**合图过程**：

1. **Scope 传播**（UpdateScopeId）：
   - ASMB 是 ASSEMBLE 操作，scope=-1
   - X 的 scope=1 会传播到 ASMB
   - ASMB 的 scope 变为 1

2. **Supernode 阶段**：
   - X（scope=1）和 ASMB（scope=1）合并成 supernode

3. **最终结果**：
   - supernode（包含 X 和 ASMB）
   - Y 独立

---

## 关键约束

### 1. Scope 相同性约束

**规则**：只有 scopeId 相同的节点才能在一个阶段内合并

**实现位置**：
- Supernode 构建：`BuildSuperNodeGraph()`
- Isomorphism 扩展：`IsLegalIsoGraphExtender()`

**影响**：不同 scope 的节点永远不会被合并到同一个子图

### 2. CV 分离平台约束

**规则**：在 CV 分离平台上，禁止将 Cube 和 Vector 操作合并到同一个 scope

**实现位置**：`ExpandFunction::ExpandOperation()`

```cpp
if (!GraphUtils::IsCVMixPlatform() &&
    scopeMap_[scopeIdx].find(CoreType::AIC) != scopeMap_[scopeIdx].end() &&
    scopeMap_[scopeIdx].find(CoreType::AIV) != scopeMap_[scopeIdx].end()) {
    APASS_LOG_ERROR_F(Elements::Function,
        "Cannot mix cube and vector op on a CV seperate platform in function: %s, please check your setting: sg_set_scope=%d",
        function.GetRawName().c_str(), scopeIdx);
    return FAILED;
}
```

**影响**：如果同一个 scope 同时包含 AIC 和 AIV 操作，则报错

### 3. 子图隔离约束

**规则**：有 scope 的 subgraph 不能与其他 subgraph 合并

**实现位置**：`SuitableForMergeCheck()`

```cpp
for (auto graphPtr : isoSubGroups_[currColor]->isoGraphs_) {
    if (graphPtr->scopeId_ != -1) {
        return false;  // 有 scope，不能与其他 group 合并
    }
}
```

**影响**：确保用户手动指定的 scope 不会被后续的自动合图算法破坏

---

## 使用方法

### Python 层设置

```python
import pypto

# 方式1：设置所有后续 operation 的 scope
pypto.set_pass_options(sg_set_scope=48)

# 在 scope 内部创建 operation
with pypto.set_scope({'sg_set_scope': 48}):
    x = pypto.placeholder(shape=(32, 32), dtype='float32')
    y = x + 1

# 恢复默认 scope（-1，不强制合图）
pypto.set_pass_options(sg_set_scope=-1)
```

### C++ 层设置

```cpp
#include "interface/function/function.h"

// 创建 operation 时，从 Pass Option 获取 scope
Operation &op = function.AddOperation(
    Opcode::OP_ADD,
    {inputTensor},
    {outputTensor}
);

// 或者直接设置
op.SetScopeId(48);
```

---

## 常见问题

### Q1：为什么设置了 sg_set_scope 但没有合图？

**可能原因**：
1. Scope 传播失败：中间有不同 scope 的节点隔断
2. CV 分离平台约束：scope 内同时包含 AIC 和 AIV 操作
3. Cycle 约束：合并后的 subgraph 超过了 cycle 上限

**调试方法**：
```bash
# 查看编译日志
grep "Cannot merge supernodes with different scopeId" compile.log
grep "Cannot mix cube and vector op" compile.log
```

### Q2：如何验证合图效果？

**方法 1：查看子图数量**

```python
import pypto

# 编译后查看子图数量
print(f"Total subgraph count: {function.GetTotalSubGraphCount()}")

# 查看每个 operation 的子图 ID
for op in function.Operations():
    print(f"{op.GetOpMagic()}: subgraph={op.GetSubgraphID()}")
```

**方法 2：查看 TensorGraph Dump**

```python
# 设置 dump 选项
pypto.set_pass_options(
    pass_verify_save_tensor=True,
    pass_verify_save_tensor_dir="./dump"
)

# 编译后查看 TensorGraph 文件
# 文件路径：./dump/XXX_tensorgraph.dot
```

### Q3：sg_set_scope 和其他 Pass Option 的关系

**与其他选项的交互**：

| Pass Option | 优先级 | 说明 |
|------------|---------|------|
| sg_set_scope | 高 | 强制合图，其他选项必须遵守 |
| pg_upper_bound | 中 | 合并后的 cycle 不能超过此值 |
| pg_parallel_num | 中 | 并行合图的阈值 |
| combine_axis | 低 | 组合轴优化，不影响 scope |

**示例**：
```python
# 即使设置了 pg_upper_bound 很小，sg_set_scope 也会强制合图
pypto.set_pass_options(
    sg_set_scope=48,      # 强制合图
    pg_upper_bound=10      # cycle 上限，但会被 sg_set_scope 覆盖
)
```

---

## 性能影响

### 优点

1. **减少内存访问**：合并成子图后，中间结果可以保持在 L0/L1 缓存中
2. **减少 kernel 启动开销**：多个 operation 合成一个 kernel
3. **提高 NPU 利用率**：更好的数据局部性和流水线

### 缺点

1. **限制编译优化**：强制合图可能阻止其他优化
2. **Cycle 激增**：合并后的子图可能超过合理范围
3. **资源竞争**：大子图可能导致寄存器、L0 缓存资源不足

**建议**：
- 仅在明确需要合图的场景使用 sg_set_scope
- 先尝试让编译器自动合图，必要时再手动干预
- 结合 `pg_upper_bound` 限制 cycle 上限

---

## 总结

`sg_set_scope` 通过三阶段合图策略实现手动控制计算图合并：

1. **Supernode 阶段**：将相同 scope 的 operation 合并成 supernode
2. **Isomorphism 阶段**：同构子图扩展时检查 scope 一致性
3. **SubGraph 合并阶段**：有 scope 的 subgraph 保持独立

**核心约束**：
- Scope 相同性：不同 scope 的节点不合并
- CV 分离：禁止混合 Cube 和 Vector 操作
- 子图隔离：有 scope 的 subgraph 不与其他 subgraph 合并

通过这些机制，用户可以精确控制计算图的合并行为，在需要时强制合图以提升性能。
