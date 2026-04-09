# sg_set_scope 接口重构与 mixId 自动分配设计文档

---

## 1. 背景与目标

**现状**：`sg_set_scope` 接受 4 元组 `(scopeId, allowParallelMerge, allowCrossScopeMerge, mixId)`，其中第 4 位 mixId 由用户手动配置，但在 `supernode_graph_builder.cpp` 的 `BuildSuperNodeGraph` 中从未被消费使用。

**目标**：
- 将接口简化为 3 元组 `(scopeId, allowParallelMerge, allowCrossScopeMerge)`，删除用户手动配置 mixId 的能力
- 在 `BuildSuperNodeGraph` 中，先构造 supernode，再按 scopeId 检测 aic+aiv 混合情况，自动分配 mixId（全局自增计数器）
- 重构 scope-based 合并逻辑：非 CVMix 平台保持原有 merge pair 方式不变；CVMix 平台改为 `Build()` 后在 supernode 级别处理合并与 mixId 标注

---

## 2. 数据流变更

| 阶段 | 修改前 | 修改后 |
|------|--------|--------|
| Python 调用 | `sg_set_scope=(1, True, True, 123)` | `sg_set_scope=(1, True, True)` |
| Python 内部存储 | 4 元素 list | 3 元素 list |
| C++ config 默认值 | `[-1, 0, 0, -1]` | `[-1, 0, 0]` |
| ScopeInfo 初始化 | 从 config 读 4 字段 | 从 config 读 3 字段，mixId 保持默认 -1 |
| BuildSuperNodeGraph | 不处理 mixId | 非 CVMix：Build 后标注 mixId；CVMix：Build 后按 scopeId 分组合并或标 mixId |
| Operation.mixId | 始终 -1 或用户传入值 | 自动分配：同 scopeId 下 aic+aiv 混合组共享自增 mixId |

---

## 3. 修改清单

### 3.1 Python 接口层

**`python/pypto/config.py`**（第 95-127 行）

- docstring 中 sg_set_scope 的 tuple 描述从 4 元素改为 3 元素，删除 `mix_id` 参数说明
- `int` 分支：`[sg_set_scope, False, False, -1]` → `[sg_set_scope, False, False]`
- `tuple` 分支：长度校验从 `!= 4` 改为 `!= 3`，删除第 4 元素的类型校验（原第 119-120 行）

**`python/tests/ut/interface/test_config_options.py`**（第 89-117 行）

- tuple 测试用例从 `(1, True, True, 123)` 改为 `(1, True, True)`
- int 向后兼容断言从 `[48, False, False, -1]` 改为 `[48, False, False]`
- 默认值断言从 `[-1, False, False, -1]` 改为 `[-1, False, False]`
- 错误校验测试：tuple 元素不足的期望长度从 4 改为 3，删除 mixId 类型错误测试

### 3.2 C++ 配置层

**`framework/src/interface/configs/tile_fwk_config.json`**（第 10 行）

- 默认值 `[-1, 0, 0, -1]` → `[-1, 0, 0]`

**`framework/src/interface/operation/operation.h`**（第 172-191 行、第 366-372 行）

- `ScopeInfo` 结构体：保留 `mixId` 字段（默认 -1），`FromConfig()` 改为只读前 3 个元素
- 新增 `SetMixId(int)` 方法，供 `BuildSuperNodeGraph` 自动分配时调用

**`framework/src/interface/function/function.cpp`**（第 1544-1551 行）

- config size 判断从 `== 4` 改为 `== 3`；建议保留 `== 4` 分支做向后兼容降级

**`framework/src/passes/tensor_graph_pass/expand_function.cpp`**（第 192-200 行）

- 写回 config 的 scopeVec 从 4 元素缩减为 3 元素
- reset 值从 `{-1, 0, 0, -1}` 改为 `{-1, 0, 0}`

### 3.3 核心逻辑：`BuildSuperNodeGraph` 重构

当前 `BuildSuperNodeGraph()` 中 scope-based merge pair 生成（L671-707）在 `superNodeInfo_->Build()` **之前**执行，存在以下问题：

1. 在 op 级别生成 merge pair，由 `Build()` 内的 `MergeSrcToDstIsland()` 处理，该函数会检查 `CoreTypeMergeable()`
2. 非 CVMix 平台上不会出现 AIC+AIV 混合，但 CVMix 平台上 `CoreTypeMergeable({AIC, AIV})` 返回 `false`（当 `useCVMixPartition_=false` 时），导致 scope-based merge 被拒绝

**重构方案**：将 scope-based 合并逻辑从 op 级别（Build 前）移到 supernode 级别（Build 后），并按平台分支处理。

#### 3.3.1 `BuildSuperNodeGraph()` 流程变更

```
修改前：                                修改后：
1. PropagateScopeInfo()                1. PropagateScopeInfo()
2. Scope-based merge pair (L671-707)   2. Scope-based merge pair → 仅非 CVMix 平台执行
3. 标准 combine merge pair (L708-727)  3. 标准 combine merge pair（不变）
4. superNodeInfo_->Build()             4. superNodeInfo_->Build()
5. Auto mixId (L738-770)               5. CVMix 后处理: ProcessScopeForCVMix() → 仅 CVMix 平台执行
                                       6. 非 CVMix 后处理: Auto mixId → 仅非 CVMix 平台执行（原逻辑）
```

#### 3.3.2 非 CVMix 平台（`!GraphUtils::IsCVMixPlatform()`）

L671-707 的 scope-based merge pair 逻辑保持不变，在 `Build()` 前执行。`Build()` 后的 auto mixId 逻辑也保持不变（仅标注，不会实际分配，因为非 CVMix 平台不会出现 AIC+AIV 混合）。

#### 3.3.3 CVMix 平台（`GraphUtils::IsCVMixPlatform()`）

跳过 L671-707 的 scope-based merge pair 生成。在 `Build()` 后新增 `ProcessScopeForCVMix()` 函数处理。

**`ProcessScopeForCVMix()` 详细步骤：**

**Step 1 — 按 scopeId 分组，检测 coreType 混合**

遍历所有 supernode（`nodeScope_[nodeIdx]`），对每个有 scopeId 的 supernode：
- 收集 `map<scopeId, set<OpCoreType>>` — scopeId 下出现的所有 coreType
- 收集 `map<scopeId, bool>` — scopeId 下是否有 `allowParallelMerge=true` 的 op

**Step 2 — 按 scopeId 内 coreType 情况决策**

对每个 scopeId，检查其 coreType 集合：

| scopeId 内 coreType | 决策 |
|---------------------|------|
| 纯 AIC（只有 AIC） | 按 `allowParallelMerge` 规则合并同 scopeId 的 supernode |
| 纯 AIV（只有 AIV） | 按 `allowParallelMerge` 规则合并同 scopeId 的 supernode |
| AIC + AIV 混合 | **不做任何合并**，仅分配 mixId |

**合并规则**（纯 coreType 场景）：
- `allowParallelMerge=true`：同 scopeId 的所有 supernode 全部 union（不看连通性）
- `allowParallelMerge=false`：仅沿 `nodeOutGraph_` 边且目标也是同 scopeId 的 supernode union
- 使用 supernode 级别 disjoint set 实现，不检查 `CoreTypeMergeable()`（同 scopeId 内 coreType 已确认一致）

**Step 3 — 重建 supernode 数据结构**（仅在有合并发生时执行）

按 disjoint set 结果重建：
- 合并 `node2Op_`，重建 `op2Node_` 映射
- 重建 `nodeInGraph_`/`nodeOutGraph_`/`nodeInGraphList_`/`nodeOutGraphList_`
- 重算 `nodeScope_`、`nodeCoreType_`、`nodeCycles_`、`nodeMergeable_`

**Step 4 — 跨 coreType 混合组分配 mixId**

对有 AIC+AIV 混合的 scopeId：
- 使用成员变量计数器 `nextMixId_`（初始 0）分配递增 mixId
- 遍历该 scopeId 下所有 supernode 的所有 operation，通过 `SetMixId()` 写回

#### 3.3.4 头文件与成员变量

**`framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.h`**

- 新增成员变量 `int nextMixId_ = 0;`（已添加）
- 新增方法声明 `Status ProcessScopeForCVMix();`
- 新增 `#include "passes/pass_utils/graph_utils.h"`

**`framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp`**

- L671-707 包裹 `if (!GraphUtils::IsCVMixPlatform())` 守卫
- `Build()` 后添加 `if (GraphUtils::IsCVMixPlatform()) { ProcessScopeForCVMix(); }` 调用
- 原有 auto mixId 逻辑（L738-770）包裹 `if (!GraphUtils::IsCVMixPlatform())` 守卫
- 新增 `ProcessScopeForCVMix()` 函数实现

#### 3.3.5 非 CVMix 与 CVMix 路径对比

| 场景 | 非 CVMix | CVMix |
|------|----------|-------|
| scope-based merge 时机 | `Build()` 前，通过 merge pair | `Build()` 后，supernode 级别 |
| 同 scopeId 纯 AIC | 合并 | 合并 |
| 同 scopeId 纯 AIV | 合并 | 合并 |
| 同 scopeId AIC+AIV 混合 | 不会出现（expand 阶段已报错拦截） | 不合并，仅标 mixId |
| CoreTypeMergeable 检查 | 需要且天然通过 | 不需要（按分组隔离） |

### 3.4 C++ UT（建议新增）

**`framework/tests/ut/passes/src/test_graph_partition.cpp`**

- 现有用例中 `info.mixId = -1` 无需修改（与 ScopeInfo 默认值一致）
- 建议新增测试：构造同一 scopeId 下既有 AIC 又有 AIV op 的图，验证自动 mixId 分配

---

## 4. 不需要修改的文件

| 文件 | 原因 |
|------|------|
| `python/src/bindings/controller.cpp` | pybind11 绑定逐元素转 `int64_t`，不依赖元素个数，无需改动 |
| `framework/src/interface/configs/config_manager_ng.h` | 常量名 `SG_SET_SCOPE` 不变 |
| `framework/src/interface/configs/tile_fwk_config_schema.json` | schema 定义为 integer array，不限制长度 |
| `framework/src/passes/block_graph_pass/mix_subgraph_split.cpp` | LeafFuncAttribute.mixId 体系独立，不受影响 |
| `framework/src/interface/function/function.h` | LeafFuncAttribute 不变 |
| `docs/api/config/pypto-set_pass_options.md` | 仅文档化 int 形式，tuple 形式未文档化，可暂不更新 |

---

## 5. 向后兼容性

- **Python 层**：传入 `int` 形式（如 `sg_set_scope=1`）的行为不变，只是内部存储从 4 元素变为 3 元素
- **C++ 层**：`function.cpp` 中建议保留 `scopeConfig.size() == 4` 的分支作为兼容降级，优先匹配 3 元素格式
- **外部配置文件**：如有 JSON 配置使用 4 元素格式，兼容分支可处理

---

## 6. 风险点

1. **外部配置文件兼容**：用户可能有外部 tile_fwk_config.json 使用 4 元素，`FromConfig` 的兼容分支可覆盖
2. **mixId 传播路径**：当前 mixId 写入 Operation::ScopeInfo 后，需确认下游 pass（如 MixSubgraphSplit）是否需要消费此值——当前分析表明 MixSubgraphSplit 使用独立的 LeafFuncAttribute.mixId，不受影响
3. **nextMixId_ 生命周期**：作为 SuperNodeGraphBuilder 成员变量，每次 BuildSuperNodeGraph 调用之间保持递增，确保不同 scopeId 组不会重复。若 Builder 实例被复用需注意重置
4. **CVMix 下 supernode 合并后的数据一致性**：`ProcessScopeForCVMix()` 中合并 supernode 后需完整重建 `node2Op_`、`op2Node_`、`nodeInGraph_`/`nodeOutGraph_` 等所有相关数据结构，否则下游 pass 会读到不一致的数据
5. **BuildHashValues 依赖**：`BuildSuperNodeGraph` 之后通常还会调用 `BuildHashValues()`，该函数依赖 `superNodeInfo_->node2Op_` 和 `op2Node_`，需确保 `ProcessScopeForCVMix()` 重建后的数据结构能被正确消费
