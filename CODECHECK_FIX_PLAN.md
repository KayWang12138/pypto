# PR #2198 CodeCheck 修复计划

> PR: https://gitcode.com/cann/pypto/pull/2198
> 分支: newscope1
> CI 最新触发: 2026-04-15 11:09 (comment_id: 168367572)
> CodeCheck 报告: codecheck ❌ FAILED (共 10 个违规)

---

## 违规总览

| # | 规则 | 文件 | 行号 | 描述 | 可自动修复 |
|---|------|------|------|------|-----------|
| 1 | 超大函数[C++] | iso_partitioner.cpp:479 | 479 | `SuitableForMergeCheck()` 59行 > 50行 | 需拆分 |
| 2 | 超大圈复杂度[C++] | supernode_graph_builder.cpp:713 | 713 | `ProcessScopeMerge()` 圈复杂度30 > 20 | 需重构 |
| 3 | 超大深度函数[C++] | supernode_graph_builder.cpp:713 | 713 | `ProcessScopeMerge()` 深度7 > 5 | 需重构 |
| 4 | 超大函数[C++] | supernode_graph_builder.cpp:713 | 713 | `ProcessScopeMerge()` 105行 > 50行 | 需重构 |
| 5 | 重复代码[C++] | test_graph_partition.cpp:860 | 860 | 块(828,838)与(860,870)重复10行 | 可自动 |
| 6 | 冗余代码[C++] | test_graph_partition.cpp:953 | 953 | 冗余代码 | 可自动 |
| 7 | 冗余代码[C++] | test_graph_partition.cpp:954 | 954 | 冗余代码 | 可自动 |
| 8 | 超大函数[C++] | test_graph_partition.cpp:1009 | 1009 | `ConstructGLMAttentionCase()` 98行 > 50行 | 需拆分 |
| 9 | 超大函数[C++] | test_graph_partition.cpp:1122 | 1122 | `TEST_F()` 62行 > 50行 | 需拆分 |
| 10 | 超大深度函数[C++] | controller.cpp:219 | 219 | `ConvertPyDictToCppMap()` 深度6 > 5 | 需重构 |

---

## 文件影响矩阵

| 文件 | 违规数 | 涉及规则 |
|------|--------|---------|
| `framework/src/.../iso_partitioner.cpp` | 1 | 超大函数 |
| `framework/src/.../supernode_graph_builder.cpp` | 3 | 超大函数 + 超大圈复杂度 + 超大深度 |
| `framework/tests/ut/.../test_graph_partition.cpp` | 4 | 重复代码 + 冗余代码 + 超大函数x2 |
| `python/src/bindings/controller.cpp` | 1 | 超大深度 |
| **合计** | **9 (去重)** | |

---

## 详细修复方案

### 第一阶段：可直接修复 (低风险)

#### Fix #6 #7: 冗余代码 — test_graph_partition.cpp:953-954

**现状**: 第 953-954 行是被注释掉的残留代码:
```cpp
// G.GetOp("ABS2")->SetScopeInfo(info2);
// G.GetOp("MUL2")->SetScopeInfo(info2);
```
这两行位于 `TestAllowCrossScopeMergeFalse` 测试的注释块内（被 `/* ... */` 包围，第 882-971 行整体为注释）。

**修复方案**: 删除第 948-954 行的注释残留。由于整个 `GetCrossScopeGraph` 函数和两个测试都被注释包围，确认这些测试暂不使用后，保持 `/* */` 注释块不变即可。但 codecheck 标记的 953-954 行在注释块内部——这可能是 codecheck 误报（在注释内），也可能是 codecheck 扫描到了注释中的代码模式。

**操作**: 需确认 codecheck 是否对注释内容报错。如果是，则清理注释块内的无用代码注释。

---

#### Fix #5: 重复代码 — test_graph_partition.cpp (828-838) vs (860-870)

**现状**: `TestAllowParallelMergeTrue` 和 `TestAllowParallelMergeFalse` 两个测试的初始化代码重复:

重复块 A (828-838):
```cpp
G.GetOp("ABS0")->SetScopeInfo(info);
G.GetOp("ABS1")->SetScopeInfo(info);
G.GetOp("ABS2")->SetScopeInfo(info);
G.GetOp("ABS3")->SetScopeInfo(info);

Function *function = G.GetFunction();
const int cycleUB = 100000;
const int parallelTH = 20;
const int cycleLB = 0;
const int useNodeHash = false;
IsoPartitioner partitioner;
```

重复块 B (860-870): 内容几乎相同（info 属性值不同，但设置代码结构一致）。

**修复方案**: 提取公共辅助函数:
```cpp
struct PartitionParams {
    int cycleUB = 100000;
    int parallelTH = 20;
    int cycleLB = 0;
    bool useNodeHash = false;
};

void SetScopeInfoForOps(
    ComputationalGraphBuilder& G,
    const std::vector<std::string>& opNames,
    const Operation::ScopeInfo& info);

IsoPartitioner CreatePartitioner(const PartitionParams& params = {});
```

---

### 第二阶段：需拆分重构 (中等风险)

#### Fix #1: 超大函数 — iso_partitioner.cpp `SuitableForMergeCheck()` (59行)

**现状**: 函数位于第 479-541 行，包含:
1. lambda `canMergeFrom` (行 483-501) — scope 合并检查
2. coreType 合并检查 (行 507-509)
3. latency 计算 (行 510-520)
4. nonIsoGraphsMerge 分支 (行 522-529)
5. isoGraphs 合并条件 (行 530-540)

**修复方案**: 拆分为 3 个子函数:
```cpp
bool CanMergeScopes(int32_t currColor, int32_t mergeColor);  // 行 483-506
int32_t CalculateMergedLatency(int32_t currColor, int32_t mergeColor);  // 行 510-520
bool CheckIsoMergeConditions(int32_t currColorSize, int32_t mergeColorSize,
                              int32_t currColor, int32_t mergeColor);  // 行 530-540
```
主函数 `SuitableForMergeCheck` 调用上述 3 个子函数，总行数降至 ~20 行。

---

#### Fix #8: 超大函数 — test_graph_partition.cpp `ConstructGLMAttentionCase()` (98行)

**现状**: 函数位于第 1009-1120 行，包含:
1. AddTensors 声明 (行 1011-1041) — ~30 行
2. Group 4: softmax forward ops (行 1048-1059) — ~12 行
3. Group 5: cube ops (行 1061-1096) — ~36 行
4. Group 6: stats update ops (行 1098-1105) — ~8 行
5. Group 7: V2 update ops (行 1107-1119) — ~13 行

**修复方案**: 按逻辑组拆分为 4 个子函数:
```cpp
void AddGLMTensors(ComputationalGraphBuilder& G);                    // tensor 声明
void AddGLMSoftmaxOps(ComputationalGraphBuilder& G);                 // Group 4
void AddGLMCubeOps(ComputationalGraphBuilder& G);                    // Group 5
void AddGLMStatsUpdateOps(ComputationalGraphBuilder& G);             // Group 6+7
```
`ConstructGLMAttentionCase` 调用 4 个子函数，主函数降至 ~5 行。

---

#### Fix #9: 超大函数 — test_graph_partition.cpp `TestGlmAttentionCasePartition` TEST_F (62行)

**现状**: 测试位于第 1122-1192 行，包含:
1. 构建图 + 设置 scope (行 1124-1144) — ~20 行
2. 执行 partition (行 1146-1155) — ~10 行
3. 验证 scope1 subgraph (行 1157-1160) — ~4 行
4. 验证 scope2 subgraph (行 1162-1167) — ~6 行
5. 验证 cube subgraph (行 1169-1182) — ~14 行
6. 验证 v2 subgraph (行 1184-1191) — ~8 行

**修复方案**: 提取验证辅助函数:
```cpp
void VerifyOpsInSameSubgraph(ComputationalGraphBuilder& G,
                               const std::vector<std::string>& opNames,
                               const std::string& label);

void VerifyOpsInDistinctSubgraphs(ComputationalGraphBuilder& G,
                                    const std::vector<std::vector<std::string>>& groups);
```
可将测试函数降至 ~30 行。若仍超标，可进一步将 scope 设置提取到辅助函数。

---

#### Fix #10: 超大深度 — controller.cpp `ConvertPyDictToCppMap()` (深度6)

**现状**: 函数位于第 219-266 行，深度 6 来源:
```
ConvertPyDictToCppMap          // depth 1
  └─ for (auto item : values)  // depth 2
       └─ if/else if chain     // depth 3
            └─ if (py::list)   // depth 4
                 └─ if (int)   // depth 5
                      └─ for   // depth 6
```

最深嵌套在 list + int 分支（行 238-257），因为要逐元素处理混合类型。

**修复方案**: 提取类型转换函数降低深度:
```cpp
npu::tile_fwk::Any ConvertPyValue(const std::string& key, const py::object& value);

npu::tile_fwk::Any ConvertPyList(const std::string& key, const py::list& lst);
```
主函数循环体仅调用 `ConvertPyValue`，深度降至 3。`ConvertPyList` 独立处理 list 内的类型判断。

---

### 第三阶段：重度重构 (高风险)

#### Fix #2 #3 #4: `ProcessScopeMerge()` — 圈复杂度30 + 深度7 + 105行

**现状**: 函数位于第 713-829 行（实际到 ~830 行），是本次修改的核心逻辑:
1. 收集 scope 信息 (行 715-732) — ~18 行
2. Union-Find 初始化 (行 734-744) — ~11 行
3. 遍历 scopeCoreTypes: CV mix 检查 (行 750-763) — 圈复杂度主要贡献
4. allowParallel 分支: union 合并 (行 768-781)
5. 非 parallel 分支: 依赖链合并 (行 782-795)
6. needRebuild: 重建 node 映射 (行 798-817)
7. isCVMix: 设置 mixId (行 819-828)

**修复方案**: 拆分为 4-5 个子函数:
```cpp
// 1. 收集 scope 信息
struct ScopeCollectResult {
    std::unordered_map<int32_t, std::unordered_set<OpCoreType>> scopeCoreTypes;
    std::unordered_map<int32_t, bool> scopeAllowParallel;
    std::unordered_map<int32_t, std::vector<int32_t>> scope2Nodes;
};
ScopeCollectResult CollectScopeInfo(int32_t numNodes);

// 2. CV mix 检查
Status CheckCVMixScopes(
    const std::unordered_map<int32_t, std::unordered_set<OpCoreType>>& scopeCoreTypes,
    std::map<int32_t, int32_t>& scopeToMixId);

// 3. 执行 scope 合并 (Union-Find)
bool MergeScopeNodes(
    const ScopeCollectResult& scopeInfo,
    const std::map<int32_t, int32_t>& scopeToMixId,
    std::vector<int32_t>& snParent);

// 4. 重建 supernode 信息
void RebuildSuperNodes(const std::vector<int32_t>& snParent, int32_t numNodes);

// 5. 设置 CVMix ID
void ApplyCVMixIds(const std::map<int32_t, int32_t>& scopeToMixId,
                    const ScopeCollectResult& scopeInfo);
```

主函数 `ProcessScopeMerge` 调用上述子函数，预计降至 ~20 行。

**风险**: 这是本次 PR 的核心功能逻辑，拆分时需确保 Union-Find 的 `snParent` 状态正确传递。建议逐函数拆分并运行 UT 验证。

---

## 执行顺序建议

| 顺序 | Fix | 文件 | 风险 | 预计改动量 |
|------|-----|------|------|-----------|
| 1 | #6 #7 冗余代码 | test_graph_partition.cpp | 低 | 删除 2 行 |
| 2 | #5 重复代码 | test_graph_partition.cpp | 低 | 新增辅助函数 + 改 2 处调用 |
| 3 | #10 超大深度 | controller.cpp | 中 | 新增 2 个辅助函数 |
| 4 | #1 超大函数 | iso_partitioner.cpp | 中 | 拆分 3 个子函数 |
| 5 | #8 #9 超大函数 | test_graph_partition.cpp | 中 | 拆分测试辅助函数 |
| 6 | #2 #3 #4 复合问题 | supernode_graph_builder.cpp | 高 | 重构核心逻辑 |

---

## 验证计划

每个 Fix 完成后:

运行 UT: `python3 build_ci.py -f=cpp -u=GraphPartitionTest.*` 确保测试通过


