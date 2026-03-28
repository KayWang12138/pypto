# sg_set_scope 功能增强实现计划

## 需求概述

增强 `sg_set_scope` 功能，从单一 scopeid 参数扩展为包含多个开关的复合配置，支持：
1. 允许并行分支合并
2. 允许含有 scope 的 supernode 和其他 supernode 合并
3. 预留 int 值接口 (mixId)

## 配置方式

用户通过 Python 接口传入 tuple 格式：`(scopeid, allow_parallel_merge, allow_cross_scope_merge, mix_id)`，也支持向后兼容的 int 格式（自动补全为默认开关值）。

## 验证命令

- 基本编译：`python3 build_ci.py --generator Ninja`
- Python UT：`python3 build_ci.py --generator Ninja -u --utest=python/tests/ut/interface/test_config_options.py`
- C++ UT：`python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestCombinedScopeSwitches`

## 实现步骤

### 步骤1：修改数据结构

- 在 Operation 类中添加 `ScopeInfo` 结构体（包含 scopeId、两个 bool 开关、mixId）
- 将原有 `int scopeId_` 成员改为 `ScopeInfo scopeInfo_`
- 保留 `SetScopeId(int)` 和 `GetScopeId()` 接口以保持向后兼容
- 添加 `SetScopeInfo()` 和各字段的 Get 接口
- 修改 `tile_fwk_config.json` 默认值为 4 元素数组，`tile_fwk_config_schema.json` 类型从 integer 改为 array

### 步骤2：完成 Python 调用 C++ 的 pybind 相关修改

- `python/pypto/config.py`：在 `set_pass_options` 中支持 `sg_set_scope` 接收 int 或 tuple/list，Python 层校验元素数量和类型，int 自动向后兼容转换
- `function.cpp` 的 `AddRawOperation`：使用 `GetPassOption<vector<int64_t>>` 读取配置，根据数组长度（4 或 1）判断新旧格式

### 步骤3：修改 pass 代码

- `supernode_graph_builder.cpp`：修改 `UpdateScopeId` 和 `BuildSuperNodeGraph`，当 `allowParallelMerge=True` 时收集相同 scopeId 的 op 合并，`False` 时保持原有逻辑
- `iso_partitioner.h/cpp`：在 SubGraph 中添加 `allowCrossScopeMerge_` 字段，在 `SuitableForMergeCheck` 中检查两个 supernode 的跨 scope 合并开关

### 步骤4：测试验证

- 添加 Python 单元测试：新格式 tuple、向后兼容 int、默认值、参数校验
- 添加 C++ 单元测试：两个开关的开启/关闭、组合行为、向后兼容性

## 实现顺序

1. **数据结构修改** (步骤1)
2. **Python 接口修改** (步骤2)
3. **Pass 逻辑修改** (步骤3)
4. **测试验证** (步骤4)

## 注意事项

### 1. 向后兼容性
- 保留 `GetScopeId()` 和 `SetScopeId(int)` 接口
- 支持旧的单 int 格式（自动补全开关默认值）

### 2. 类型转换（权宜之计）
- Python 侧使用 bool 值（长期目标）
- C++ 侧使用 int 值（受限于 ConfigManager），通过 `!= 0` 转换
- 未来扩展 ConfigManager 支持 bool 数组后可消除转换

### 3. 默认值
- `allowParallelMerge` 默认 False（保持原有行为）
- `allowCrossScopeMerge` 默认 False（保持原有行为）
- `mixId` 默认 -1

## 验证说明

- 编译：`python3 build_ci.py --generator Ninja`
- Python UT：`python3 build_ci.py --generator Ninja -u --utest=python/tests/ut/interface/test_config_options.py`
- C++ UT（开关/组合/兼容性）：`python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.*`

## 阶段二：ScopeInfo 完整传播修复（问题9）— ✅ 已完成

### 根因分析

`Operation` 类缺少 `GetScopeInfo()` 整体获取接口，导致所有 scope 传播点都使用"只取 scopeId → 只设 scopeId"的模式，静默丢弃 `allowParallelMerge`、`allowCrossScopeMerge`、`mixId` 三个字段。问题不局限于单一文件，而是涉及所有需要传播 ScopeInfo 的代码路径。

### 受影响位置（6个文件，7处）

| # | 文件 | 问题描述 |
|---|------|---------|
| 1 | `operation.h` | 缺少 `GetScopeInfo()` 整体获取接口 |
| 2 | `expand_function.cpp` | ExpandOperation 只读 scopeId，且 SetPassOption 传入 int 类型与下游 vector 类型不匹配 |
| 3 | `graph_utils.cpp` (Assemble) | 新建 Assemble op 时只传播 scopeId，丢失开关 |
| 4 | `graph_utils.cpp` (Reshape) | 新建 Reshape op 时只传播 scopeId，丢失开关 |
| 5 | `supernode_graph_builder.cpp` (else分支) | 向邻居传播 scope 时只传 scopeId，丢失开关 |
| 6 | `supernode_graph_builder.cpp` (if分支) | 同上，且直接访问 scopeInfo_ 成员而非通过接口 |
| 7 | `operation.cpp` (CloneOperation) | 克隆 op 时未复制 ScopeInfo，克隆体会获取当前全局配置而非原 op 的 scope |

### 修复思路

核心策略：**添加 `GetScopeInfo()` 接口，全局统一用 `SetScopeInfo(GetScopeInfo())` 替换 `SetScopeId(GetScopeId())`**

1. **添加接口**：在 `operation.h` 中添加 `GetScopeInfo()` 返回 `const ScopeInfo&`，一处添加解决所有传播点
2. **统一传播**：所有 `SetScopeId(src.GetScopeId())` 替换为 `SetScopeInfo(src.GetScopeInfo())`，涉及 graph_utils（2处）和 supernode_graph_builder（2处）
3. **消除直接成员访问**：supernode_graph_builder if 分支中直接访问 `scopeInfo_` 改为通过 `GetScopeInfo()` 接口
4. **修复 ExpandFunction 类型问题**：通过 `GetScopeInfo()` 获取完整结构体后转为 `vector<int64_t>` 再传入 `SetPassOption`，解决 int 与 vector 的类型不匹配；重置时也使用 vector 默认值
5. **修复 CloneOperation**：在 `AddRawOperation` 调用后显式 `SetScopeInfo` 复制原 op 的 ScopeInfo
6. **补全模板实例化**：在 `config_manager_ng.cpp` 中添加 `SetOptionsNg<vector<long>>` 的显式实例化，支持 vector 类型通过 `SetPassOption` 设置

### 修复原则

- 一处添加接口，全局一次性修复，未来新增传播逻辑也不会遗漏字段
- 消除直接访问 `scopeInfo_` 成员的代码，统一通过接口访问
- 最小化改动：不改变任何逻辑，只是把"部分传播"改为"完整传播"

### 验证结果

| 测试套件 | 结果 |
|---------|------|
| GraphPartitionTest（25个用例） | ✅ 全部通过 |
| TestExpandFunctionPass（12个用例） | ✅ 全部通过 |
| FunctionUtilsTest.TestCloneOperation | ✅ 通过 |

注：TestExpandFunction.ExpandFunctionTest 失败是已有的环境问题（JSON 输出目录不存在），与本次修改无关。

---

## 实现总结

### 阶段一：功能增强 ✅ 已完成

1. **数据结构扩展**：添加 `ScopeInfo` 结构体（scopeId、两个bool开关、mixId），保持向后兼容的 Get/Set 接口
2. **Python 接口支持**：支持 tuple 格式输入，向后兼容 int 格式，Python 层保持 bool 类型
3. **C++ 层适配**：通过 `vector<int64_t>` 接收配置（权宜之计），C++ 层 `!= 0` 转换 int 到 bool
4. **Pass 逻辑实现**：开关1（allowParallelMerge）支持并行分支合并；开关2（allowCrossScopeMerge）支持跨 scope 合并
5. **测试验证**：Python UT 8个通过，C++ UT 6个通过

### 阶段二：ScopeInfo 传播修复 ✅ 已完成

1. **添加 `GetScopeInfo()` 接口**：一处添加，全局受益
2. **修复 6 个文件 7 处传播点**：统一用 `SetScopeInfo(GetScopeInfo())` 替换 `SetScopeId(GetScopeId())`
3. **补全模板实例化**：支持 `SetPassOption` 传入 vector 类型
4. **验证通过**：GraphPartitionTest 25/25，TestExpandFunctionPass 12/12，TestCloneOperation 通过

### 核心设计决策

| 方面 | 当前实现 | 设计意图 |
|------|---------|---------|
| Python 类型 | bool | 长期目标，类型安全 |
| C++ 类型 | int (0/1) | 权宜之计，受限于 ConfigManager |
| 转换位置 | C++ 层 (`!= 0`) | 最小化对 Python 层的侵入 |
| 用户接口 | tuple | 清晰明确，易用性高 |
| 传播方式 | `GetScopeInfo()` + `SetScopeInfo()` | 完整传播，不遗漏字段 |

### 技术限制与未来方向

1. 扩展 ConfigManager 支持 bool 数组的配置和读取
2. 在配置层直接使用 bool 类型，消除转换开销
3. 考虑使用结构体或命名参数替代 tuple，提升可读性
