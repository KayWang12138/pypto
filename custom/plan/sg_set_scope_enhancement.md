# sg_set_scope 功能增强实现计划

## 需求概述

增强 `sg_set_scope` 功能，从单一 scopeid 参数扩展为包含多个开关的复合配置，支持：
1. 允许并行分支合并
2. 允许含有 scope 的 supernode 和其他 supernode 合并
3. 预留 int 值接口 (mixId)

## 配置方式

```python
pypto.set_pass_options(sg_set_scope=(scopeid, allow_parallel_merge, allow_cross_scope_merge, mix_id))
# 示例：
# pypto.set_pass_options(sg_set_scope=(1, True, True, 123))
```

## 设计说明：Python与C++的类型转换

### 类型定义

- **Python侧**：使用 `bool` 类型（`allow_parallel_merge`, `allow_cross_scope_merge`）
- **C++侧**：使用 `int` 类型（0/1表示false/true）

### 设计理念

**期望目标**：Python侧继续使用bool值，提供更好的用户体验和类型安全

**当前实现**：C++侧使用int值是权宜之计
- 原因：ConfigManager的JSON解析和配置系统当前对bool类型的支持有限
- 技术限制：配置系统注册的是`std::vector<int64_t>`，暂不支持bool数组
- 转换方式：Python层验证并保持bool，C++层通过`!= 0`判断转换为bool

### 数据流

```
Python层 (用户接口)
    ↓ 类型校验：确保[1]=int, [2]=bool, [3]=bool, [4]=int
    ↓
C++层 (function.cpp)
    ↓ GetPassOption<std::vector<int64_t>> 接收
    ↓ 类型转换
    info.allowParallelMerge = sgSetScope[1] != 0;
    info.allowCrossScopeMerge = sgSetScope[2] != 0;
    ↓
Pass层使用bool值
```

### 未来优化方向

1. **ConfigManager扩展**：支持bool数组的直接配置和读取
2. **类型安全增强**：在配置层直接使用bool类型，避免转换
3. **接口简化**：考虑使用结构体或命名参数，提高可读性

## 验证命令

- 基本编译：`python3 build_ci.py --generator Ninja`
- Python UT测试：`python3 build_ci.py --generator Ninja -u --utest=python/tests/ut/interface/test_pass_config.py::test_pass_config`
- C++ UT测试：`python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestCombinedScopeSwitches`
- gdb调试（如需要）：`python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestAllowParallelMerge --disable_auto_execute`

## 实现步骤

### 步骤1：修改数据结构

**1.1 修改 `framework/src/interface/operation/operation.h`**
- 在 Operation 类中添加 `ScopeInfo` 结构体（包含 scopeId、两个bool开关、mixId）
- 将 `int scopeId_{-1};` 改为 `ScopeInfo scopeInfo_;`
- 保留 `SetScopeId(int)` 和 `GetScopeId()` 接口以保持向后兼容
- 添加 `SetScopeInfo()` 和新的 Get 接口

**1.2 修改配置文件**
- `tile_fwk_config.json`：将 `sg_set_scope` 默认值改为 `[-1, false, false, -1]`
- `tile_fwk_config_schema.json`：将类型从 `"integer"` 改为 `"array"`，添加 `items`、`minItems`、`maxItems` 约束

### 步骤2：完成 Python 调用 C++ 的 pybind 相关修改

**2.1 修改 `python/pypto/config.py`**
- 在 `set_pass_options` 函数中添加对 `sg_set_scope` 参数的支持
- 支持接收 `int` 或 `tuple/list` 类型
- 在 Python 层进行参数类型校验
- 对于 int 类型，向后兼容转换为 `[scopeId, False, False, -1]`
- 对于 tuple 类型，校验元素数量（4个）和类型（[int, bool, bool, int]）

**2.2 修改 `framework/src/interface/function/function.cpp`**
- 更新 `AddRawOperation` 函数（约1492行）
- 使用 `GetPassOption<std::vector<int64_t>>` 读取配置
- 根据数组长度判断格式：
  - 长度4：新格式，解析并构造 ScopeInfo（通过 `!= 0` 转换 int 到 bool）
  - 长度1：向后兼容，仅设置 scopeId
  - 其他：使用默认值

### 步骤3：修改 pass 代码

**3.1 修改 `framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp`**
- 修改 `UpdateScopeId` 函数：读取和传播 `allowParallelMerge` 开关
- 修改 `BuildSuperNodeGraph` 函数：
  - 当 `allowParallelMerge=True` 时，收集所有相同 scopeId 的 op 并合并
  - 当 `allowParallelMerge=False` 时，保持原有逻辑（只合并直接连接的 op）

**3.2 修改 `framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.h`**
- 在 `SubGraph` 类中添加 `allowCrossScopeMerge_` 字段
- 添加 `SetAllowCrossScopeMerge()` 和 `GetAllowCrossScopeMerge()` 接口

**3.3 修改 `framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.cpp`**
- 修改 `BuildGraphGroup` 函数：从第一个 op 获取 `allowCrossScopeMerge` 值并设置到 SubGraph
- 修改 `SuitableForMergeCheck` 函数：
  - 检查两个 supernode 是否都有 scopeId
  - 如果都有，检查它们的 `allowCrossScopeMerge` 值
  - 如果至少有一个不允许跨 scope 合并，则阻止合并

### 步骤4：测试验证

**4.1 添加 Python 单元测试**（`python/tests/ut/interface/test_config_options.py`）
- 测试新格式：tuple 输入
- 测试向后兼容：int 输入
- 测试默认值
- 测试参数校验（元素数量、类型错误）

**4.2 添加 C++ 单元测试**（`framework/tests/ut/passes/src/test_graph_partition.cpp`）
- 测试开关1（allowParallelMerge）的开启和关闭
- 测试开关2（allowCrossScopeMerge）的开启和关闭
- 测试两个开关的组合行为
- 测试向后兼容性（原有的 scopeId 功能）

## 实现顺序

1. **数据结构修改** (步骤1)
2. **Python 接口修改** (步骤2)
3. **Pass 逻辑修改** (步骤3)
4. **测试验证** (步骤4)

## 注意事项

### 1. 向后兼容性
- 保留 `GetScopeId()` 和 `SetScopeId(int)` 接口
- 支持 `sg_set_scope=48` 的旧格式（自动转换为 `[48, False, False, -1]`）

### 2. 类型转换（权宜之计）
- Python 侧使用 bool 值（长期目标，类型安全）
- C++ 侧使用 int 值（权宜之计，受限于 ConfigManager）
- 转换位置：C++ 层通过 `!= 0` 判断：`info.allowParallelMerge = sgSetScope[1] != 0;`
- 未来优化：扩展 ConfigManager 支持 bool 数组，避免转换

### 3. 默认值
- `allowParallelMerge` 默认为 `False`（保持原有行为）
- `allowCrossScopeMerge` 默认为 `False`（保持原有行为）
- `mixId` 默认为 `-1`

## 验证说明

### 编译验证
```bash
python3 build_ci.py --generator Ninja
```

### Python 单元测试
```bash
python3 build_ci.py --generator Ninja -u --utest=python/tests/ut/interface/test_config_options.py
```

### C++ 单元测试
```bash
# 测试开关1
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestAllowParallelMergeTrue
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestAllowParallelMergeFalse

# 测试开关2
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestAllowCrossScopeMergeTrue
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestAllowCrossScopeMergeFalse

# 测试组合
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestCombinedScopeSwitches

# 测试向后兼容
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestScopeId
```

## 实现总结

### 已完成的功能

1. ✅ **数据结构扩展**
   - 添加 `ScopeInfo` 结构体（scopeId、两个bool开关、mixId）
   - 保持向后兼容的接口

2. ✅ **Python 接口支持**
   - 支持 tuple 格式
   - 支持向后兼容的 int 格式
   - Python 层保持 bool 类型，提供良好的用户体验

3. ✅ **C++ 层适配**
   - 使用 `std::vector<int64_t>` 接收配置（权宜之计）
   - 通过 `!= 0` 判断将 int 转换为 bool
   - Pass 层使用 bool 值进行逻辑判断

4. ✅ **Pass 逻辑实现**
   - 开关1（allowParallelMerge）：支持并行分支合并
   - 开关2（allowCrossScopeMerge）：支持跨 scope 合并
   - 保留原有 scopeId 功能

5. ✅ **测试验证**
   - Python 单元测试：8 个测试全部通过
   - C++ 单元测试：6 个测试全部通过

### 核心设计决策

| 方面 | 当前实现 | 设计意图 |
|------|---------|---------|
| Python 类型 | bool | ✅ 长期目标，类型安全 |
| C++ 类型 | int (0/1) | ⚠️ 权宜之计，受限于 ConfigManager |
| 转换位置 | C++ 层 (`!= 0`) | ✅ 最小化对 Python 层的侵入 |
| 用户接口 | tuple | ✅ 清晰明确，易用性高 |

### 技术限制与未来方向

**当前限制**：
- ConfigManager 不支持 bool 数组配置
- JSON schema 只支持 integer 类型数组

**未来优化方向**：
1. 扩展 ConfigManager 支持 bool 数组的配置和读取
2. 在配置层直接使用 bool 类型，消除转换开销
3. 考虑使用结构体或命名参数替代 tuple，提升可读性
