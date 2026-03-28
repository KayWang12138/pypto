# sg_set_scope 功能增强开发问题记录

## 问题发现时间
2026-03-26

---

## 历史问题（已解决）

### 问题1：JSON 配置文件不支持数组格式

**问题**：
- `tile_fwk_config_schema.json` 中 `sg_set_scope` 定义为 `"type": "integer"`
- `tile_fwk_config.json` 中 `sg_set_scope` 为单个整数值 `-1`
- 需要支持数组格式：`[scopeid, allow_parallel_merge, allow_cross_scope_merge, mix_id]`

**解决**：
- ✅ 修改 `tile_fwk_config_schema.json`：将 `sg_set_scope` 类型从 `"integer"` 改为 `"array"`
- ✅ 修改 `tile_fwk_config.json`：将 `sg_set_scope` 从 `-1` 改为 `[-1, 0, 0, -1]`

### 问题2：function.cpp 无法读取数组配置

**问题**：
- `function.cpp` 的 `AddRawOperation` 方法使用 `GetPassOption<int>` 读取单个整数值
- 无法读取数组格式并传递 `allowParallelMerge`、`allowCrossScopeMerge`、`mixId` 参数

**解决**：
- ✅ 使用 `GetPassOption<std::vector<int64_t>>` 读取数组配置
- ✅ 支持 4 元素数组格式：`[scopeId, allowParallelMerge, allowCrossScopeMerge, mixId]`
- ✅ 支持 1 元素数组格式：向后兼容原有 `scopeId` 单值配置

### 问题3：JSON Schema 重复定义

**问题**：初始修改时在 schema 中重复定义了 `items` 字段，导致 JSON 解析错误

**解决**：✅ 删除重复的 `items` 定义，只保留 `"items": {"type": "integer"}` 配合 `minItems` 和 `maxItems` 限制

### 问题4：类型不匹配

**问题**：配置系统注册的是 `std::vector<int64_t>`，但代码中使用 `std::vector<int>`，导致 `std::bad_cast`

**解决**：✅ 修改为使用 `std::vector<int64_t>` 并在赋值时进行类型转换

---

## 当前问题（待解决）

### 问题5：Python 测试运行失败 🔴 P0 ✅ 已解决

**错误现象**（已解决）：
```python
RuntimeError: [json.exception.parse_error.101] parse error at line 120, column 13:
syntax error while parsing object key - unexpected '}'; expected string literal
```

**根本原因**：
- 初始测试使用了错误的命令和方法
- 正确的测试命令为：`python3 build_ci.py --generator Ninja -u --utest=python/tests/ut/interface/test_config_options.py`

**解决步骤**：
1. ✅ 使用正确的build_ci.py命令进行测试
2. ✅ 测试环境会自动卸载旧包并重新编译安装最新版本
3. ✅ 修改 `python/tests/ut/interface/test_config_options.py` 中的 `test_pass_option` 函数
4. ✅ 将期望值从 `48` 改为 `[48, 0, 0, -1]` 以适配新的数组格式

**修复内容**：
- 修改了 `test_pass_option` 函数的断言：
  ```python
  # 修改前
  assert pass_option["sg_set_scope"] == 48
  
  # 修改后
  assert pass_option["sg_set_scope"] == [48, 0, 0, -1]
  ```

**验证结果**：
```
python/tests/ut/interface/test_config_options.py::test_pass_option PASSED
python/tests/ut/interface/test_config_options.py::test_sg_set_scope_new_format PASSED
======================= 8 passed, 16 warnings in 21.05s ========================
```

**优先级**：P0（阻塞）→ ✅ 已解决

---

### 问题6：配置文件值格式需要确认 🟡 P3 ✅ 已解决

**问题现象**：
- 源文件：`"sg_set_scope": [-1, 0, 0, -1]`（使用 0/1）
- Python 测试期望：`[-1, False, False, -1]`（使用布尔值）

**代码中的转换**：
- `config.py:108`: `processed_sg_set_scope = [sg_set_scope, False, False, -1]`
- `function.cpp:190-191`:
  ```cpp
  info.allowParallelMerge = sgSetScope[1] != 0;
  info.allowCrossScopeMerge = sgSetScope[2] != 0;
  ```

**问题分析**：
- **设计理念**：
  - Python侧使用bool值：长期目标，提供类型安全的用户接口
  - C++侧使用int值（0/1）：权宜之计，受限于ConfigManager不支持bool数组
  - 转换方式：Python层验证并保持bool，C++层通过`!= 0`判断转换为bool

- **数据流**：
  ```
  Python层: (scopeid, bool, bool, int)
      ↓
  C++层: std::vector<int64_t>
      ↓ 类型转换
  Pass层: bool值
  ```

- **技术限制**：
  - ConfigManager当前只支持`std::vector<int64_t>`，不支持bool数组
  - JSON schema定义`type: "array"`配合`items: {"type": "integer"}`

**解决计划**：
1. ✅ 用户已确认设计：Python侧保持bool，C++侧使用int（权宜之计）
2. ✅ 已在文档中明确说明设计理念
3. ✅ 已实现Python层的bool类型校验
4. ✅ 已实现C++层的int到bool转换（`!= 0`判断）

**未来优化方向**：
1. 扩展ConfigManager支持bool数组的配置和读取
2. 在配置层直接使用bool类型，消除转换开销

**优先级**：P3（需用户确认）→ ✅ 已确认并文档化

---

### 问题7：功能完整性未验证 🟡 P1

**需求功能点**：
1. ✅ 开关1：允许并行分支合并（allowParallelMerge）
2. ✅ 开关2：允许跨 scope 合并（allowCrossScopeMerge）
3. ✅ 预留接口：mixId
4. ✅ 向后兼容：支持旧的 `sg_set_scope=48` 格式

**未验证的点**：
- ❌ 开关1的功能是否按预期工作（并行分支合并）
- ❌ 开关2的功能是否按预期工作（跨 scope 合并）
- ❌ 两个开关同时启用时的交互行为
- ❌ 向后兼容性测试（旧的 int 格式）
- ❌ 边界情况处理（空值、无效值等）

**影响**：
- 不确定功能是否正常工作
- 可能存在未发现的 bug

**解决计划**：
1. 设计端到端测试用例，覆盖所有功能点
2. 使用实际的 PyPTO 代码创建测试场景
3. 验证编译和运行结果
4. 对比不同开关配置下的分区结果

**优先级**：P1（部分阻塞）

---

### 问题8：测试用例执行状态 🟡 P4

**C++ 测试**：
- ✅ TestScopeId - 通过
- ✅ TestAllowParallelMergeTrue - 通过
- ✅ TestAllowParallelMergeFalse - 通过
- ✅ TestAllowCrossScopeMergeTrue - 通过
- ✅ TestAllowCrossScopeMergeFalse - 通过
- ✅ TestCombinedScopeSwitches - 通过

**Python 测试**：
- ✅ test_print_options - 通过
- ✅ test_pass_option - 通过（已修复）
- ✅ test_host_option - 通过
- ✅ test_reset_option - 通过
- ✅ test_operation_option - 通过
- ✅ test_global_option - 通过
- ✅ test_option_map - 通过
- ✅ test_sg_set_scope_new_format - 通过

**测试覆盖不足**：
- 缺少 mixId 的专门测试用例（虽然在新格式测试中有覆盖）
- 缺少向后兼容性的端到端测试
- 缺少错误处理的测试（无效参数、类型错误等）

**影响**：
- 测试覆盖不完整
- 可能遗漏边界情况

**解决计划**：
1. 补充 mixId 的测试用例
2. 添加向后兼容性的端到端测试
3. 添加错误处理的测试用例
4. 提高测试覆盖率

**优先级**：P4（可选）

---

### 问题9：ScopeInfo 传播不完整 — 多处丢失开关配置 🔴 P0

**问题描述**：
`Operation` 类缺少 `GetScopeInfo()` 整体获取接口，导致所有 scope 传播点使用 `GetScopeId()` → `SetScopeId()` 模式，静默丢弃 `allowParallelMerge`、`allowCrossScopeMerge`、`mixId`。

**受影响位置（5个文件，7处）**：

| # | 文件 | 行号 | 当前代码 | 问题 |
|---|------|------|---------|------|
| 1 | `operation.h` | 353 | 无 `GetScopeInfo()` | 缺少整体获取接口 |
| 2 | `expand_function.cpp` | 165,173,175 | `GetScopeId()` + `SetPassOption(int)` | 丢失开关 + 类型不匹配 |
| 3 | `graph_utils.cpp` | 53 | `SetScopeId(originOp->GetScopeId())` | Assemble op 丢失开关 |
| 4 | `graph_utils.cpp` | 65 | `SetScopeId(originOpPtr->GetScopeId())` | Reshape op 丢失开关 |
| 5 | `supernode_graph_builder.cpp` | 630 | `consumer->SetScopeId(targetScope)` | else 分支丢失开关 |
| 6 | `supernode_graph_builder.cpp` | 635 | `producer->SetScopeId(targetScope)` | else 分支丢失开关 |
| 7 | `operation.cpp` | 768 | `CloneOperation` 不复制 scopeInfo_ | 克隆 op 获取全局配置而非原 op scope |

**根本原因**：
- `ScopeInfo` 从单个 `int` 扩展为 4 字段结构体，但只暴露了 `GetScopeId()` 接口
- 所有传播点只能获取 `scopeId`，无法获取完整结构体
- 需要添加 `GetScopeInfo()` 接口，并全局替换传播代码

**修复方案**（详见 `sg_set_scope_enhancement.md` 阶段二）：
1. `operation.h`：添加 `const ScopeInfo &GetScopeInfo() const`
2. `graph_utils.cpp`：`SetScopeId(x.GetScopeId())` → `SetScopeInfo(x.GetScopeInfo())`
3. `supernode_graph_builder.cpp`：else 分支同理替换
4. `expand_function.cpp`：用 `GetScopeInfo()` 转 `vector<int64_t>` 后 `SetPassOption`
5. `operation.cpp`：`CloneOperation` 中添加 `op.SetScopeInfo(scopeInfo_)`

**优先级**：P0（阻塞 - 核心功能失效）→ 🟡 待修复

---

### 问题10：文档更新 🟡 P2

**缺失的文档**：

1. **API 文档更新**
   - Python 层 `set_pass_options()` 的参数说明
   - 新参数 `allow_parallel_merge`、`allow_cross_scope_merge`、`mix_id` 的详细说明

2. **使用示例**
   - 如何使用新的 tuple 格式
   - 两个开关的使用场景和效果
   - 与原有 scopeId 功能的对比

3. **实现机制文档**
   - 更新 `sg_set_scope_merge_mechanism.md`，说明新开关的工作原理
   - 添加新功能的架构图和数据流图

4. **已知限制和注意事项**
   - 两个开关的约束和优先级
   - 性能影响分析
   - 最佳实践建议

**影响**：
- 用户无法正确使用新功能
- 增加学习成本

**解决计划**：
1. 更新 `python/pypto/config.py` 的文档字符串
2. 创建使用示例文件
3. 更新 `sg_set_scope_merge_mechanism.md`
4. 编写开发者指南

**优先级**：P2（不阻塞）

---

### 问题11：代码质量检查 🟢 P5

**需要检查的点**：
- 代码风格一致性
- 内存泄漏风险
- 边界条件处理
- 错误信息完整性
- 日志输出充分性

**影响**：
- 可能存在代码质量问题
- 影响长期维护性

**解决计划**：
1. 运行代码检查工具（如 clang-tidy）
2. 审查所有修改的代码
3. 添加必要的边界检查
4. 改进错误消息和日志

**优先级**：P5（可选）

---

## 解决计划

### 阶段1：修复阻塞问题（P0）

1. **解决问题5：Python 测试运行失败** ✅ 已解决
   - 步骤1：检查已安装的配置文件内容
   - 步骤2：对比源文件和已安装文件的差异
   - 步骤3：定位 JSON 解析错误的根本原因
   - 步骤4：修复问题
   - 步骤5：重新编译和安装
   - 步骤6：验证 Python 测试可以运行

 2. **解决问题9：ScopeInfo 传播不完整（5文件7处）**
    - 步骤1：`operation.h` 添加 `GetScopeInfo()` 接口
    - 步骤2：`graph_utils.cpp` 两处 `SetScopeId` → `SetScopeInfo`
    - 步骤3：`supernode_graph_builder.cpp` 两处 else 分支 `SetScopeId` → `SetScopeInfo`
    - 步骤4：`expand_function.cpp` 用 `GetScopeInfo()` 转 `vector<int64_t>` 后 `SetPassOption`
    - 步骤5：`operation.cpp` `CloneOperation` 添加 `SetScopeInfo`
    - 步骤6：编译 + 运行 UT 验证

### 阶段2：功能验证（P1）

2. **解决问题7：功能完整性验证**
   - 步骤1：设计端到端测试用例
   - 步骤2：实现测试代码
   - 步骤3：验证开关1功能（并行分支合并）
   - 步骤4：验证开关2功能（跨 scope 合并）
   - 步骤5：验证两个开关的交互行为
   - 步骤6：验证向后兼容性

### 阶段3：文档更新（P2）

3. **解决问题9：文档更新**
   - 步骤1：更新 API 文档
   - 步骤2：创建使用示例
   - 步骤3：更新机制文档
   - 步骤4：编写开发者指南

### 阶段4：测试补充（P4）

4. **解决问题8：测试用例补充**
   - 步骤1：添加 mixId 测试用例
   - 步骤2：添加向后兼容性测试
   - 步骤3：添加错误处理测试
   - 步骤4：提高测试覆盖率

### 阶段5：代码质量（P5）

5. **解决问题10：代码质量检查**
   - 步骤1：运行代码检查工具
   - 步骤2：审查代码
   - 步骤3：改进代码质量

---

## 验证结果

### C++ 单元测试
- ✅ TestScopeId (2 ms)
- ✅ TestAllowParallelMergeTrue (0.77 ms)
- ✅ TestAllowParallelMergeFalse (0.77 ms)
- ✅ TestCombinedScopeSwitches (2 ms)
- ✅ TestAllowCrossScopeMergeTrue (2 ms)
- ✅ TestAllowCrossScopeMergeFalse (2 ms)

### Python 单元测试
- ✅ test_config_options.py::test_print_options - 通过
- ✅ test_config_options.py::test_pass_option - 通过（已修复）
- ✅ test_config_options.py::test_host_option - 通过
- ✅ test_config_options.py::test_reset_option - 通过
- ✅ test_config_options.py::test_operation_option - 通过
- ✅ test_config_options.py::test_global_option - 通过
- ✅ test_config_options.py::test_option_map - 通过
- ✅ test_config_options.py::test_sg_set_scope_new_format - 通过

### 端到端功能测试
- ❌ 未执行（待进行）

---

## 修改的文件

### 已修改的文件（历史问题）
1. ✅ `framework/src/interface/configs/tile_fwk_config.json` (第10行)
2. ✅ `framework/src/interface/configs/tile_fwk_config_schema.json` (第110-118行)
3. ✅ `framework/src/interface/function/function.cpp` (第1489-1505行)

### 新增/修改的文件（当前实现）
4. ✅ `framework/src/interface/operation/operation.h` (第171-180行，添加 ScopeInfo 结构体)
5. ✅ `python/pypto/config.py` (第62-121行，支持 tuple 格式)
6. ✅ `framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp` (第608-680行，实现开关1)
7. ✅ `framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.h` (第41-62行，添加 allowCrossScopeMerge_)
8. ✅ `framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.cpp` (第156-193行，第506-573行，实现开关2)
9. ✅ `framework/tests/ut/passes/src/test_graph_partition.cpp` (第764-938行，添加测试用例)
10. ✅ `python/tests/ut/interface/test_config_options.py` (第89-116行，添加测试用例)

### 待修复的文件（ScopeInfo 传播修复）
11. 🔴 `framework/src/interface/operation/operation.h` (第353行附近，**添加 `GetScopeInfo()` 接口**)
12. 🔴 `framework/src/passes/tensor_graph_pass/expand_function.cpp` (第164-177行，**用 `GetScopeInfo()` 替换 `GetScopeId()`，修复类型不匹配**)
13. 🔴 `framework/src/passes/pass_utils/graph_utils.cpp` (第53行、第65行，**`SetScopeId` → `SetScopeInfo`**)
14. 🔴 `framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp` (第630行、第635行，**else 分支 `SetScopeId` → `SetScopeInfo`**)
15. 🔴 `framework/src/interface/operation/operation.cpp` (第768行，**`CloneOperation` 添加 `SetScopeInfo`**)

---

## 问题优先级总结

| 优先级 | 问题 | 影响 | 阻塞 | 状态 |
|--------|------|------|------|------|
| P0 | 问题5: Python 测试失败 | 无法验证 Python 接口 | ✅ 是 | ✅ 已解决 |
| P0 | 问题9: ScopeInfo 传播不完整（5文件7处） | 核心功能失效，开关配置丢失 | ✅ 是 | 🟡 待修复 |
| P1 | 问题7: 功能完整性验证 | 不确定功能是否正常工作 | ⚠️ 部分 | 🟡 待解决 |
| P2 | 问题10: 文档更新 | 用户无法正确使用新功能 | ❌ 否 | 🟡 待解决 |
| P3 | 问题6: 配置格式 | 需要用户确认设计意图 | ❌ 否 | 🟢 已确认 |
| P4 | 问题8: 测试覆盖 | 需要补充测试用例 | ❌ 否 | 🟡 待解决 |
| P5 | 问题11: 代码质量 | 提升代码质量 | ❌ 否 | 🟢 可选 |

---

## 下一步行动

1. ✅ **立即行动**：
   - ✅ 解决问题5（Python 测试失败）- 已完成
    - 🔴 解决问题9（ScopeInfo 传播不完整，5文件7处）- **最高优先级，核心功能失效**
2. **短期行动**：解决问题7（功能完整性验证）
3. **中期行动**：解决问题10（文档更新）
4. **长期行动**：解决问题8（测试补充）和问题11（代码质量）

---

**更新时间**：2026-03-28
**更新人**：AI Assistant
**最后更新**：2026-03-28（问题9升级：从 ExpandFunction 单点问题 → ScopeInfo 传播不完整的全局问题，5文件7处）
