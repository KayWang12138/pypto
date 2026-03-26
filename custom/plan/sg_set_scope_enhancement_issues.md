# sg_set_scope 功能增强开发问题记录

## 问题发现时间
2026-03-26

## 问题描述

在实现 `sg_set_scope` 功能增强时，发现了多个关键问题，影响了测试的通过和功能的完整性。

---

## 已解决的问题

### 1. JSON 配置文件问题 ✅

**问题描述**：
- `tile_fwk_config_schema.json` 中 `sg_set_scope` 定义为 `"type": "array"` 但**缺少 `items` 字段**
- 导致 ConfigManager 在解析 JSON 时抛出异常：
  ```
  [json.exception.type_error.305] cannot use operator[] with a string argument with null
  ```
- 异常发生在 `config_manager_ng.cpp` 的 `parse_array_type` 函数中

**根本原因**：
`parse_array_type` 函数期望读取 `jData["items"]["type"]`，但 `sg_set_scope` 没有定义 `items`，导致访问失败。

**解决方案**：
修改了两个配置文件：

1. `framework/src/interface/configs/tile_fwk_config.json`:
   - 将 bool 值改为 int（0/1）
   - `[-1, false, false, -1]` → `[-1, 0, 0, -1]`

2. `framework/src/interface/configs/tile_fwk_config_schema.json`:
   - 添加 `items` 定义：
   ```json
   "items": {
       "type": "integer"
   }
   ```

**验证结果**：
- `TestScopeId` 测试通过 ✅

---

### 2. 测试代码输入向量不匹配问题 ✅

**问题描述**：
`GetParallelBranchesGraph` 函数中，`ioperands` 和 `ooperands` 只有 2 个元素，但 `opCodes` 和 `opNames` 有 3 个元素。

**错误代码**：
```cpp
std::vector<std::vector<std::string>> ioperands{{"t1" + br}, {"t2" + br}};
std::vector<std::vector<std::string>> ooperands{{"t2" + br}, {"t3" + br}};
std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_MUL, Opcode::OP_COPY_OUT};  // 3 个元素
std::vector<std::string> opNames{"COPY_IN" + br, "MUL" + br, "COPY_OUT" + br};  // 3 个元素
```

**根本原因**：
`AddOps` 的检查逻辑：
```cpp
if (opcodes.size() != ioperandss.size() || opcodes.size() != ooperandss.size() || opcodes.size() != names.size()) {
    return false;
}
```
由于 3 != 2，导致 `AddOps` 返回 false，操作未添加到图中。

**解决方案**：
添加了第 3 个操作（COPY_OUT）的输入和输出：
```cpp
std::vector<std::vector<std::string>> ioperands{{"t1" + br}, {"t2" + br}, {"t3" + br}};
std::vector<std::vector<std::string>> ooperands{{"t2" + br}, {"t3" + br}, {}};
```

**验证结果**：
- 图构建成功 ✅

---

## 未解决的问题

### 3. TestAllowParallelMerge 测试的 segfault 问题 ❌

**问题描述**：
运行 `TestAllowParallelMerge` 测试时，出现 segmentation fault。

**测试代码**：
```cpp
TEST_F(GraphPartitionTest, TestAllowParallelMerge) {
    ComputationalGraphBuilder G;
    GetParallelBranchesGraph(G);
    Function *function = G.GetFunction();

    const int cycleUB = 100000;
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);

    Operation::ScopeInfo scopeInfo;
    scopeInfo.scopeId = 1;
    scopeInfo.allowParallelMerge = true;
    scopeInfo.allowCrossScopeMerge = false;
    scopeInfo.mixId = -1;

    for (int i = 0; i < 3; i++) {
        G.GetOp("COPY_IN" + std::to_string(i))->SetScopeInfo(scopeInfo);
        G.GetOp("MUL" + std::to_string(i))->SetScopeInfo(scopeInfo);
        G.GetOp("COPY_OUT" + std::to_string(i))->SetScopeInfo(scopeInfo);
    }

    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);

    // 检查代码...
}
```

**gdb 调试信息**：
```
Program received signal SIGSEGV, Segmentation fault.
0x0000fffff78a9bd4 in npu::tile_fwk::Operation::Operation(...) 
#0  0x0000fffff78a9bd4 in npu::tile_fwk::Operation::Operation(npu::tile_fwk::Function&, npu::tile_fwk::Opcode, 
   std::vector<std::shared_ptr<npu::tile_fwk::LogicalTensor>, std::allocator<std::shared_ptr<npu::tile_fwk::LogicalTensor> > >, 
   std::vector<std::shared_ptr<npu::tile_fwk::LogicalTensor>, std::allocator<std::shared_ptr<npu::tile_fwk::LogicalTensor> > >, 
   bool, int) () from /mnt/workspace/gitCode/cann/pypto/build/output/lib/libtile_fwk_interface.so
#1  0x0000fffff769f55c in npu::tile_fwk::Function::AddRawOperation(...) 
#2  0x0000aaaaab571540 in npu::tile_fwk::ComputationalGraphBuilder::AddOp(...) 
#3  0x0000aaaaab572730 in npu::tile_fwk::ComputationalGraphBuilder::AddOps(...) 
#4  0x0000aaaaab755a58 in npu::tile_fwk::GetParallelBranchesGraph(...) 
#5  0x0000aaaaab768e3c in npu::tile_fwk::GraphPartitionTest_TestAllowParallelMerge_Test::TestBody() 
```

**segfault 发生位置**：
- 在 `Operation::Operation` 构造函数中
- 说明 `SetScopeInfo` 调用本身没有问题
- 问题可能在于构造函数内部

**可能原因分析**：

1. **ScopeInfo 成员初始化问题**：
   - `ScopeInfo` 结构体中的 bool 成员可能未正确初始化
   - 需要检查 `operation.h` 中的 `ScopeInfo` 定义

2. **内存访问问题**：
   - `GetOp("COPY_IN" + std::to_string(i))` 可能返回 nullptr
   - 需要在调用 `SetScopeInfo` 之前检查返回值

3. **Operation 对象生命周期问题**：
   - `GetOp` 返回的 Operation 指针可能指向已销毁的对象
   - 需要检查 Operation 的生命周期管理

4. **多次调用 `SetScopeInfo` 的问题**：
   - 在循环中多次调用 `SetScopeInfo` 可能导致内部状态不一致
   - 需要检查 `SetScopeInfo` 方法的实现

**调试尝试**：

1. **禁用 `SetScopeInfo` 调用**：
   - 注释掉测试代码中的 `SetScopeInfo` 调用后，测试通过 ✅
   - 说明问题确实在 `SetScopeInfo` 调用上

2. **禁用 `UpdateScopeId` 中的 `SetScopeInfo`**：
   - 在 `UpdateScopeId` 中注释掉 `SetScopeInfo` 调用，测试仍然 segfault
   - 说明问题不在这里，而在测试代码的 `SetScopeInfo` 调用中

3. **简化测试代码**：
   - 只测试第一个分支，仍然 segfault
   - 说明问题与循环次数无关

**下一步调试方向**：

1. 在 `Operation::SetScopeInfo` 方法中添加断点，查看调用栈
2. 检查 `ScopeInfo` 结构体的内存布局
3. 使用 AddressSanitizer 运行测试，定位内存访问错误
4. 检查 `Operation` 对象的构造和析构过程

---

### 4. 测试文件语法问题 ⚠️

**问题描述**：
在添加 `TestAllowParallelMerge` 测试时，出现了多次语法错误。

**错误示例**：
```
../framework/tests/ut/passes/src/test_graph_partition.cpp:811:1: error: expected '}' at end of input
  811 | } // namespace npu
      | ^
../framework/tests/ut/passes/src/test_graph_partition.cpp:31:20: note: to match this '{'
   31 | namespace tile_fwk {
      |                    ^
```

**根本原因**：
- 文件末尾缺少正确的大括号闭合
- 多次编辑导致代码结构混乱

**解决方案**：
- 需要仔细检查整个测试文件的语法结构
- 确保所有大括号正确闭合
- 建议使用代码格式化工具检查语法

---

## 相关代码文件

### 修改的文件
1. `framework/src/interface/configs/tile_fwk_config.json`
2. `framework/src/interface/configs/tile_fwk_config_schema.json`

### 需要检查的文件
1. `framework/src/interface/operation/operation.h`
   - `ScopeInfo` 结构体定义
   - `SetScopeInfo` 方法实现

2. `framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp`
   - `UpdateScopeId` 函数
   - `BuildSuperNodeGraph` 函数

3. `framework/tests/ut/passes/src/test_graph_partition.cpp`
   - `TestAllowParallelMerge` 测试
   - `GetParallelBranchesGraph` 辅助函数

---

## 测试验证命令

### 编译验证
```bash
python3 build_ci.py --generator Ninja
```

### 运行 TestScopeId
```bash
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestScopeId
```

### 运行 TestAllowParallelMerge（当前失败）
```bash
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestAllowParallelMerge
```

---

## 总结

### 已完成
- ✅ 修复 JSON 配置文件问题
- ✅ 修复测试代码输入向量不匹配问题
- ✅ `TestScopeId` 测试通过

### 待完成
- ❌ 解决 `TestAllowParallelMerge` 的 segfault 问题
- ⚠️ 修复测试文件语法问题
- ⚠️ 完善并行分支合并逻辑
- ⚠️ 完善 `allowCrossScopeMerge` 逻辑

### 建议
1. 先解决 segfault 问题，确保基本的 `SetScopeInfo` 功能正常工作
2. 然后逐步添加并行分支合并和跨 scope 合并的逻辑
3. 使用更严格的内存检查工具（如 AddressSanitizer）进行调试
4. 添加更多的单元测试，覆盖各种边界情况
