# 动态 Cell Match Table 运行时分配改造方案

## 1. 问题背景

当前 `partial update` 的 `cell match table` 在编译期写入 `DevAscendProgram`，并使用静态数组承载。  
对局部动态 tensor（如 `tmp`）来说，`cellShape` 在编译期可能为 `-1`，导致：

- 编译期 `table size` 退化为 1；
- 运行时不同切片落到同一个 cell；
- 发生 producer 覆盖，stitch 依赖关系错误，最终精度异常。

## 2. 目标

1. **局部动态 tensor 的 cell table 不再走编译期静态数组**。
2. **引入运行时 allocator**，在 launch 阶段根据真实 shape 分配并初始化动态 cell table。
3. **GetWorkSpaceSize 可计算并预留动态 cell table 的内存预算**。
4. 与静态 cell match 逻辑融合：静态 tensor 仍复用旧逻辑，动态 tensor 走新路径。

## 3. 设计总览

### 3.1 编译期（Encode 阶段）

- 对 `partialUpdate`：
  - 若 `cellShape` 含动态标记（`<= 0`，当前主要是 `-1`），则：
    - 编译期 `cellMatchRuntimePartialUpdateTable` 大小置 0；
    - 标记 `runtimeDynamicCellMatch=true`；
  - 否则保持原有静态分配逻辑。
- 在 `TensorWorkspaceResult` 中新增：
  - `dynamicCellMatchSlotNum`
  - `maxDynamicCellMatchTableMem`（`SymbolicScalar`）
- 估算策略：
  - `dynamicCellMatchSlotNum`：需要运行时分配表的 partial-update slot 数；
  - `maxDynamicCellMatchTableMem`：保守上界，先按动态 assemble outcast bytes 推导（用于 workspace 预留）。

### 3.2 程序结构（DevAscendProgram / Attribute）

- `DyndevFunctionAttribute` 新增 `maxDynamicCellMatchTableMem`（符号表达式）。
- `DevAscendProgram::memBudget.tensor` 新增：
  - `maxDynamicCellMatchTableMem`
  - `dynamicCellMatchSlotNum`
- `Total()` 将该预算计入 tensor workspace。

### 3.3 运行时分配（DeviceWorkspaceAllocator）

- `MetadataAllocator` 新增 `WsSlotAllocator dynamicCellMatch`。
- 在 `DeviceWorkspaceAllocator::Init(DevStartArgs*)` 初始化阶段：
  - 使用运行时已确定的 `memBudget.tensor.maxDynamicCellMatchTableMem` 与 `dynamicCellMatchSlotNum` 初始化该 allocator。
- 为每个动态 partial-update slot 按需申请一个固定大小 slot，供本 slot 的 cell table 使用。

### 3.4 运行时 cell table 生成（Slot Update 阶段）

- 在 `device_slot_context.cpp::UpdateSlotsForStitch` 的 partial-update 分支中：
  - 若 `runtimeDynamicCellMatch=true`：
    1. 基于运行时表达式求各 producer 对应 raw shape；
    2. 构造运行时 cellShape/stride（动态维退化值不再使用 `-1` 参与索引）；
    3. 计算 table size；
    4. 通过新 allocator 绑定 `cellMatchRuntimePartialUpdateTable` 指针与大小；
    5. 初始化为 `AICORE_TASK_INIT` 后再执行 `CellMatchFillIncastOutcast`。

## 4. 代码改动清单（实施范围）

- `framework/src/interface/function/function.h`
  - `DyndevFunctionAttribute` 增加 `maxDynamicCellMatchTableMem`。
- `framework/src/machine/utils/dynamic/dev_encode_program.h`
  - `memBudget.tensor` 增加动态 cell table 预算字段并纳入 `Total()`。
- `framework/src/machine/utils/dynamic/dev_encode_function_stitch.h`
  - `DevAscendProgramPartialUpdate` 增加运行时动态标记字段。
- `framework/src/machine/utils/dynamic/dev_encode.cpp`
  - 编译期 partial-update 初始化：动态表 size 置 0；
  - 计算并回填 `dynamicCellMatchSlotNum/maxDynamicCellMatchTableMem`。
- `framework/src/machine/utils/dynamic/allocator/allocators.h`
  - `MetadataAllocator` 增加 `WsSlotAllocator dynamicCellMatch`。
- `framework/src/machine/utils/dynamic/dev_workspace.h`
  - 初始化和管理动态 cell table allocator；
  - 提供 runtime table 申请接口。
- `framework/src/machine/device/dynamic/context/device_slot_context.cpp`
  - partial-update 路径增加运行时 table 构建与绑定。
- `framework/src/machine/runtime/device_launcher_binding.h`
  - `GetWorkSpaceSize` 计算时评估并写回 `maxDynamicCellMatchTableMem`。
- `python/src/bindings/runtime.cpp`
  - `KernelBinary::GetWorkspaceSize` 同步评估并写回 `maxDynamicCellMatchTableMem`。

## 5. 风险与防护

- 风险：预算不足导致运行时表过大分配失败。  
  - 防护：运行时校验 `tableSize * sizeof(uint64_t) <= slotByteSize`，不满足则显式报错。
- 风险：静态路径回归。  
  - 防护：仅在 `runtimeDynamicCellMatch=true` 的 partial-update 分支启用新逻辑。

## 6. 验证计划

1. 编译：`pip install -e .`
2. 样例验证：
  - `python issue/minimal_608_embedding_tmp_sum_case.py`
  - `python issue/608.py`
3. 重点观察：
  - `L=128` 下 `d_emb` 精度恢复；
  - 动态 partial-update slot 的 table size > 1；
  - 依赖关系不再发生错误覆盖。

