# 动态 Cell Match / Boundary Outcast 全量复盘报告（汇报版）

> 目标：整理“除日志打印外”的实现改动、每处逻辑、整体流程、问题根因与修复方式。  
> 适用范围：本轮 `tmp` 动态链路精度问题 + `BOUNDARY_OUTCAST` 预算问题 + 后续 `608.py` 运行时异常分析。

---

## 0. 执行摘要

- `tmp` 相关精度问题的根因已明确：`BOUNDARY_OUTCAST slot_bytes` 低估导致越界覆盖动态 cell match 池。
- 根修方向已落地：运行时预算评估 + `SlotMemGuard` 检测 + 动态 cell match 生命周期绑定回收。
- `only.py` / `minimal_608_embedding_tmp_sum_case.py` 精度恢复。
- `608.py` 仍存在独立的运行时异常（AICPU `kernel param invalid`），与最初的越界根因是**不同层级问题**。

---

## 1. 变更索引（不含日志打印）

> 说明：以下用 `Mxx` 编号，后文流程和问题分析会引用这些编号。

### M01 - 元数据分配器新增动态 cell match 池

- 文件：`framework/src/machine/utils/dynamic/allocator/allocators.h`
- 位置：`MetadataAllocator`
- 改动：新增 `WsSlotAllocator dynamicCellMatch`
- 目的：动态 partial cell table 脱离编译期静态数组，改为运行时独立分配。

### M02 - runtime outcast 结构绑定动态表 allocation

- 文件：`framework/src/machine/utils/dynamic/runtime_outcast_tensor.h`
- 位置：`RuntimeOutcastTensor`
- 改动：新增 `WsAllocation dynamicCellMatchAllocation{}`
- 目的：把动态 cell table 生命周期绑到对应 boundary outcast，统一延迟回收。

### M03 - partial slot 编译期初始化重构（静/动分流）

- 文件：`framework/src/machine/utils/dynamic/dev_encode.cpp`
- 位置：`DevAscendProgram::InitPartialUpdateSlot`
- 改动：
  - 所有 `partialUpdate` 先安全初始化（避免未初始化访问）
  - 静态表走原 `cellMatchRuntimePartialUpdateTableList`
  - 动态表保持 `HostAssignDataSize(0,0)`，不占编译期表体
- 目的：消除 ABI/初始化隐患，明确“动态表运行时创建”。

### M04 - 动态 assemble outcast 内存符号计算修正

- 文件：`framework/src/machine/utils/dynamic/dev_encode.cpp`
- 位置：`ProcessAssembleOutcast`
- 改动：引入 symbolic shape 识别（如 `HasSymbolicDim`），确保动态 shape 时走 `GetDynRawTensorSize`
- 目的：避免 `maxDynamicAssembleOutcastMem` 在编译期链路中被低估/置零。

### M05 - memBudget 扩展动态 cell match 预算字段

- 文件：`framework/src/machine/utils/dynamic/dev_encode_program.h`
- 位置：`memBudget.tensor`
- 改动：新增 `maxDynamicCellMatchTableMem`、`dynamicCellMatchSlotNum`
- 目的：让 workspace 总预算包含动态表池。

### M06 - 运行时属性扩展（符号预算透传）

- 文件：`framework/src/interface/function/function.h`
- 位置：`DyndevFunctionAttribute`
- 改动：新增 `SymbolicScalar maxDynamicCellMatchTableMem`
- 目的：把编译期符号预算传递到 runtime 求值。

### M07 - GetWorkspaceSize 运行时预算求值扩展

- 文件：`framework/src/machine/runtime/device_launcher_binding.h`
- 位置：`ExportedOperator::GetWorkSpaceSize`
- 改动：运行时同时评估
  - `maxDynamicAssembleOutcastMem`
  - `maxDynamicCellMatchTableMem`
  并引入 `runtimeTensorUpperBound` 兜底 assemble
- 目的：让 workspace 按当前输入形状动态伸缩，避免 boundary slot 低估。

### M08 - Python runtime 预算求值扩展（KernelBinary 路径）

- 文件：`python/src/bindings/runtime.cpp`
- 位置：`KernelBinary::GetWorkspaceSize`
- 改动：同 M07，对 runtime budget 求值与兜底。
- 目的：覆盖 Python 执行路径，避免预算链路只在一侧生效。

### M09 - workspace 初始化时创建 dynamicCellMatch 池

- 文件：`framework/src/machine/utils/dynamic/dev_workspace.h`
- 位置：`InitTensorAllocators`
- 改动：
  - 按 `dynamicCellMatchSlotNum / maxDynamicCellMatchTableMem` 初始化池
  - 若编译期 slot_num=0，运行时回扫 partialUpdateList 补充统计
  - 若 slot_bytes=0，fallback 到安全下界
- 目的：确保运行时有可用的动态表分配池。

### M10 - Assemble 路径分配动态 cell table（入口 A）

- 文件：`framework/src/machine/utils/dynamic/dev_workspace.h`
- 位置：`AssignOutcastAddresses` -> `TryAllocateDynamicCellMatchForAssembleSlot`
- 改动：
  - 在 assemble slot 首次分配时，计算 runtime table size 并分配
  - 更新 `cellShape/strideShape`
  - 表初始化为 `AICORE_TASK_INIT`
  - 绑定到 `RuntimeOutcastTensor.dynamicCellMatchAllocation`
- 目的：动态 table 生命周期与 assemble outcast 一致。

### M11 - Stitch 更新路径的动态表准备（入口 B，仅校验/补形态）

- 文件：`framework/src/machine/device/dynamic/context/device_slot_context.cpp`
- 位置：`PrepareRuntimeDynamicPartialUpdateTable`
- 改动：
  - 当 partial table 仍空时进行 runtime shape/tableSize 计算
  - 在“只保留 A 分配”策略下，B 路径不再兜底分配，仅报告并返回
- 目的：避免多入口重复分配引起生命周期混乱。

### M12 - 动态表回收

- 文件：`framework/src/machine/utils/dynamic/dev_workspace.h`
- 位置：`TriggerDelayedRecycle`
- 改动：boundary outcast 回收时同步回收 `dynamicCellMatchAllocation`
- 目的：防泄漏、防悬挂引用，保证下个窗口可复用。

### M13 - 运行时安全检测（Boundary Outcast）

- 文件：`framework/src/machine/utils/dynamic/dev_workspace.h`
- 位置：`AssignOutcastAddresses`
- 改动：`SlotMemGuard`：`required > slot_bytes` 立即断言
- 目的：把“静默越界”转成“可定位失败”。

### M14 - 启动期保底策略（动态 slot 但预算为 0）

- 文件：`framework/src/machine/runtime/device_launcher.h`
- 位置：`PrepareDevProgArgs`
- 改动：当动态 slot 存在且 assemble 预算为 0 时，提升到保守下界
- 目的：防止极端情况下 0 预算导致 boundary slot 失真。

### M15 - FillDeviceKernelArgs 接口支持传入动态 workspace 值

- 文件：
  - `framework/src/machine/runtime/device_launcher.h`
  - `framework/src/machine/runtime/device_launcher.cpp`
- 改动：`FillDeviceKernelArgs(..., int64_t dynWorkspaceSize = 0)`
- 目的：允许 launch 参数构建携带本次动态预算。

### M16 - slot 分配策略回归（仅 Assemble 首次分配）

- 文件：`framework/src/machine/utils/dynamic/dev_workspace.h`
- 改动：
  - 取消输出/assemble 冲突时的“强制 assemble 优先”
  - 取消 `AssembleReuse` 场景再分配动态表
- 目的：回归你确认的策略：只在 assemble 首次分配处理 partial 动态表。

---

## 2. Cell Match Table 整体实现流程（当前）

## 2.1 编译期（描述子准备）

1. 构建 `partialUpdateList`（M03）
2. 静态 partial：绑定编译期表体
3. 动态 partial：保留空表（`size=0/data=null`），只保留 `cellMatchTableDesc`
4. 统计动态预算符号（M04/M05/M06）

## 2.2 运行时 workspace 预算

1. `GetWorkspaceSize` 评估动态符号（M07/M08）
2. assemble 预算与输入输出上界取 max（兜底）
3. `memBudget.Total()` 作为本次 workspace 大小

## 2.3 运行时池初始化

1. 初始化 `devTaskBoundaryOutcasts`（slot_bytes = `MaxOutcastMem()`）
2. 初始化 `dynamicCellMatch` 池（M09）

## 2.4 运行时动态表创建

1. 入口 A（主入口）：`AssignOutcastAddresses` 在 assemble 首次分配时创建（M10）
2. 入口 B（辅助）：`UpdateSlotsForStitch` 仅做动态表形态校验/重建检查，不兜底分配（M11）

## 2.5 写表/读表

1. `CellMatchFillIncastOutcast` 写 producer 信息
2. `PartialUpdateStitch` 根据 consumer offset/shape 查表建立依赖

## 2.6 回收

1. outcast 延迟回收触发
2. 同步回收绑定的 dynamicCellMatch allocation（M12）

---

## 3. BOUNDARY_OUTCAST 专题

## 3.1 原始计算方式（修复前）

`slot_bytes` 来源：

1. `devProg->memBudget.tensor.MaxOutcastMem()`
2. `MaxOutcastMem() = max(maxStaticOutcastMem, maxDynamicAssembleOutcastMem)`
3. `InitTensorAllocators` 用该值初始化 `devTaskBoundaryOutcasts`

问题在于：`maxDynamicAssembleOutcastMem` 在部分路径未被正确 runtime 求值/覆盖，导致 `slot_bytes` 偏小（1024/4096）。

## 3.2 实际故障

- 实际需求如 `required=8256`
- slot 仅 1024/4096
- boundary outcast 越界写，覆盖后续 dynamicCellMatch 区域
- 表污染 -> 依赖错乱 -> 精度异常 / NaN

## 3.3 现修复方式

1. 运行时预算评估扩展（M07/M08）
2. 输入输出上界兜底 assemble（M07/M08）
3. `SlotMemGuard` 运行时硬校验（M13）
4. 启动保底避免 0 预算（M14）

---

## 4. 为什么不是 OOM，而是参数非法（608）

- OOM（如 `EL0004`）发生在分配阶段；
- 608 当前是 `AICPU execute kernel param invalid (507018)`，发生在执行/同步阶段；
- 表示分配已过，但某个任务参数组合不合法。

这与 `tmp` 链路精度问题是不同层级故障：

- `tmp` 精度问题：越界覆盖导致数据污染；
- `608` 异常：复杂图路径下 task 参数非法（需单独定位）。

---

## 5. 当前状态与建议

### 5.1 当前状态

- `minimal_608_embedding_tmp_sum_case.py`：精度恢复
- `only.py`：精度恢复
- `608.py`：仍有独立运行时异常（507018）

### 5.2 建议

1. 保持 M13（`SlotMemGuard`）长期启用；
2. 维持“仅 Assemble 首次分配”策略（M16）；
3. 对 `608` 走单独专项：定位 `task_id` 对应参数构造链路，不与 `tmp` 精度问题混合处理。

---

## 6. 汇报可用的“标记位”

- **预算链路标记**：M04 / M07 / M08 / M14
- **分配链路标记**：M09 / M10 / M11 / M16
- **回收链路标记**：M02 / M12
- **防护标记**：M13
- **边界问题闭环**：3.1 -> 3.2 -> 3.3

> 建议汇报时按“问题根因闭环 + 生命周期完整性 + 现网风险（608）”三段讲述。

