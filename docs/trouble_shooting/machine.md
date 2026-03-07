# MACHINE 组件错误码

- **范围**：F7-F8XXXX
- 本文档说明 MACHINE 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义和场景说明

### 1. SCHEDULER（调度链路，`MachineErrorCategory::SCHEDULER`，F7xxxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------------------------------|----------|
| `PREFETCH_CHECK_FAILED` | **F70001** | `sche.task.pre.prefetch` | SDMA 预取参数非法（`prefetchNum` 超范围）。 |
| `AIC_TASK_WAIT_TIMEOUT` | **F70002** | `sche.task.run.wait_aic` | 等待尾部 AIC 任务完成超时。 |
| `AIV_TASK_WAIT_TIMEOUT` | **F70003** | `sche.task.run.wait_aiv` | 等待尾部 AIV 任务完成超时。 |
| `TAIL_TASK_WAIT_TIMEOUT` | **F70004** | `sche.task.run.sync.wait` | 等待尾部任务完成超时（按核心循环 wait tail task）。 |
| `ALL_AICORE_SYNC_TIMEOUT` | **F70005** | `sche.task.end.sync.timeout` | 全体 AIC/AIV 核同步等待超时，核心未完全停机。 |
| `HANDSHAKE_TIMEOUT` | **F70006** | `sche.task.pre.handshake`、`sche.handshake.error`、`sche.handshake.timeout` | HandShake / HandShakeByGm 超时或失败（含按 core 的详细 dump）。 |
| `READY_QUEUE_OVERFLOW` | **F70007** | `sche.resolve.enqueue` | Ready 队列入队溢出（`tail > capacity`）。 |
| `SIGNAL_QUEUE_OVERFLOW` | **F70008** | `ctrl.task.pre.task.enqueue`、`ctrl.task.pre.task.create` | Signal / 任务队列满、插入任务失败。 |
| `QUEUE_DEQUEUE_WHEN_EMPTY` | **F70009** | `sche.task.end.task.dequeue` | 队列为空时误 Dequeue。 |
| `CORE_TASK_PROCESS_FAILED` | **F70010** | `sche.check.aic.process`、`sche.check.aiv.process`、`sche.task.end.prof.dump` | AIC/AIV 核任务处理失败、prof dump 失败。 |
| `AICPU_TASK_SYNC_TIMEOUT` | **F70011** | `sche.task.end.sync.timeout`（`SyncAicpuTaskFinish`） | AICPU 任务同步等待超时。 |
| `EXCEPTION_RESET_TRIGGERED` | **F70012** | `ctrl.signal.reset.enter`、`ctrl.signal.reset.leave` | 异常场景触发 ResetRegAll/重置流程。 |
| `EXCEPTION_SIGNAL_RECEIVED` | **F70013** | `sche.except.signal`、`sche.except.reset` | 收到信号、调用原始 signal handler、重复 reset 等异常信号处理场景。 |
| `THREAD_INIT_ARGS_INVALID` | **F70014** | `sche.thread.init` | 线程 / device run 入口参数非法（如调度线程启动参数不满足约束）。 |

### 2. CONTROL_FLOW（控制流执行，`MachineErrorCategory::CONTROL_FLOW`，F71xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `CTRL_FLOW_EXEC_FAILED` | **F71001** | `ctrl.ctrlflow.leave` | 控制流执行返回错误码（整体 control flow 失败）。 |
| `ROOT_ALLOC_CTX_NULL` | **F71002** | `ctrl.ctrlflow.call.root_alloc` | RootAlloc 调用时上下文指针为空。 |
| `ROOT_STITCH_CTX_NULL` | **F71003** | `ctrl.ctrlflow.call.root_stitch` | RootStitch 调用时上下文指针为空。 |
| `SYNC_FLAG_WAIT_TIMEOUT` | **F71004** | `ctrl.ctrlflow.wait_value` | 等待 `syncFlag` 置位超时。 |
| `DEVICE_TASK_BUILD_FAILED` | **F71005** | `ctrl.buildtask.leave` | 构建 DeviceTask 失败（返回 nullptr）。 |
| `READY_QUEUE_INIT_FAILED` | **F71006** | `ctrl.task.pre.queue.init` | Ready 队列初始化时 coreFunctionCnt 超出 stitchFunctionsize 等。 |
| `DEP_DUMP_FAILED` | **F71007** | `ctrl.dep.dump` | DumpDepend 过程中依赖 / workspace / tensor 地址 dump 异常。 |
| `READY_QUEUE_DUMP_FAILED` | **F71008** | `ctrl.queue.dump` | DumpReadyQueue 打印 ready 队列信息异常。 |
| `TASK_STATS_ABNORMAL` | **F71009** | `ctrl.task.end.stats` | ShowStats 输出统计信息异常（用于标记任务统计异常场景）。 |

### 3. WORKSPACE / SLAB（`MachineErrorCategory::WORKSPACE`，F72xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `SLAB_ADD_CACHE_FAILED` | **F72001** | `ctrl.task.pre.workspace.slab.alloc` | Slab cache 增加失败 / 无可用 slab。 |
| `SLAB_STAGE_LIST_INCONSISTENT` | **F72002** | `ctrl.task.pre.workspace.alloc.stage` | stageAlloc 链表指针异常、遍历后指针为空等。 |
| `SLAB_TYPE_INVALID` | **F72003** | `ctrl.task.pre.workspace.slab.type` | slab 内存类型非法。 |
| `WORKSPACE_INIT_RESOURCE_ERROR` | **F72004** | `workspace.init.resource` | workspace 初始资源（如 slabCapacity 数组）为空。 |
| `WORKSPACE_INIT_PARAM_INVALID` | **F72005** | `workspace.init.check` | slabTypeNum 超过允许值等初始化参数非法。 |
| `WS_TENSOR_ADDRESS_OUT_OF_RANGE` | **F72006** | `ctrl.task.pre.workspace.verify` | workspace 张量落在错误区域、跨界或非 in/out 张量不在 workspace。 |
| `SLAB_CAPACITY_CALC_INVALID` | **F72007** | `ctrl.task.pre.workspace.dump` | Workspace dump 时发现 size / 地址异常，可归为容量计算不一致。 |

### 4. DUMP / DFX / PROFILING（`MachineErrorCategory::DUMP_DFX`，F73xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `DUMP_MEMCPY_FAILED` | **F73001** | `sche.dump.prep` | DumpTensorData 中 `memcpy_s` 失败。 |
| `DUMP_TENSOR_INFO_FAILED` | **F73002** | `sche.dump.info` | dump tensor info 失败。 |
| `DUMP_TENSOR_DATA_FAILED` | **F73003** | `sche.dump.data` | dump tensor data 失败。 |
| `METRIC_ALLOC_OR_WAIT_TIMEOUT` | **F73004** | `sche.prof.aicore.getaddr`、`sche.prof.aicore.wait_finish` | metric 指针为空或等待 metric 结束超时。 |
| `PERF_TRACE_FORMAT_ERROR` | **F73005** | `sche.task.end.perf.format` | PerfEvtMgr 格式化 perf 文本出错。 |
| `PERF_TRACE_DUMP_ERROR` | **F73006** | `sche.task.end.perf.trace` | Dump perf trace（header、事件、profile）过程异常。 |
| `DFX_AICPU_TIMEOUT` | **F73007** | `sche.task.run.sync.timeout`（DFX 分支） | DFX 模式下 aicpu 超时退出。 |

### 5. PROGRAM ENCODE（编解码与一致性，`MachineErrorCategory::PROGRAM_ENCODE`，F74xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `DYNFUNC_DATA_ALIGNMENT_ERROR` | **F74001** | `ctrl.task.pre.dynfunc.process` | DynFuncData 内多种指针（opAttrs / offsets / exprTbl / rawTensorAddr）对齐错误。 |
| `FUNC_OP_SIZE_MISMATCH` | **F74002** | `ctrl.task.pre.func.dump` | DevAscendFunctionDupped / RawTensorAddr dump 时 operationSize / addr 不一致。 |
| `STITCH_PRED_SUCC_MISMATCH` | **F74003** | `ctrl.task.pre.stitch.check` | dynPredCount 与 dynSuccCount 不一致。 |
| `STITCH_LIST_TOO_LARGE` | **F74004** | `ctrl.stitch.toomany_root` | stitchedList size 超过 `MAX_CACHED_FUNC_NUM`。 |
| `STITCH_HANDLE_INDEX_OUT_OF_RANGE` | **F74005** | `ctrl.task.pre.stitch.handle`、`ctrl.stitch.invalid_slot` | stitch 中 producer/consumer opIndex 或 slotIdx 越界。 |
| `CELL_MATCH_PARAM_INVALID` | **F74006** | `ctrl.task.pre.stitch.cell_match` | cellMatch 维度参数无效 / 维度过多等。 |
| `PROGRAM_RANGE_VERIFY_FAILED` | **F74007** | `ctrl.program.verify` | 控制流 cache 范围检查失败，begin/end/overlap/dataEnd 不一致。 |
| `CACHE_RELOC_KIND_INVALID` | **F74008** | `ctrl.task.pre.cache.reloc` | cacheKind 非法。 |

### 6. TENSOR META（张量元信息，`MachineErrorCategory::TENSOR_META`，F75xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `TENSOR_DIM_COUNT_EXCEEDED` | **F75001** | `task.pre.tensor.init` | 维度个数超过 `DEV_SHAPE_DIM_MAX`。 |
| `TENSOR_ENCODE_PTR_MISMATCH` | **F75002** | `task.pre.tensor.encode` | encode tensor data 时指针走过头/不在预期范围。 |
| `RAW_TENSOR_INDEX_OUT_OF_RANGE` | **F75003** | `ctrl.task.pre.attr.dump` | rawIndex 超过 rawTensorSize。 |
| `SHAPE_VALUE_MISMATCH` | **F75004** | `ctrl.task.pre.attr.dump` | 期望 shape 与实际 attr 中 shape 不一致。 |
| `TENSOR_DUMP_INFO_INCONSISTENT` | **F75005** | `ctrl.task.pre.tensor.dump` | DumpTensor 时头/尾/内容不一致。 |

### 7. SERVER / KERNEL（AICPU server，`MachineErrorCategory::SERVER_KERNEL`，F76xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `DYN_SERVER_ARGS_NULL` | **F76001** | `aicpu.static.server`、`sche.task.pre.dyn.server` | Static/Dynamic server 启动时 args / DeviceKernelArgs 为空。 |
| `DYN_SERVER_SAVE_SO_FAILED` | **F76002** | `aicpu.static.server`、`sche.task.pre.dyn.server` | 保存 so 文件失败（写入 / 创建失败）。 |
| `KERNEL_EXEC_FUNC_FAILED` | **F76003** | `aicpu.static.server`、`sche.task.run.dyn.server`、`sche.task.pre.dyn.server.init` | 执行 kernelFunc / init func 失败。 |
| `KERNEL_SO_OR_FUNC_LOAD_FAILED` | **F76004** | `sche.task.pre.kernel.load`、`sche.func.exec` | 打开 so / 查找 kernelName / funcKey 对应符号失败。 |
| `DYN_SERVER_RUN_FAILED` | **F76005** | `sche.task.run.dyn.server` | 动态 server 运行期执行失败。 |
| `DYN_SERVER_INIT_FAILED` | **F76006** | `sche.task.pre.dyn.server.init` | 动态 server 初始化阶段执行失败。 |

### 8. THREAD / MACHINE（线程 / 机器级，`MachineErrorCategory::THREAD_MACHINE`，F77xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `DEVICE_ARGS_INVALID` | **F77001** | `dev.unistream.init.no_cpu`、`sche.thread.init` | AICPU 数量不足 / 设备参数非法，无法满足调度线程需求。 |
| `SIGNAL_HANDLER_ABNORMAL` | **F77002** | `sche.except.signal` | 信号处理流程异常（原 handler 为 SIG_DFL / SIG_IGN / 需转调原 handler 等）。 |
| `RESET_REG_ALL_TRIGGERED` | **F77003** | `ctrl.signal.reset.enter`、`ctrl.signal.reset.leave` | 执行 ResetRegAll 前后触发的 reset 行为。 |

### 9. DATA STRUCTURE（内部数据结构，`MachineErrorCategory::DATA_STRUCTURE`，F78xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `DEV_RELOC_VECTOR_INDEX_OOB` | **F78001** | `data.valid` | `DevRelocVector` 下标越界访问。 |
| `SMALL_ARRAY_RESIZE_OOB` | **F78002** | `array.resize` | `SmallArray::resize` 请求超过最大容量。 |

---

## 排查建议

### F70006 HANDSHAKE_TIMEOUT

1. **确认设备与驱动**：NPU 设备可用、驱动正常，`npu-smi info` 无异常。
2. **确认资源与负载**：当前进程/容器内 NPU 占用是否过高，是否存在多进程争用同一设备。
3. **确认超时配置**：若存在握手/同步超时配置项，检查是否过短或与环境不符。
4. **查日志上下文**：结合同线程前后日志（如 “Schedule run init succ” 之后、AbnormalStop 相关）确认是首次握手失败还是运行中异常。

**关联 Skill**：[pypto-environment-setup](../../.opencode/skills/pypto-environment-setup/SKILL.md)（环境与 NPU 设备诊断、`npu-smi`、驱动与编译运行）