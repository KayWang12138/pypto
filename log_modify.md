# DEV_ERROR 报错场景梳理

本文档梳理代码中所有 `DEV_ERROR` 的报错场景，包含报错原因与报错提示。按模块/文件分类。

---

## 1. framework/src/machine/device/dynamic/aicore_manager.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `Aicpu %d proc finish %lu %lu %lu, but timeout !.` | Aicpu 执行任务后 SyncTaskFinish 未在预期内完成，finishedFunctionCnt 与 coreFunctionCnt 不一致导致超时 |
| 2 | `ProcessTaskLoop TIMEOUT: aicpuIdx=%d taskId=%lu allSentCnt=%u coreFunctionCnt=%lu lastSent=%u elapsedCycles=%lu ...` | ProcessTaskLoop 中发送/执行核函数超过 TIMEOUT_CYCLES 未完成 |
| 3 | `ProcessTaskLoop TIMEOUT AIC core[%d] pendingId=%u runningId=%u regFin=%lu status=%lu.` | 上述超时发生时，AIC 核上仍有未完成的 pending/running 任务 |
| 4 | `ProcessTaskLoop TIMEOUT AIV core[%d] pendingId=%u runningId=%u regFin=%lu status=%lu.` | 上述超时发生时，AIV 核上仍有未完成的 pending/running 任务 |
| 5 | `status %lu,pending taskid: %s,funcdata: %s` | DumpLastWord：核上有 pending 任务时打印状态与任务数据 |
| 6 | `status %lu,running taskid:%s,funcdata: %s` | DumpLastWord：核上有 running 任务时打印状态与任务数据 |
| 7 | `task %lu execute error %d, skip rest tasks` | 任务执行返回非 OK，跳过后续任务并进入异常处理 |
| 8 | `hand shake timeout.` | 与 AICore 握手超时（HandShake 失败） |
| 9 | `left aic core %d not stop, status %d, pending:%u, rungning:%u, regfinishid: %lu, core last status:%lu` | SyncTaskFinish 超时时，AIC 核未进入 CORE_FINISH_STOP |
| 10 | `left aiv core %d not stop, status %d, pending:%u, rungning:%u, regfinishid: %lu, core last status:%lu` | SyncTaskFinish 超时时，AIV 核未进入 CORE_FINISH_STOP |
| 11 | `SyncAicoreDevTaskFinish timeout notstopNum=%d.` | 等待所有 AICore 停止时超时，未停止核数量为 notstopNum |
| 12 | `readyQue tail=%u > readyQue capacity=%u` | 非设备模式下，就绪队列 PushReadyQue 后 tail 超过 capacity，队列越界 |
| 13 | `Aic core %d hand shake success.phyid %d` | 握手超时诊断：AIC 核握手成功时打印（用于 DumpAicoreStatusWhenTimeout） |
| 14 | `Aic core %d hand shake timeout status=%lu` | 握手超时诊断：AIC 核握手超时，打印核状态 |
| 15 | `Aiv core %d hand shake success.phyid %d` | 握手超时诊断：AIV 核握手成功时打印 |
| 16 | `Aiv core %d hand shake timeout status=%lu` | 握手超时诊断：AIV 核握手超时，打印核状态 |
| 17 | `HandShakeByGmWithPreSendTask timeout notHandshakeNum=%d.` | 通过 GM 与预发任务的握手超时，未完成握手核数 |
| 18 | `Aicpu %d handshake failed end.` | HandShake() 返回非 OK，Aicpu 握手失败结束 |
| 19 | `proc aicore aic %d failed.` | ForEachManageAicoreWithRet 中处理 AIC 核回调返回非 OK |
| 20 | `proc aicore aiv %d failed.` | ForEachManageAicoreWithRet 中处理 AIV 核回调返回非 OK |

---

## 2. framework/src/machine/device/dynamic/aicpu_task_manager.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `SyncAicpuTaskFinish timeout.` | 等待 Aicpu 侧任务全部完成时超过 TIMEOUT_CYCLES，Finished() 仍为 false |

---

## 3. framework/src/machine/device/dynamic/device_sche.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `Device machinr run invalid args aicnum:%u, blockdim:%u, launchAicpu num:%u` | Run 时参数非法：nrAic==0 或 nrValidAic==0 或 nrAicpu < NEED_LAUNCH_AICPU_MINNUM |
| 2 | `ResetRegAll` | 异常处理流程中开始复位所有核寄存器 |
| 3 | `Exception reset reg finish.` | 异常处理流程中寄存器复位完成 |
| 4 | `tile_fwk aicore prof:%s` | 性能跟踪：分批输出 aicore 性能数据（用 DEV_ERROR 作为输出通道） |
| 5 | `Aicpu num[%u] less than sche num[%u].` | 调度参数错误：scheCpuNum > nrAicpu-1，调度核数大于可用 Aicpu 数 |
| 6 | `Exception Signum[%d] Act.` | 信号处理：收到异常信号（如 SIGFPE/SIGBUS/SIGSEGV 等）时触发 |
| 7 | `Exception Already reset.` | 信号处理中已处于 reset 状态，避免重复处理 |
| 8 | `Exception call ori sigact.` | 未初始化时收到信号，需要调用原始信号处理 |
| 9 | `Ori sigact SIG_DFL.` | 原始处理为 SIG_DFL |
| 10 | `Ori sigact SIG_IGN.` | 原始处理为 SIG_IGN |
| 11 | `Call Ori sigact.` | 调用用户注册的原始信号处理函数 |
| 12 | `Server init failed` | CtrlServerInit 返回非 OK，内核控制服务初始化失败 |
| 13 | `Begin dump machine perf trace:` | 所有调度线程退出后开始写性能 trace（信息性输出） |
| 14 | `Finish dump machine perf trace.` | 性能 trace 写入完成（信息性输出） |

---

## 4. framework/src/machine/device/dynamic/device_ctrl.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `Init Task control failed, which ctx is null.` | InitTaskCtrl 时 DeviceExecuteContext* ctx 为空 |
| 2 | `Args has null in inputs[%p] outputs[%p] work[%p] or cfg[%p].\n` | EntryInit 时 kargs->inputs / outputs / workspace / cfgdata 任一为空 |

---

## 5. framework/src/machine/device/distributed/shmem_wait_until.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `taskCount : %u >= AICPU_TASK_ARRAY_SIZE : %lu` | CreateTaskData 时当前任务数已达数组上限，无法再创建任务 |
| 2 | `newTask is nullptr` | CreateTaskData 返回空（因上述满），InsertTask 失败 |
| 3 | `SignalTileOp* queue_ is Full, Need resize queue_, front_ = %u, rear_ = %u` | CircularQueue 入队时 rear 追上 front，队列已满需扩容 |
| 4 | `Queue is empty.` | CircularQueue 空队列时执行 Dequeue |
| 5 | `There is no this taskId: %lu` | EnqueueOp 时在 hashMap 中找不到对应 taskId 的 SignalTileOp |

---

## 6. framework/src/machine/device/dynamic/device_perf.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `%s.`（分隔线） | Dump 时用 DEV_ERROR 打印重复字符作为表格线 |
| 2 | `%40s \| %10s \| %10s \| %10s.` | Dump 表头："EventType \| Count \| Total(us) \| Avg(us)" |
| 3 | `%-40s \| %10ld \| %10lu \| %10.1f.` | Dump 每行性能事件：事件名、次数、总时间(us)、平均时间(us) |
| 4 | `tile_fwk aicpu prof:%s` | 分批输出 aicpu 性能 trace 字符串（用 DEV_ERROR 作为输出） |

说明：device_perf.h 中 DEV_ERROR 主要用于在设备侧输出性能统计/表格，并非严格“错误”语义。

---

## 7. framework/src/machine/device/dynamic/device_utils.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `%s dfx_timeout, aicpu force exit, ttl=%lu.` | CheckTimeOut 中超过 DFX_TIME_OUT_THRESHOLD（仅 DEV_IF_VERBOSE_DEBUG） |
| 2 | `%s timeout, aicpu force exit, ttl=%lu.` | CheckTimeOut 中超过 TIME_OUT_THRESHOLD，强制 Aicpu 退出 |

---

## 8. framework/src/machine/device/dynamic/context/device_execute_context.cpp

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `Wait sync flag timeout.` | 设备侧等待 startArgs->syncFlag == 1 超过 HAND_SHAKE_TIMEOUT（disableSync==0 时） |

---

## 9. framework/src/machine/device/dynamic/context/device_task_context.cpp

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `Stitched function count: %10lu.` 等 6 条 | ShowStats 统计输出：stitchedFuncNum、rootFuncNum、leafFuncNum、readyTaskNum、dynFuncDataSize、leafFuncDataSize（用 DEV_ERROR 输出） |
| 2 | `coreFunctionCnt (%lu) exceeds stitchFunctionsize (%u), cannot build ready queue.` | InitReadyQueues 时 coreFunctionCnt 大于 devProg->stitchFunctionsize，无法构建就绪队列 |
| 3 | `cceBinary address is not aligned.` | BuildDynFuncData 中 cceBinary 地址未按 CCE_BINARY_MOD 对齐 |
| 4 | `hcclContext size mismatch, dyndata size: %zu, devProg size: %zu` | DynFuncData 与 DevAscendProgram 的 hcclContext 大小不一致 |
| 5 | `opAttrs address is not aligned.` | opAttrs 未按 OP_ATTRS_PRE_NUM 对齐 |
| 6 | `opAtrrOffsets address is not aligned.` | opAtrrOffsets 未按 OP_ATTRS_OFFSET_PRE_NUM 对齐 |
| 7 | `exprTbl address is not aligned.` | exprTbl 未按 EXPR_TABLE_PRE_NUM 对齐 |
| 8 | `rawTensorAddr address is not aligned.` | rawTensorAddr 未按 RAW_TENSOR_ADDR_MASK 对齐 |
| 9 | `%s: coreFunctionCnt: %d` / `ready queue aiv: %d-%d` 等 | DumpReadyQueue 调试输出：coreFunctionCnt、各就绪队列 head/tail 及元素（用 DEV_ERROR 输出） |

---

## 10. framework/src/machine/utils/dynamic/dev_workspace.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `Func (%2zu) %16s rawTensor[%2zu], @% PRIx64 [%zu bytes]%s.` | 内存 dump/校验时打印函数与 rawTensor 信息（DEBUG_MEM_DUMP 等） |
| 2 | `Invalid workspace tensor (not completely inside any workspace segment):` | VerifyStitchedListMemory 中张量不在任一 workspace 段内 |
| 3 | `Memory crossing workspace boundary:` | 张量跨越 workspace 边界 |
| 4 | `Non input/output tensor outside of workspace:` | 非输入/输出张量不在 workspace 内 |
| 5 | `slabCapacity is nullptr` | CalculateSlabCapacityPerType 中 slabCapacity 为空 |
| 6 | `slabTypeNum exceeds the allowed typenum %u` | slabTypeNum 超过 WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT |
| 7 | `Slab alloc null,type=%u,objsize=%u.` | SlabAlloc 在可重试条件下仍分配失败（首任务即失败） |
| 8 | `Invalid slab memory type: %u` | SlabTryDynAddCache 中 type 不在合法区间（非 COHERENT 且非 STITCH 范围） |

---

## 11. framework/src/machine/utils/dynamic/dev_encode_types.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `Index out of bounds: idx=%zu, size=%zu` | DevRelocVector::operator[] 下标 idx >= size_（const 与非 const 各一处） |

---

## 12. framework/src/machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[RelocDescFromCache] Invalid kind: %lu\n` | RelocDescFromCache 中 desc.cacheKind 不是已知的 ADDRESS_CACHE_KIND_* 之一 |

---

## 13. framework/src/machine/utils/dynamic/dev_encode_program.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `Assertion failed: data (0x%p) != rangeList[0].begin (0x%p)` | 校验 data 与 rangeList[0].begin 一致 |
| 2 | `Assertion failed: rangeList[0].begin (0x%p) > rangeList[0].end (0x%p)` | 第一段 range 的 begin <= end 校验 |
| 3 | `Ranges overlap: range[%d].end (0x%p) > range[%d].begin (0x%p)` | 相邻 range 重叠 |
| 4 | `Invalid range: range[%d].begin (0x%p) > range[%d].end (0x%p)` | 某段 range 的 begin > end |
| 5 | `Last range end does not match data end: rangeList.back().end (0x%p) != dataEnd (0x%p)` | 最后一段的 end 与 data 末尾不一致 |

---

## 14. framework/src/machine/utils/dynamic/dev_encode_function_dupped_data.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `GetOperationStitch: operation %d has invalid outcast stitch index 0` | GetOperationStitch(maybeNull=false) 时 outcastStitchIndex==0，非法（两处重载） |
| 2 | `Invalid rawIndex=%lu, exceeds raw tensor size=%lu` | DumpAttr 中 rawIndex >= GetRawTensorSize() |
| 3 | `Dimension mismatch: info.dim=%d, rawTensor->dim=%d` | 属性维度与 rawTensor 维度不一致 |
| 4 | `Shape mismatch at dim %d: expacted=%ld, got=%ld` | 某维 shape 与期望不符 |
| 5 | `Final dimension mismatch after shape validation: info.dim=%d, rawTensor->dim=%d` | 维度校验后仍不一致 |
| 6 | `Tensor address mismatch at index %lu: addr=%lu, addrEx=%lu.` | DumpDyn 中 GetRawTensorAddr(i) != GetRawTensorAddrEx(i) |

---

## 15. framework/src/machine/utils/dynamic/dev_encode_function_dupped_data.cpp

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `GetOperationSize mismatch: source=%zu, self=%u` | Dump 时 DuppedData 的 GetOperationSize 与 Source 的 GetOperationSize 不一致 |

---

## 16. framework/src/machine/utils/device_log.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `Assertion failed (%s): ` + fmt | DEV_ASSERT_MSG(expr, fmt, ...) 中 expr 为假 |
| 2 | `Assertion failed (%s)` | DEV_ASSERT(expr) 中 expr 为假 |
| 3 | `backtrace %s count:%d` | PrintBacktrace：打印回溯帧数 |
| 4 | `backtrace %s frame[%d]: %s` | PrintBacktrace：打印每一帧符号 |

说明：DEV_ERROR 在此处为宏定义及断言/回溯的通用输出接口，非业务单独错误场景。

---

## 汇总统计

- **aicore_manager.h**：20 条（超时、握手、就绪队列、核状态等）
- **device_sche.h**：14 条（参数、异常信号、性能 dump）
- **device_task_context.cpp**：9 条（对齐/大小校验、就绪队列、统计 dump）
- **dev_workspace.h**：8 条（workspace 校验、slab 分配与类型）
- **dev_encode_function_dupped_data.h**：6 条（stitch/索引/维度/形状/地址）
- **shmem_wait_until.h**：5 条（任务数组满、队列满/空、taskId 不存在）
- **device_ctrl.h**：2 条（ctx 空、kargs 空）
- **device_perf.h**：4 条（多为性能表/输出）
- **device_utils.h**：2 条（超时强制退出）
- **device_execute_context.cpp**：1 条（等待 syncFlag 超时）
- **aicpu_task_manager.h**：1 条（SyncAicpuTaskFinish 超时）
- **dev_encode_program.h**：5 条（range 校验）
- **dev_encode_types.h**：2 条（下标越界）
- **dev_encode_program_ctrlflow_cache.h**：1 条（cache kind 非法）
- **dev_encode_function_dupped_data.cpp**：1 条（OperationSize 不一致）
- **device_log.h**：断言与 backtrace 的通用 DEV_ERROR 使用

以上为当前仓库中 **DEV_ERROR** 的全部使用场景（不含 643.diff / 649.diff 等补丁文件内的改动）。

---

## 17. framework/src/machine/device/machine_interface/pypto_aicpu_interface.cpp

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[init] StaticPyptoKernelServer failed: args is null.` | StaticPyptoKernelServer 时 args 为空 |
| 2 | `[resource] StaticPyptoKernelServer failed: cannot save so file.` | 保存 so 文件失败 |
| 3 | `[kernel_exec] StaticPyptoKernelServer failed: kernelFunc [%s] execution error.` | 静态服务器内核函数执行失败 |
| 4 | `[init] DynPyptoKernelServerNull failed: input args is null.` | 动态服务器初始化时输入参数为空 |
| 5 | `[init] DynPyptoKernelServerNull failed: DeviceKernelArgs is null.` | DeviceKernelArgs 为空 |
| 6 | `[resource] DynPyptoKernelServerNull failed: cannot save so file for device %u.` | 为指定设备保存 so 文件失败 |
| 7 | `[kernel_exec] DynPyptoKernelServer failed: kernelFunc [%s] execution error.` | 动态服务器内核函数执行失败 |
| 8 | `[kernel_exec] DynPyptoKernelServerInit failed: kernelFunc [%s] execution error.` | 动态服务器初始化函数执行失败 |

---

## 18. framework/src/machine/device/machine_interface/pypto_aicpu_interface.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[resource] SaveSoFile failed: cannot create file [%s].` | 无法创建 so 文件 |
| 2 | `[resource] SaveSoFile failed: cannot write to file [%s].` | 无法写入 so 文件 |
| 3 | `[kernel_exec] ExecuteFunc failed: kernel func[%lu] is invalid, cannot get from so %s.` | 无法从 so 获取内核函数 |
| 4 | `[resource] LoadTileFwkKernelFunc failed: cannot open so %s.` | 无法打开 so 文件 |
| 5 | `[kernel_exec] LoadTileFwkKernelFunc failed: kernelName [%s] is null in so.` | 内核函数在 so 中为空 |
| 6 | `[kernel_exec] GetTileFwkKernelFunc failed: function[%lu] is null.` | 函数为空 |

---

## 19. framework/src/machine/device/dynamic/wrap_manager.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[task_dispatch] Cannot find wrapInfo in wrapQueueForThread: wrapId not found.` | 在 wrapQueueForThread 中找不到 wrapInfo |

---

## 20. framework/src/machine/device/dynamic/context/device_stitch_context.cpp

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[data_valid] Stitch check failed: dynPredCount=%u does not match dynSuccCount=%u.` | 动态前驱计数与后继计数不匹配 |
| 2 | `[resource] Stitch list size %u exceeds maximum allowed cached function number %zu.` | Stitch 列表大小超过最大缓存函数数 |
| 3 | `[data_valid] producerOperationIdx %zu exceeds the size of GetOperation %zu.` | 生产者操作索引越界 |
| 4 | `[data_valid] consumerOperationIdx %zu exceeds the size of GetOperation %zu.` | 消费者操作索引越界 |
| 5 | `[data_valid] slotIdx %d is larger than slotSize %zu.` | slotIdx 超过 slotSize |

---

## 21. framework/src/machine/device/dynamic/aicore_hal.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[sync_wait] WaitFinQueue timeout: CoreId %d cannot get finish flag.` | 等待完成队列超时，无法获取完成标志 |
| 2 | `[resource] GetMetrics failed: aicore %d has null metric.` | 获取指标失败，metric 为空 |
| 3 | `[sync_wait] GetMetrics timeout: wait metrics done timeout.` | 等待指标完成超时 |
| 4 | `[init] HandShakeByGm timeout: core %d handshake failed.` | 通过 GM 握手超时 |

---

## 22. framework/src/machine/device/dump/aicore_dump.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[data_valid] DumpTensorData failed: memcpy_s failed with ret=%d.` | memcpy_s 失败 |
| 2 | `[data_valid] Dump failed: tensor info dump not successful.` | 张量信息 dump 失败 |
| 3 | `[data_valid] Dump failed: tensor data dump not successful.` | 张量数据 dump 失败 |
| 4 | `[resource] DoDump failed: IdeDumpStart, IdeDumpData, IdeDumpEnd function not found.` | IDE dump 函数未找到 |

---

## 23. framework/src/machine/device/distributed/shmem_wait_until.cpp

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[kernel_exec] PollCompleted failed: AicoreManager is nullptr.` | AicoreManager 为空 |

---

## 24. framework/src/machine/device/aicore_manager.h (非dynamic目录)

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[kernel_exec] ForEachManageAicoreWithRet failed: AIC core %d processing error.` | AIC 核处理失败 |
| 2 | `[kernel_exec] ForEachManageAicoreWithRet failed: AIV core %d processing error.` | AIV 核处理失败 |

---

## 25. framework/src/machine/device/aicore_manager.cpp

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[data_valid] SdmaPrefetch failed: invalid prefetch num %ld.` | 预取数量无效 |
| 2 | `[sync_wait] RunTask failed: wait tail AIC task timeout.` | 等待尾部 AIC 任务超时 |
| 3 | `[sync_wait] RunTask failed: wait tail AIV task timeout.` | 等待尾部 AIV 任务超时 |
| 4 | `[sync_wait] WaitAllAicoreFinish timeout: coreIdx=%d wait tail task finish timeout.` | 等待所有 AICore 完成超时 |
| 5 | `[kernel_exec] ResolveByCoreType failed: invalid core type MIX.` | 无效的核类型 MIX |
| 6 | `[init] HandkShake timeout: core %d handshake failed.` | 握手超时 |

---

## 26. framework/src/machine/utils/dynamic/dev_workspace.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[data_valid] Invalid workspace tensor: not completely inside any workspace segment.` | 张量不在任一 workspace 段内 |
| 2 | `[data_valid] Memory crossing workspace boundary.` | 张量跨越 workspace 边界 |
| 3 | `[data_valid] Non input/output tensor outside of workspace.` | 非输入/输出张量不在 workspace 内 |
| 4 | `[init] CalculateSlabCapacityPerType failed: slabCapacity is nullptr.` | slabCapacity 为空 |
| 5 | `[data_valid] CalculateSlabCapacityPerType failed: slabTypeNum %u exceeds the allowed typenum %u.` | slabTypeNum 超过允许的类型数 |

---

## 27. framework/src/machine/utils/dynamic/dev_encode_function_dupped_data.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[data_valid] GetOperationStitch failed: operation %d has invalid outcast stitch index 0.` | 操作的 outcast stitch 索引无效（两处重载） |

---

## 28. framework/src/machine/utils/dynamic/small_array.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[data_valid] resize failed: size %zu exceeds maximum allowed value %zu.` | resize 大小超过最大允许值 |

---

## 29. framework/src/machine/utils/dynamic/dev_tensor_creator.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[data_valid] Init failed: dimension count (%d) exceeds maximum allowed (%d).` | 维度数超过最大允许值 |
| 2 | `[data_valid] Decode failed: pointer mismatch, ptr (0x%p) != data.data() + data.size() (0x%p).` | 指针不匹配 |

---

## 30. framework/src/machine/utils/dynamic/dev_encode_function_stitch.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[data_valid] CellMatchGetIndexRange failed: cellMatchShapeDim is zero for dimension %d.` | cellMatchShapeDim 为零 |

---

## 31. framework/src/machine/utils/dynamic/allocator/slab_ws_allocator.h

| 序号 | 报错提示 | 报错原因 |
|------|----------|----------|
| 1 | `[resource] KeepStageAllocTail failed: stageAllocHead is null for cache index %u.` | stageAllocHead 为空 |
| 2 | `[resource] KeepStageAllocTail failed: stageAllocHead is null after loop for cache index %u, stageAllocTail: %p.` | 循环后 stageAllocHead 为空 |

---

## 更新后汇总统计

| 文件 | 条数 | 标签类型 |
|------|------|----------|
| aicore_manager.h (dynamic) | 20 | sync_wait, kernel_exec, init, task_dispatch |
| aicore_manager.h (非dynamic) | 2 | kernel_exec |
| aicore_manager.cpp | 6 | data_valid, sync_wait, kernel_exec, init |
| aicpu_task_manager.h | 1 | sync_wait |
| device_sche.h | 14 | init, exception, perf_trace |
| device_ctrl.h | 2 | init |
| shmem_wait_until.h | 5 | task_dispatch |
| shmem_wait_until.cpp | 1 | kernel_exec |
| device_perf.h | 4 | perf_trace |
| device_utils.h | 2 | sync_wait |
| device_execute_context.cpp | 5 | data_valid, kernel_exec, sync_wait, resource |
| device_task_context.cpp | 37 | perf_trace, data_valid, task_dispatch |
| device_stitch_context.cpp | 5 | data_valid, resource |
| wrap_manager.h | 1 | task_dispatch |
| aicore_hal.h | 4 | sync_wait, resource, init |
| aicore_dump.h | 4 | data_valid, resource |
| pypto_aicpu_interface.cpp | 8 | init, resource, kernel_exec |
| pypto_aicpu_interface.h | 6 | resource, kernel_exec |
| dev_workspace.h | 5 | data_valid, init |
| dev_encode_function_dupped_data.h | 2 | data_valid |
| small_array.h | 1 | data_valid |
| dev_tensor_creator.h | 2 | data_valid |
| dev_encode_function_stitch.h | 1 | data_valid |
| slab_ws_allocator.h | 2 | resource |

**总计：约140条 DEV_ERROR 日志**

---

## 标签分类说明

| 标签 | 用途 | 示例场景 |
|------|------|----------|
| `[sync_wait]` | 同步等待超时相关 | 任务同步超时、握手超时、等待完成标志超时 |
| `[kernel_exec]` | 内核执行错误相关 | 内核函数执行失败、核处理错误 |
| `[init]` | 初始化相关错误 | 参数为空、初始化失败、握手失败 |
| `[data_valid]` | 数据校验相关 | 地址未对齐、大小不匹配、索引越界 |
| `[task_dispatch]` | 任务调度相关 | 队列满/空、任务分发失败 |
| `[perf_trace]` | 性能跟踪输出 | 性能统计、trace输出 |
| `[exception]` | 异常处理相关 | 信号处理、寄存器复位 |
| `[resource]` | 资源分配相关 | 内存分配失败、文件操作失败 |

---

*文档更新时间：2026-02-24*
