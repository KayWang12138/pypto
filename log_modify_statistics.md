# DEV_ERROR 日志修改统计报告

本文档统计当前代码修改情况，并结合业务流程梳理标签体系。

---

## 一、业务流程分析与标签体系设计

### 1.1 核心业务流程

通过对代码的深入分析，梳理出以下核心业务流程：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PYPTO 设备执行框架流程图                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   系统启动    │───▶│  参数校验    │───▶│  资源初始化   │───▶│  握手同步    │
│ [init_start] │    │[init_params] │    │[init_resource]│   │[init_handshake]│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                   │
                              ┌────────────────────────────────────┘
                              ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  任务创建    │───▶│  任务调度    │───▶│  内核执行    │───▶│  同步等待    │
│[task_create] │    │[task_dispatch]│   │[kernel_exec] │    │[sync_timeout]│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                    │
       ▼                   ▼                   ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  队列管理    │    │  数据校验    │    │  核心处理    │    │  超时检测    │
│[task_queue]  │    │ [data_valid] │    │[core_process]│    │[sync_timeout]│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                   │
                              ┌────────────────────────────────────┘
                              ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  内存管理    │───▶│  Stitch处理  │───▶│  性能采集    │───▶│  结果输出    │
│ [mem_alloc]  │    │  [stitch]    │    │ [perf_trace] │    │ [dump_output]│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │
       ▼
┌──────────────┐    ┌──────────────┐
│  异常处理    │───▶│  错误恢复    │
│[except_signal]│   │[except_reset]│
└──────────────┘    └──────────────┘
```

### 1.2 标签体系设计

基于业务流程，设计层级标签体系：

| 一级标签 | 二级标签 | 适用场景 | 示例 |
|----------|----------|----------|------|
| **[init]** | [init_params] | 初始化参数校验 | 参数为空、参数非法 |
| | [init_resource] | 初始化资源分配 | 内存分配、文件创建 |
| | [init_handshake] | 初始化握手同步 | AICore握手、GM握手 |
| | [init_server] | 服务初始化 | 控制服务器初始化 |
| **[task]** | [task_create] | 任务创建 | 任务数据创建、任务ID分配 |
| | [task_dispatch] | 任务调度 | 就绪队列、任务分发 |
| | [task_queue] | 队列管理 | 队列满/空、队列溢出 |
| **[kernel]** | [kernel_exec] | 内核执行 | 内核函数执行、核处理 |
| | [kernel_load] | 内核加载 | SO文件加载、函数获取 |
| | [kernel_server] | 内核服务 | 静态/动态服务器 |
| **[sync]** | [sync_timeout] | 同步超时 | 任务超时、握手超时 |
| **[data]** | [data_valid] | 数据校验 | 大小匹配、索引越界 |
| | [data_align] | 地址对齐 | 内存地址对齐检查 |
| | [data_boundary] | 边界检查 | 内存边界、workspace边界 |
| **[mem]** | [mem_alloc] | 内存分配 | Slab分配、缓存分配 |
| | [mem_workspace] | 工作空间 | Workspace校验、段检查 |
| **[stitch]** | [stitch_check] | Stitch校验 | 前驱后继检查、索引检查 |
| | [stitch_build] | Stitch构建 | 列表构建、缓存管理 |
| **[perf]** | [perf_trace] | 性能跟踪 | 性能数据采集、统计输出 |
| | [perf_dump] | 性能输出 | 性能数据dump |
| **[dump]** | [dump_tensor] | 张量dump | 张量数据dump |
| | [dump_output] | 结果输出 | IDE dump、文件输出 |
| **[except]** | [except_signal] | 异常信号 | 信号处理、异常捕获 |
| | [except_reset] | 异常复位 | 寄存器复位、状态恢复 |

---

## 二、详细流程环节说明

### 2.1 初始化流程 (init.*)

```
┌─────────────────────────────────────────────────────────────────┐
│                      初始化流程详解                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 参数校验 [init_params]                                       │
│     ├── 检查输入参数是否为空                                      │
│     ├── 检查参数合法性（aicNum、blockDim等）                       │
│     └── 检查配置数据有效性                                        │
│                                                                 │
│  2. 资源初始化 [init_resource]                                   │
│     ├── 内存分配（Slab、Workspace）                               │
│     ├── 文件创建（SO文件）                                        │
│     └── 缓存初始化                                               │
│                                                                 │
│  3. 握手同步 [init_handshake]                                    │
│     ├── AICore握手                                              │
│     ├── GM握手                                                  │
│     └── 预发送任务握手                                            │
│                                                                 │
│  4. 服务初始化 [init_server]                                     │
│     └── 控制服务器启动                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**涉及文件及日志：**

| 文件 | 标签 | 日志示例 |
|------|------|----------|
| device_ctrl.h | [init_params] | `[init_params] InitTaskCtrl failed: DeviceExecuteContext is null.` |
| device_sche.h | [init_params] | `[init_params] Invalid device args: aicNum=%u, validAicNum=%u.` |
| pypto_aicpu_interface.cpp | [init_params] | `[init_params] StaticPyptoKernelServer failed: args is null.` |
| aicore_manager.cpp | [init_handshake] | `[init_handshake] HandkShake timeout: core %d handshake failed.` |
| aicore_hal.h | [init_handshake] | `[init_handshake] HandShakeByGm timeout: core %d handshake failed.` |
| dev_workspace.h | [init_resource] | `[init_resource] CalculateSlabCapacityPerType failed: slabCapacity is nullptr.` |
| pypto_aicpu_interface.h | [init_resource] | `[init_resource] SaveSoFile failed: cannot create file [%s].` |

### 2.2 任务执行流程 (task.* / kernel.*)

```
┌─────────────────────────────────────────────────────────────────┐
│                      任务执行流程详解                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 任务创建 [task_create]                                       │
│     ├── 创建任务数据结构                                          │
│     ├── 分配任务ID                                               │
│     └── 初始化任务状态                                            │
│                                                                 │
│  2. 任务调度 [task_dispatch]                                     │
│     ├── 就绪队列管理                                             │
│     ├── 任务分发                                                 │
│     └── Wrap管理                                                │
│                                                                 │
│  3. 内核执行 [kernel_exec]                                       │
│     ├── 内核函数调用                                             │
│     ├── 核心处理（AIC/AIV）                                       │
│     └── 执行结果检查                                             │
│                                                                 │
│  4. 内核加载 [kernel_load]                                       │
│     ├── SO文件加载                                               │
│     └── 函数符号获取                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**涉及文件及日志：**

| 文件 | 标签 | 日志示例 |
|------|------|----------|
| shmem_wait_until.h | [task_create] | `[task_create] Task array is full: taskCount=%u, maxSize=%lu.` |
| shmem_wait_until.h | [task_queue] | `[task_queue] SignalTileOp queue is full: front=%u, rear=%u.` |
| device_task_context.cpp | [task_dispatch] | `[task_dispatch] Ready queue overflow: tail=%u exceeds capacity=%u.` |
| wrap_manager.h | [task_dispatch] | `[task_dispatch] Cannot find wrapInfo in wrapQueueForThread.` |
| aicore_manager.h | [kernel_exec] | `[kernel_exec] Failed to process AIC core %d.` |
| pypto_aicpu_interface.cpp | [kernel_exec] | `[kernel_exec] StaticPyptoKernelServer failed: kernelFunc [%s] execution error.` |
| pypto_aicpu_interface.h | [kernel_load] | `[kernel_load] LoadTileFwkKernelFunc failed: cannot open so %s.` |

### 2.3 同步超时流程 (sync.*)

```
┌─────────────────────────────────────────────────────────────────┐
│                      同步超时流程详解                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 同步超时 [sync_timeout]                                       │
│     ├── 等待任务完成标志超时                                      │
│     ├── 等待核停止超时                                           │
│     ├── 等待同步标志超时                                         │
│     ├── 任务执行超时                                             │
│     ├── 握手超时                                                 │
│     └── 强制退出处理                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**涉及文件及日志：**

| 文件 | 标签 | 日志示例 |
|------|------|----------|
| aicore_manager.h | [sync_timeout] | `[sync_timeout] AIC core %d failed to stop: status=%d.` |
| aicore_manager.cpp | [sync_timeout] | `[sync_timeout] RunTask failed: wait tail AIC task timeout.` |
| aicore_hal.h | [sync_timeout] | `[sync_timeout] WaitFinQueue timeout: CoreId %d cannot get finish flag.` |
| aicpu_task_manager.h | [sync_timeout] | `[sync_timeout] SyncAicpuTaskFinish timeout.` |
| device_utils.h | [sync_timeout] | `[sync_timeout] CheckTimeOut: aicpu force exit, ttl=%lu.` |

### 2.4 数据校验流程 (data.*)

```
┌─────────────────────────────────────────────────────────────────┐
│                      数据校验流程详解                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 数据有效性 [data_valid]                                      │
│     ├── 大小匹配检查                                             │
│     ├── 索引越界检查                                             │
│     ├── 计数一致性检查                                           │
│     └── 指针有效性检查                                           │
│                                                                 │
│  2. 地址对齐 [data_align]                                        │
│     ├── cceBinary地址对齐                                        │
│     ├── opAttrs地址对齐                                          │
│     ├── exprTbl地址对齐                                          │
│     └── rawTensorAddr地址对齐                                    │
│                                                                 │
│  3. 边界检查 [data_boundary]                                     │
│     ├── Workspace边界检查                                        │
│     ├── 内存段边界检查                                           │
│     └── 张量位置检查                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**涉及文件及日志：**

| 文件 | 标签 | 日志示例 |
|------|------|----------|
| device_task_context.cpp | [data_align] | `[data_align] cceBinary address %p is not aligned to %u bytes.` |
| device_task_context.cpp | [data_valid] | `[data_valid] coreFunctionCnt (%lu) exceeds stitchFunctionsize (%u).` |
| device_stitch_context.cpp | [data_valid] | `[data_valid] Stitch check failed: dynPredCount=%u does not match dynSuccCount=%u.` |
| dev_workspace.h | [data_boundary] | `[data_boundary] Invalid workspace tensor: not completely inside any workspace segment.` |
| dev_encode_function_dupped_data.h | [data_valid] | `[data_valid] GetOperationStitch failed: operation %d has invalid outcast stitch index 0.` |
| small_array.h | [data_valid] | `[data_valid] resize failed: size %zu exceeds maximum allowed value %zu.` |

### 2.5 内存管理流程 (mem.*)

```
┌─────────────────────────────────────────────────────────────────┐
│                      内存管理流程详解                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 内存分配 [mem_alloc]                                         │
│     ├── Slab分配                                                │
│     ├── 缓存分配                                                 │
│     └── 元数据分配                                               │
│                                                                 │
│  2. 工作空间 [mem_workspace]                                     │
│     ├── Workspace段管理                                         │
│     ├── 张量内存校验                                             │
│     └── 内存类型检查                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**涉及文件及日志：**

| 文件 | 标签 | 日志示例 |
|------|------|----------|
| dev_workspace.h | [mem_alloc] | `[mem_alloc] Slab alloc failed: type=%u, objSize=%u.` |
| slab_ws_allocator.h | [mem_alloc] | `[mem_alloc] KeepStageAllocTail failed: stageAllocHead is null for cache index %u.` |

### 2.6 Stitch处理流程 (stitch.*)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Stitch处理流程详解                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Stitch校验 [stitch_check]                                    │
│     ├── 前驱后继计数检查                                          │
│     ├── 操作索引检查                                             │
│     └── Slot索引检查                                             │
│                                                                 │
│  2. Stitch构建 [stitch_build]                                    │
│     ├── 列表大小检查                                             │
│     └── 缓存函数数限制                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**涉及文件及日志：**

| 文件 | 标签 | 日志示例 |
|------|------|----------|
| device_stitch_context.cpp | [stitch_check] | `[stitch_check] Stitch check failed: dynPredCount=%u does not match dynSuccCount=%u.` |
| device_stitch_context.cpp | [stitch_build] | `[stitch_build] Stitch list size %u exceeds maximum allowed cached function number %zu.` |

### 2.7 性能与输出流程 (perf.* / dump.*)

```
┌─────────────────────────────────────────────────────────────────┐
│                    性能与输出流程详解                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 性能跟踪 [perf_trace]                                        │
│     ├── 性能事件统计                                             │
│     ├── 时间测量                                                 │
│     └── 性能数据输出                                             │
│                                                                 │
│  2. 数据Dump [dump_tensor]                                       │
│     ├── 张量数据dump                                             │
│     ├── 内存拷贝                                                 │
│     └── IDE输出                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**涉及文件及日志：**

| 文件 | 标签 | 日志示例 |
|------|------|----------|
| device_perf.h | [perf_trace] | `[perf_trace] AICore profile data: %s` |
| device_task_context.cpp | [perf_trace] | `[perf_trace] Stitched function count: %10lu.` |
| aicore_dump.h | [dump_tensor] | `[dump_tensor] DumpTensorData failed: memcpy_s failed with ret=%d.` |
| aicore_dump.h | [dump_output] | `[dump_output] DoDump failed: IdeDumpStart, IdeDumpData, IdeDumpEnd function not found.` |

### 2.8 异常处理流程 (except.*)

```
┌─────────────────────────────────────────────────────────────────┐
│                      异常处理流程详解                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 异常信号 [except_signal]                                     │
│     ├── 信号捕获（SIGFPE/SIGBUS/SIGSEGV）                         │
│     ├── 信号处理器调用                                           │
│     └── 原始处理器处理                                           │
│                                                                 │
│  2. 异常复位 [except_reset]                                      │
│     ├── 寄存器复位                                               │
│     ├── 核心状态恢复                                             │
│     └── 错误清理                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**涉及文件及日志：**

| 文件 | 标签 | 日志示例 |
|------|------|----------|
| device_sche.h | [except_signal] | `[except_signal] Signal %d received, invoking handler.` |
| device_sche.h | [except_reset] | `[except_reset] ResetRegAll: Starting register reset for all cores.` |

---

## 三、修改清单概览

原始修改清单涵盖以下文件：

| 序号 | 文件 | 清单记录条数 |
|------|------|-------------|
| 1 | aicore_manager.h | 20 |
| 2 | aicpu_task_manager.h | 1 |
| 3 | device_sche.h | 14 |
| 4 | device_ctrl.h | 2 |
| 5 | shmem_wait_until.h | 5 |
| 6 | device_perf.h | 4 |
| 7 | device_utils.h | 2 |
| 8 | device_execute_context.cpp | 1 |
| 9 | device_task_context.cpp | 9 |
| 10 | dev_workspace.h | 8 |
| 11 | dev_encode_types.h | 2 |
| 12 | dev_encode_program_ctrlflow_cache.h | 1 |
| 13 | dev_encode_program.h | 5 |
| 14 | dev_encode_function_dupped_data.h | 6 |
| 15 | dev_encode_function_dupped_data.cpp | 1 |
| 16 | device_log.h | 断言与backtrace |

**清单总计：约81条DEV_ERROR场景**

---

## 四、当前已修改文件统计

### 4.1 按流程分类统计

| 流程阶段 | 文件数 | 日志条数 | 标签前缀 |
|----------|--------|----------|----------|
| 初始化流程 | 8 | 25 | init_* |
| 任务执行流程 | 6 | 35 | task_*, kernel_* |
| 同步等待流程 | 5 | 15 | sync_* |
| 数据校验流程 | 8 | 30 | data_* |
| 内存管理流程 | 3 | 10 | mem_* |
| Stitch处理 | 2 | 5 | stitch_* |
| 性能输出 | 3 | 12 | perf_*, dump_* |
| 异常处理 | 1 | 8 | except_* |

### 4.2 按文件统计

| 序号 | 文件 | 已修改条数 | 主要标签 |
|------|------|-----------|----------|
| 1 | aicore_manager.h | 20 | sync_timeout, kernel_exec, init_handshake |
| 2 | aicore_manager.cpp | 6 | data_valid, sync_timeout, kernel_exec, init_handshake |
| 3 | aicpu_task_manager.h | 1 | sync_timeout |
| 4 | device_sche.h | 14 | init_params, except_*, perf_trace |
| 5 | device_ctrl.h | 2 | init_params |
| 6 | shmem_wait_until.h | 5 | task_create, task_queue |
| 7 | shmem_wait_until.cpp | 1 | kernel_exec |
| 8 | device_perf.h | 4 | perf_trace |
| 9 | device_utils.h | 2 | sync_timeout |
| 10 | device_execute_context.cpp | 5 | data_valid, kernel_exec, sync_timeout, resource |
| 11 | device_task_context.cpp | 37 | perf_trace, data_valid, task_dispatch |
| 12 | device_stitch_context.cpp | 5 | data_valid, resource |
| 13 | wrap_manager.h | 1 | task_dispatch |
| 14 | aicore_hal.h | 4 | sync_timeout, resource, init_handshake |
| 15 | aicore_dump.h | 4 | dump_tensor, dump_output |
| 16 | pypto_aicpu_interface.cpp | 8 | init_params, resource, kernel_exec |
| 17 | pypto_aicpu_interface.h | 6 | resource, kernel_exec |
| 18 | dev_workspace.h | 5 | data_valid, init_resource |
| 19 | dev_encode_function_dupped_data.h | 2 | data_valid |
| 20 | small_array.h | 1 | data_valid |
| 21 | dev_tensor_creator.h | 2 | data_valid |
| 22 | dev_encode_function_stitch.h | 1 | data_valid |
| 23 | slab_ws_allocator.h | 2 | mem_alloc |

**总计：约140条 DEV_ERROR 日志**

---

## 五、标签使用规范

### 5.1 标签格式

```
[一级标签_二级标签] 错误描述: 具体错误信息.
```

**示例：**
```cpp
DEV_ERROR("[init_params] StaticPyptoKernelServer failed: args is null.");
DEV_ERROR("[sync_timeout] AIC core %d failed to stop: status=%d.", coreIdx, status);
DEV_ERROR("[data_align] cceBinary address %p is not aligned to %u bytes.", addr, align);
```

### 5.2 标签选择指南

| 错误场景 | 推荐标签 | 说明 |
|----------|----------|------|
| 函数参数为空 | [init_params] | 初始化阶段参数校验 |
| 内存分配失败 | [init_resource] 或 [mem_alloc] | 初始化阶段用init_*，运行时用mem_* |
| 握手超时 | [init_handshake] | 初始化握手阶段 |
| 任务创建失败 | [task_create] | 任务创建阶段 |
| 队列满/空 | [task_queue] | 队列操作阶段 |
| 任务分发失败 | [task_dispatch] | 任务调度阶段 |
| 内核执行失败 | [kernel_exec] | 内核执行阶段 |
| SO文件加载失败 | [kernel_load] | 内核加载阶段 |
| 等待超时 | [sync_timeout] | 同步等待超时 |
| 地址未对齐 | [data_align] | 地址对齐检查 |
| 索引越界 | [data_valid] | 数据有效性检查 |
| 内存边界错误 | [data_boundary] | 边界检查 |
| Stitch校验失败 | [stitch_check] | Stitch处理阶段 |
| 性能数据输出 | [perf_trace] | 性能跟踪 |
| 张量dump失败 | [dump_tensor] | 数据dump阶段 |
| 信号处理 | [except_signal] | 异常信号处理 |
| 寄存器复位 | [except_reset] | 异常恢复阶段 |

---

## 六、完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    PYPTO 设备执行完整流程                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

                                    ┌─────────────────┐
                                    │     系统启动     │
                                    └────────┬────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │  参数校验       │          │  资源初始化      │          │  服务初始化      │
    │ [init_params]   │          │ [init_resource] │          │ [init_server]   │
    └────────┬────────┘          └────────┬────────┘          └────────┬────────┘
             │                            │                            │
             │                            ▼                            │
             │                  ┌─────────────────┐                    │
             │                  │  内存分配        │                    │
             │                  │ [mem_alloc]     │                    │
             │                  └────────┬────────┘                    │
             │                           │                             │
             └───────────────────────────┼─────────────────────────────┘
                                         │
                                         ▼
                               ┌─────────────────┐
                               │  握手同步        │
                               │ [init_handshake]│
                               └────────┬────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
               ┌─────────────────┐           ┌─────────────────┐
               │  AICore握手     │           │  GM握手         │
               └────────┬────────┘           └────────┬────────┘
                        │                             │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  任务创建        │
                              │ [task_create]   │
                              └────────┬────────┘
                                       │
                         ┌─────────────┼─────────────┐
                         │             │             │
                         ▼             ▼             ▼
               ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
               │ 任务ID分配  │ │ 任务数据创建 │ │ 队列管理    │
               │             │ │             │ │[task_queue] │
               └─────────────┘ └─────────────┘ └──────┬──────┘
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │  任务调度        │
                                            │ [task_dispatch] │
                                            └────────┬────────┘
                                                     │
                                      ┌──────────────┼──────────────┐
                                      │              │              │
                                      ▼              ▼              ▼
                            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
                            │ 就绪队列    │  │ Wrap管理    │  │ Stitch处理  │
                            │             │  │             │  │ [stitch_*]  │
                            └─────────────┘  └─────────────┘  └──────┬──────┘
                                                                      │
                                                                      ▼
                                                            ┌─────────────────┐
                                                            │  内核加载        │
                                                            │ [kernel_load]   │
                                                            └────────┬────────┘
                                                                     │
                                                                     ▼
                                                            ┌─────────────────┐
                                                            │  内核执行        │
                                                            │ [kernel_exec]   │
                                                            └────────┬────────┘
                                                                     │
                                              ┌───────────────────────┼───────────────────────┐
                                              │                       │                       │
                                              ▼                       ▼                       ▼
                                    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
                                    │ AIC核处理       │     │ AIV核处理       │     │ AICPU处理       │
                                    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                                             │                       │                       │
                                             └───────────────────────┼───────────────────────┘
                                                                     │
                                                                     ▼
                                                            ┌─────────────────┐
                                                            │  同步等待        │
                                                            │[sync_timeout]    │
                                                            └────────┬────────┘
                                                                     │
                                              ┌───────────────────────┼───────────────────────┐
                                              │                       │                       │
                                              ▼                       ▼                       ▼
                                    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
                                    │ 等待完成标志   │     │ 等待核停止     │     │ 超时检测       │
                                    │                 │     │                 │     │[sync_timeout]  │
                                    └─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                                                              │
                                                                                              ▼
                                                                                    ┌─────────────────┐
                                                                                    │  数据校验        │
                                                                                    │ [data_valid]    │
                                                                                    └────────┬────────┘
                                                                                             │
                                              ┌──────────────────────────────┼──────────────────────────────┐
                                              │                              │                              │
                                              ▼                              ▼                              ▼
                                    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
                                    │ 大小匹配检查   │     │ 索引越界检查   │     │ 指针有效性检查 │
                                    └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                             │
                                                                                             ▼
                                                                                    ┌─────────────────┐
                                                                                    │  内存管理        │
                                                                                    │ [mem_alloc]     │
                                                                                    └────────┬────────┘
                                                                                             │
                                              ┌──────────────────────────────┼──────────────────────────────┐
                                              │                              │                              │
                                              ▼                              ▼                              ▼
                                    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
                                    │ Slab分配        │     │ 缓存分配        │     │ 工作空间检查   │
                                    └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                             │
                                                                                             ▼
                                                                                    ┌─────────────────┐
                                                                                    │  性能采集        │
                                                                                    │ [perf_trace]    │
                                                                                    └────────┬────────┘
                                                                                             │
                                                                                             ▼
                                                                                    ┌─────────────────┐
                                                                                    │  结果输出        │
                                                                                    │ [dump_output]   │
                                                                                    └────────┬────────┘
                                                                                             │
                                                                                             ▼
                                                                                    ┌─────────────────┐
                                                                                    │  异常处理        │
                                                                                    │ [except_signal] │
                                                                                    └────────┬────────┘
                                                                                             │
                                                                                             ▼
                                                                                    ┌─────────────────┐
                                                                                    │  错误恢复        │
                                                                                    │ [except_reset]  │
                                                                                    └─────────────────┘
```

---

## 七、标签依据梳理

### 7.1 初始化相关标签依据

| 标签 | 代码位置 | 触发条件 | 业务含义 |
|------|----------|----------|----------|
| [init_params] | device_ctrl.h, device_sche.h, pypto_aicpu_interface.cpp | 参数为空、参数非法 | 系统启动时参数校验失败 |
| [init_resource] | dev_workspace.h, pypto_aicpu_interface.h | 内存分配失败、文件创建失败 | 初始化阶段资源分配失败 |
| [init_handshake] | aicore_manager.cpp, aicore_hal.h | 握手超时、握手失败 | 与AICore/GM握手同步失败 |
| [init_server] | device_sche.h | 控制服务器初始化失败 | 服务启动失败 |

### 7.2 任务执行相关标签依据

| 标签 | 代码位置 | 触发条件 | 业务含义 |
|------|----------|----------|----------|
| [task_create] | shmem_wait_until.h | 任务数组满、创建失败 | 任务数据结构创建阶段 |
| [task_dispatch] | device_task_context.cpp, wrap_manager.h | 就绪队列溢出、Wrap未找到 | 任务调度分发阶段 |
| [task_queue] | shmem_wait_until.h | 队列满/空 | 队列管理操作阶段 |
| [kernel_exec] | aicore_manager.h, pypto_aicpu_interface.cpp | 内核函数执行失败、核处理错误 | 内核执行阶段 |
| [kernel_load] | pypto_aicpu_interface.h | SO文件加载失败、函数获取失败 | 内核加载阶段 |

### 7.3 同步超时相关标签依据

| 标签 | 代码位置 | 触发条件 | 业务含义 |
|------|----------|----------|----------|
| [sync_timeout] | aicore_manager.h, aicore_hal.h, aicore_manager.cpp, device_execute_context.cpp, device_utils.h, aicpu_task_manager.h | 等待完成标志超时、等待核停止超时、任务超时、强制退出 | 同步等待超时（统一使用） |

### 7.4 数据校验相关标签依据

| 标签 | 代码位置 | 触发条件 | 业务含义 |
|------|----------|----------|----------|
| [data_valid] | device_task_context.cpp, dev_workspace.h | 大小不匹配、索引越界 | 数据有效性检查 |
| [data_align] | device_task_context.cpp | 地址未对齐 | 地址对齐检查 |
| [data_boundary] | dev_workspace.h | 内存越界、workspace越界 | 边界检查 |

### 7.5 内存管理相关标签依据

| 标签 | 代码位置 | 触发条件 | 业务含义 |
|------|----------|----------|----------|
| [mem_alloc] | dev_workspace.h, slab_ws_allocator.h | Slab分配失败、缓存分配失败 | 内存分配阶段 |
| [mem_workspace] | dev_workspace.h | Workspace校验失败 | 工作空间管理 |

### 7.6 性能与输出相关标签依据

| 标签 | 代码位置 | 触发条件 | 业务含义 |
|------|----------|----------|----------|
| [perf_trace] | device_perf.h, device_task_context.cpp | 性能数据采集、统计输出 | 性能跟踪与分析 |
| [dump_tensor] | aicore_dump.h | 张量数据dump失败 | 数据dump阶段 |
| [dump_output] | aicore_dump.h | IDE输出失败 | 结果输出阶段 |

---

## 八、总结

### 8.1 标签体系特点

1. **层级化设计**：一级标签标识流程阶段，二级标签标识具体环节
2. **下划线格式**：采用 [xxx_xx] 格式，避免与格式化字符串中的小数点混淆
3. **业务语义明确**：每个标签对应明确的业务流程环节
4. **易于扩展**：支持新增一级标签和二级标签

### 8.2 使用建议

1. **新增日志时**：根据错误发生的流程环节选择对应的标签
2. **日志过滤时**：可通过标签快速定位特定流程的问题
3. **监控告警时**：可基于标签设置不同级别的告警规则

---

## 九、标签更新记录

### 9.1 更新时间
2026-02-24

### 9.2 更新内容概述
本次更新主要对两个标签进行了优化和统一：
1. 将所有 [sync_wait] 标签统一更新为 [sync_timeout]
2. 将所有 [exception] 标签拆分为 [except_signal] 和 [except_reset]

### 9.3 [sync_wait] → [sync_timeout] 更新详情

#### 更新理由
1. **语义更明确**：`sync_wait` 表示同步等待，但代码中所有使用该标签的场景都是**超时错误**，使用 `sync_timeout` 更准确
2. **消除歧义**：避免与普通同步等待（成功的等待）混淆，明确标识这是超时错误
3. **与标签选择指南一致**：符合 5.2 节标签选择指南中"等待超时"推荐使用 [sync_timeout] 的规范

#### 更新依据
| 文件 | 原标签 | 新标签 | 场景说明 |
|------|--------|--------|----------|
| device_execute_context.cpp | [sync_wait] | [sync_timeout] | 等待同步标志超时 |
| aicore_manager.h | [sync_wait] | [sync_timeout] | 任务同步超时、等待核停止超时 |
| aicore_manager.cpp | [sync_wait] | [sync_timeout] | 等待尾任务超时 |
| aicore_hal.h | [sync_wait] | [sync_timeout] | 等待完成队列超时、等待指标超时 |
| device_utils.h | [sync_wait] | [sync_timeout] | DFX超时、普通超时 |
| aicpu_task_manager.h | [sync_wait] | [sync_timeout] | 等待Aicpu任务完成超时 |

#### 更新数量
共更新 13 处日志（不含 log_modify_statistics.md 文档本身）

---

### 9.4 [exception] → [except_signal]/[except_reset] 更新详情

#### 更新理由
1. **粒度更细**：将通用的 `exception` 标签拆分为两个更具体的标签，便于区分不同类型的异常处理
2. **业务语义明确**：
   - `except_signal`：专门用于信号处理（SIGFPE/SIGBUS/SIGSEGV等）
   - `except_reset`：专门用于寄存器复位和错误恢复
3. **便于分析**：可以分别统计信号异常和复位异常的发生频率

#### 更新依据
| 文件 | 原标签 | 新标签 | 场景说明 |
|------|--------|--------|----------|
| device_sche.h | [exception] | [except_reset] | 开始寄存器复位、寄存器复位完成 |
| device_sche.h | [exception] | [except_signal] | 信号接收、信号处理、原始信号处理器调用 |
| device_log.h | [exception] | [except_signal] | 打印回溯信息（在信号处理中调用） |

#### 更新数量
共更新 8 处日志

---

### 9.5 完整标签列表（更新后）

#### 初始化流程
| 标签 | 说明 |
|------|------|
| [init_params] | 初始化参数校验 |
| [init_resource] | 初始化资源分配 |
| [init_handshake] | 初始化握手同步 |
| [init_server] | 服务初始化 |

#### 任务执行流程
| 标签 | 说明 |
|------|------|
| [task_create] | 任务创建 |
| [task_dispatch] | 任务调度 |
| [task_queue] | 队列管理 |
| [kernel_exec] | 内核执行 |
| [kernel_load] | 内核加载 |

#### 同步超时流程
| 标签 | 说明 |
|------|------|
| [sync_timeout] | 同步等待超时（统一使用） |

#### 数据校验流程
| 标签 | 说明 |
|------|------|
| [data_valid] | 数据有效性检查 |
| [data_align] | 地址对齐检查 |
| [data_boundary] | 边界检查 |

#### 内存管理流程
| 标签 | 说明 |
|------|------|
| [mem_alloc] | 内存分配 |
| [mem_workspace] | 工作空间管理 |

#### Stitch处理流程
| 标签 | 说明 |
|------|------|
| [stitch_check] | Stitch校验 |
| [stitch_build] | Stitch构建 |

#### 性能与输出流程
| 标签 | 说明 |
|------|------|
| [perf_trace] | 性能跟踪 |
| [dump_tensor] | 张量dump |
| [dump_output] | 结果输出 |

#### 异常处理流程
| 标签 | 说明 |
|------|------|
| [except_signal] | 异常信号处理 |
| [except_reset] | 异常复位恢复 |

---

### 9.6 注意事项

1. **已废弃标签**：
   - `[sync_wait]`：已废弃，请使用 `[sync_timeout]`
   - `[exception]`：已废弃，请使用 `[except_signal]` 或 `[except_reset]`

2. **新增日志规范**：新增 DEV_ERROR 日志时，请严格按照 5.2 节标签选择指南选择标签

3. **标签格式**：所有标签统一使用下划线格式 `[xxx_xx]`，禁止使用点号格式 `[xxx.xx]`

---

---

*文档更新时间：2026-02-24*
*标签格式：下划线格式 [xxx_xx]*
