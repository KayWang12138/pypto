# MACHINE 超时检测机制优化设计文档

## 一、总览

### 1.1 超时检测修改总表（按修改类型排序）

#### 新增超时检测（4处）

| # | 函数名 | 文件 | 原状态 | 新状态 | 新阈值 | 新警告间隔 |
|---|--------|------|--------|--------|--------|-----------|
| 1 | `WaitForCpuMaskReady` | device_sche.h | — | 真超时 | 20分钟 | 10秒 |
| 2 | `SlabStageAllocMemSubmmit` | dev_workspace.h | 无超时 | 真超时 | 20分钟 | 1分钟 |
| 3 | `AllocateWait` | dev_start_args.h | 无超时 | 永不退出 | 无 | 10分钟 |
| 4 | `ProcessKeepTailCase` | slab_ws_allocator.h | 无超时 | 永不退出 | 无 | 10分钟 |

#### 修复伪超时（6处）

| # | 函数名 | 文件 | 原状态 | 原阈值 | 新状态 | 新阈值 | 新警告间隔 |
|---|--------|------|--------|--------|--------|--------|-----------|
| 5 | `AllocThreadIdxForDav2201` | device_sche.h | **伪超时** | 3秒重置 | 真超时 | 20分钟 | 10秒 |
| 6 | `ScheWait` (循环1) | device_sche.h | **伪超时** | 3秒重置 | 真超时 | 1分钟 | 10秒 |
| 7 | `ScheWait` (循环2) | device_sche.h | **伪超时** | 3秒重置 | 真超时 | 1分钟 | 10秒 |
| 8 | `AllocNewTaskCtrl` | device_ctrl.h | **伪超时** | 3秒重置 | 真超时 | 1分钟 | 10秒 |
| 9 | `SlabAlloc` (外层) | dev_workspace.h | **伪超时** | 10秒重置 | 真超时 | 20分钟 | 2分钟 |
| 10 | `SlabAlloc` (内层) | dev_workspace.h | **伪超时** | 10秒重置 | 真超时 | 20分钟 | 1分钟 |

#### 修改已有超时（7处）

| # | 函数名 | 文件 | 原状态 | 原阈值 | 新状态 | 新阈值 | 新警告间隔 |
|---|--------|------|--------|--------|--------|--------|-----------|
| 11 | `AllocThreadIdxForDav3510` | device_sche.h | 有超时 | 10秒 | 真超时 | 20分钟 | 10秒 |
| 12 | `ProcessTaskLoop` | aicore_manager.h | 有超时 | 10秒 | 真超时 | 10秒 | 1秒 |
| 13 | `RunManager` | aicore_manager.h | 有超时 | 10秒 | 真超时 | 10秒 | 1秒 |
| 14 | `SyncAicoreDevTaskFinish` | aicore_manager.h | 有超时 | 10秒 | 真超时 | 10秒 | 1秒 |
| 15 | `HandShakeByGmWithPreSendTask` | aicore_manager.h | 有超时 | 48秒 | 真超时 | 16分钟 | 1分钟 |
| 16 | `SyncAicpuTaskFinish` | aicpu_task_manager.h | 有超时 | 10秒 | 真超时 | 10秒 | 1秒 |
| 17 | `GetMetrics` | aicore_hal.h | 有超时 | 10秒 | 真超时 | 10秒 | 1秒 |

#### 保持不变（7处）

| # | 函数名 | 文件 | 状态 | 超时阈值 | 说明 |
|---|--------|------|------|---------|------|
| 18 | `GetNextLeafTask` | aicore_entry.h | ✅规范 | 250ms | 每1000次检查 |
| 19 | `GetRegHighValue` | aicore_entry.h | ✅规范 | 250ms | 每1000次检查 |
| 20 | `GetCoreFuncionData` | aicore_entry.h | ✅规范 | 250ms | 多处检查 |
| 21 | `AiCoreEntry` (外层) | aicore_entry.h | ✅规范 | 3秒 | 每1000次检查 |
| 22 | `AiCoreEntry` (内层) | aicore_entry.h | ✅规范 | 3秒 | 每1000次检查 |
| 23 | `WaitWaveSignal` | aicore_entry.h | ✅规范 | 50ms | dcci等待 |
| 24 | `RefreshParallelDevTaskByModifyFlag` | aicore_entry.h | ✅规范 | 250ms | 每1000次检查 |

#### 删除（2处）

| # | 内容 | 文件 | 原状态 | 说明 |
|---|------|------|--------|------|
| 25 | `WaitFinQueue` | aicore_hal.h | void返回超时 | 无实际使用 |
| 26 | `device_channel.h` | device_channel.h | 218行文件 | 阻塞接口已不使用 |

### 1.2 修改统计

| 修改类型 | 数量 |
|---------|------|
| 新增超时检测 | 4 |
| 修复伪超时 | 6 |
| 修改已有超时 | 7 |
| 保持不变 | 7 |
| 删除 | 2 |

### 1.3 关键修正

| 原文档错误 | 实际情况 |
|---------|---------|
| ❌ "超时阈值250ms" | ✅ `TIMEOUT_CYCLES` 在A2/A3（50MHz）上约为 **10秒** |
| ❌ "TIMEOUT_ONE_MINUTE = 1分钟" | ✅ 实际值为 **3秒**（已删除） |
| ❌ "超时后重置继续" | ✅ 新方案 **真正退出** |

### 1.4 新增统一超时常量（纳秒）

```cpp
constexpr uint64_t TIMEOUT_NS_10SEC   = 10ULL * NSEC_PER_SEC;     // 10秒
constexpr uint64_t TIMEOUT_NS_1MIN    = 60ULL * NSEC_PER_SEC;      // 1分钟
constexpr uint64_t TIMEOUT_NS_2MIN    = 120ULL * NSEC_PER_SEC;     // 2分钟
constexpr uint64_t TIMEOUT_NS_10MIN   = 600ULL * NSEC_PER_SEC;     // 10分钟
constexpr uint64_t TIMEOUT_NS_20MIN   = 1200ULL * NSEC_PER_SEC;    // 20分钟
constexpr uint64_t TIMEOUT_NS_INFINITE = UINT64_MAX;               // 永不超时
constexpr uint64_t HAND_SHAKE_TIMEOUT_NS = 960ULL * NSEC_PER_SEC;  // 16分钟
```

### 1.5 新增 `TimeoutState` 和 `__PYPTO_TIMEOUT_CHECK` 宏

```cpp
struct TimeoutState {
    uint64_t startCycles;
    uint64_t freq;
    uint64_t lastWarnNs;
    bool warnPrinted;
    
    TimeoutState() : startCycles(GetCycles()), freq(GetFreq()), 
                     lastWarnNs(0), warnPrinted(false) {}
    
    inline uint64_t ElapsedNs() const {
        return ((GetCycles() - startCycles) * NSEC_PER_SEC) / freq;
    }
};

#define __PYPTO_TIMEOUT_CHECK(state, timeout_ns, warn_interval_ns, error_code, action, \
                              warn_fmt, error_fmt, ...) \
    do { \
        uint64_t elapsed_ns = state.ElapsedNs(); \
        uint64_t elapsed_sec = elapsed_ns / NSEC_PER_SEC; \
        if (timeout_ns != TIMEOUT_NS_INFINITE && elapsed_ns > timeout_ns) { \
            DEV_ERROR(error_code, error_fmt ", elapsed %lu sec", ##__VA_ARGS__, elapsed_sec); \
            action; \
        } \
        if (elapsed_ns > state.lastWarnNs + warn_interval_ns) { \
            DEV_WARN(warn_fmt ", elapsed %lu sec", ##__VA_ARGS__, elapsed_sec); \
            state.lastWarnNs = elapsed_ns; \
        } \
    } while (0)
```

---

## 二、每个函数的while循环代码

### 2.1 新增超时检测（4处）

#### #1 `WaitForCpuMaskReady`（新增）

**文件**：`device_sche.h`
**超时**：20分钟退出，每10秒警告

```cpp
// 新增函数，统一Dav2201和Dav3510的逻辑
int WaitForCpuMaskReady(DeviceArgs* devArgs, int cpu, int curThreadIdx)
{
    TimeoutState state;
    while (__builtin_popcount(cpumask_.load(std::memory_order_acquire)) != static_cast<int>(devArgs->nrAicpu)) {
        __PYPTO_TIMEOUT_CHECK(state, TIMEOUT_NS_20MIN, TIMEOUT_NS_10SEC,
            ThreadErr::THREAD_CPU_ALLOC_FAILED,
            return DEVICE_MACHINE_ERROR,
            "#sche.thread.init: Thread alloc still waiting, threadIdx=%d, physicalCpu=%d.",
            "#sche.thread.init: Thread alloc timeout, threadIdx=%d, physicalCpu=%d.",
            curThreadIdx, cpu);
        sched_yield();
    }
    return npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
}
```

---

#### #2 `SlabStageAllocMemSubmmit`（新增）

**文件**：`dev_workspace.h`
**超时**：20分钟退出，每1分钟警告

```cpp
// 原代码（无超时）：
void SlabStageAllocMemSubmmit(DynDeviceTask* devTask) {
    while (!submmitTaskQueue_.TryEnqueue(devTask)) {
        DeviceTaskMemTryRecycle();
    }
}

// 新增后：
void SlabStageAllocMemSubmmit(DynDeviceTask* devTask) {
    TimeoutState state;
    
    while (!submmitTaskQueue_.TryEnqueue(devTask)) {
        DeviceTaskMemTryRecycle();
        
        __PYPTO_TIMEOUT_CHECK(state, TIMEOUT_NS_20MIN, TIMEOUT_NS_1MIN,
            WsErr::SLAB_ADD_CACHE_FAILED,
            return,
            "#workspace.submit: SlabStageAllocMemSubmmit still waiting 10min.",
            "#workspace.submit: SlabStageAllocMemSubmmit timeout 10min.");
    }
}
```

---

#### #3 `AllocateWait`（新增）

**文件**：`dev_start_args.h`
**超时**：永不退出，每10分钟警告

```cpp
// 原代码（无超时）：
void AllocateWait()
{
    while (Full()) {
        RuntimeYield();
    }
}

// 新增后：
void AllocateWait()
{
    TimeoutState state;
    
    while (Full()) {
        RuntimeYield();
        
        __PYPTO_TIMEOUT_CHECK(state, TIMEOUT_NS_INFINITE, TIMEOUT_NS_10MIN,
            WsErr::WORKSPACE_CAPACITY_INSUFFICIENT,
            ,
            "#ringbuffer.alloc: AllocateWait still waiting 10min, ring buffer full.",
            "#ringbuffer.alloc: AllocateWait timeout 10min, ring buffer full.");
    }
}
```

---

#### #4 `ProcessKeepTailCase`（新增）

**文件**：`slab_ws_allocator.h`
**超时**：永不退出，每10分钟警告

```cpp
// 原代码（无超时）：
void* temp = caches_[i].stageAllocHead;
while (*static_cast<void**>(temp) != caches_[i].stageAllocTail) {
    temp = *static_cast<void**>(temp);
}

// 新增后：
void ProcessKeepTailCase(uint32_t cacheIdx, StageAllocInfo& info)
{
    SlabCache& cache = caches_[cacheIdx];
    // ... 其他逻辑 ...
    
    TimeoutState stageAllocTimeoutState;
    void* targetTail = cache.stageAllocTail;
    while (*static_cast<void**>(temp) != targetTail) {
        __PYPTO_TIMEOUT_CHECK(stageAllocTimeoutState, TIMEOUT_NS_INFINITE, TIMEOUT_NS_10MIN,
            WsErr::SLAB_STAGE_LIST_INCONSISTENT,
            ,
            "workspace.slab.stage: Stage alloc traversal still waiting for cacheIndex=%u.",
            "workspace.slab.stage: Stage alloc traversal timeout 10min for cacheIndex=%u.",
            cacheIdx);
        temp = *static_cast<void**>(temp);
    }
}
```

---

### 2.2 修复伪超时（6处）

#### #5 `AllocThreadIdxForDav2201`（修复伪超时）

**文件**：`device_sche.h`
**超时**：20分钟退出，每10秒警告

```cpp
// 原代码（伪超时）：
TIMEOUT_CHECK_START();
while (__builtin_popcount(cpumask_.load(std::memory_order_acquire)) != static_cast<int>(devArgs->nrAicpu)) {
    TIMEOUT_CHECK_AND_RESET(
        TIMEOUT_ONE_MINUTE, ThreadErr::THREAD_CPU_ALLOC_FAILED,
        "#sche.thread.init: Thread alloc timeout over 1 min: threadIdx=%d, physicalCpu=%d.", curThreadIdx, cpu);
    sched_yield();
}

// 修复后：调用统一的WaitForCpuMaskReady
cpumask_.fetch_or(1 << cpu, std::memory_order_release);

int ret = WaitForCpuMaskReady(devArgs, cpu, curThreadIdx);
if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
    return ret;
}
```

---

#### #6-#7 `ScheWait`（修复伪超时）

**文件**：`device_sche.h`
**超时**：1分钟退出，每10秒警告

```cpp
// 原代码（伪超时）：
void ScheWait(DevAscendProgram* devProg)
{
    TIMEOUT_CHECK_START();
    while (unlikely(!devProg->runtimeDataRingBufferInited)) {
        RuntimeYield(0);
        TIMEOUT_CHECK_AND_RESET(
            TIMEOUT_ONE_MINUTE, SchedErr::RINGBUFFER_WAIT_TIMEOUT, "Sche wait ring buf init over 1 min.");
    }
    RuntimeDataRingBufferHead* ringBufferHead = devProg->GetRuntimeDataList();
    while (unlikely(ringBufferHead->Empty())) {
        RuntimeYield(0);
        TIMEOUT_CHECK_AND_RESET(
            TIMEOUT_ONE_MINUTE, SchedErr::RINGBUFFER_WAIT_TIMEOUT, "Sche wait ring buf data over 1 min.");
    }
}

// 修复后：
void ScheWait(DevAscendProgram* devProg)
{
    TimeoutState state1;
    
    while (unlikely(!devProg->runtimeDataRingBufferInited)) {
        RuntimeYield(0);
        
        __PYPTO_TIMEOUT_CHECK(state1, TIMEOUT_NS_1MIN, TIMEOUT_NS_10SEC,
            SchedErr::RINGBUFFER_WAIT_TIMEOUT,
            return,
            "#sche.wait: RingBuffer init still waiting.",
            "#sche.wait: RingBuffer init timeout.");
    }
    RuntimeDataRingBufferHead* ringBufferHead = devProg->GetRuntimeDataList();
    
    TimeoutState state2;
    
    while (unlikely(ringBufferHead->Empty())) {
        RuntimeYield(0);
        
        __PYPTO_TIMEOUT_CHECK(state2, TIMEOUT_NS_1MIN, TIMEOUT_NS_10SEC,
            SchedErr::RINGBUFFER_WAIT_TIMEOUT,
            return,
            "#sche.wait: RingBuffer data still waiting.",
            "#sche.wait: RingBuffer data timeout.");
    }
}
```

---

#### #8 `AllocNewTaskCtrl`（修复伪超时）

**文件**：`device_ctrl.h`
**超时**：1分钟退出，每10秒警告

```cpp
// 原代码（伪超时）：
int AllocNewTaskCtrl()
{
    uint32_t& taskCtrlIndex = devStartArgs_->devCtrlState.taskCtrlIndex;
    TIMEOUT_CHECK_START();
    while (true) {
        if (taskCtrlIndex == MAX_DEVICE_TASK_NUM)
            taskCtrlIndex = 0;
        if (!GetTaskCtrlInPool(taskCtrlIndex).IsNotFree()) {
            return taskCtrlIndex++;
        }
        taskCtrlIndex++;
        TIMEOUT_CHECK_AND_RESET(TIMEOUT_ONE_MINUTE, CtrlErr::CTRL_ALLOC_TIMEOUT, "Alloc new task ctrl over 1 min.");
    }
}

// 修复后：
int AllocNewTaskCtrl()
{
    uint32_t& taskCtrlIndex = devStartArgs_->devCtrlState.taskCtrlIndex;
    
    TimeoutState state;
    
    while (true) {
        if (taskCtrlIndex == MAX_DEVICE_TASK_NUM)
            taskCtrlIndex = 0;
        if (!GetTaskCtrlInPool(taskCtrlIndex).IsNotFree()) {
            return taskCtrlIndex++;
        }
        taskCtrlIndex++;
        
        __PYPTO_TIMEOUT_CHECK(state, TIMEOUT_NS_1MIN, TIMEOUT_NS_10SEC,
            CtrlErr::CTRL_ALLOC_TIMEOUT,
            return DEVICE_MACHINE_ERROR,
            "#ctrl.alloc: AllocNewTaskCtrl still waiting, taskCtrlIndex=%u.",
            "#ctrl.alloc: AllocNewTaskCtrl timeout, taskCtrlIndex=%u.",
            taskCtrlIndex);
    }
}
```

---

#### #9-#10 `SlabAlloc`（修复伪超时）

**文件**：`dev_workspace.h`
**超时**：外层20分钟退出/每2分钟警告，内层20分钟退出/每1分钟警告

```cpp
// 原代码（伪超时）：
WsAllocation SlabAlloc(uint32_t objSize, WsAicpuSlabMemType type)
{
    void* ptr = nullptr;
    SlabTryDynAddCache(type, objSize);
    do {
        if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
            ptr = metadataAllocators_.generalSlab.Alloc(ToUnderlying(type));
        } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
            ptr = metadataAllocators_.stitchSlab.Alloc(ToUnderlying(type));
        }
        if (ptr != nullptr) { break; }

        uint64_t ttlstart = GetCycles();
        while (!DeviceTaskMemTryRecycle()) {
            if (GetCycles() - ttlstart > TIMEOUT_CYCLES) {
                ttlstart = GetCycles();  // 重置计时器，继续等（伪超时）
                DEV_WARN("Waiting for device task finished for too long.");
            }
        };
    } while (true);

    WsAllocation allocation;
    allocation.ptr = reinterpret_cast<uintdevptr_t>(ptr);
    return allocation;
}

// 修复后：
WsAllocation SlabAlloc(uint32_t objSize, WsAicpuSlabMemType type)
{
    void* ptr = nullptr;
    SlabTryDynAddCache(type, objSize);
    
    TimeoutState state;
    
    do {
        if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
            ptr = metadataAllocators_.generalSlab.Alloc(ToUnderlying(type));
        } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
            ptr = metadataAllocators_.stitchSlab.Alloc(ToUnderlying(type));
        }
        if (ptr != nullptr) { break; }

        if (submmitTaskQueue_.IsEmpty()) {
            // ... 错误处理 ...
        }
        
        TimeoutState innerState;
        while (!DeviceTaskMemTryRecycle()) {
            __PYPTO_TIMEOUT_CHECK(innerState, TIMEOUT_NS_20MIN, TIMEOUT_NS_1MIN,
                WsErr::SLAB_ADD_CACHE_FAILED,
                ,
                "#workspace.alloc.inner_timeout: Inner recycle still waiting over 1min, type=%u, objSize=%u.",
                "#workspace.alloc.inner_timeout: Inner recycle timeout, type=%u, objSize=%u.",
                ToUnderlying(type), objSize);
        };
        
        __PYPTO_TIMEOUT_CHECK(state, TIMEOUT_NS_20MIN, TIMEOUT_NS_2MIN,
            WsErr::SLAB_ADD_CACHE_FAILED,
            { WsAllocation emptyAlloc; emptyAlloc.ptr = 0; return emptyAlloc; },
            "#workspace.alloc: SlabAlloc still waiting, type=%u, objSize=%u.",
            "#workspace.alloc: SlabAlloc timeout, type=%u, objSize=%u.",
            ToUnderlying(type), objSize);
    } while (true);

    WsAllocation allocation;
    allocation.ptr = reinterpret_cast<uintdevptr_t>(ptr);
    return allocation;
}
```

---

### 2.3 修改已有超时（7处）

#### #11 `AllocThreadIdxForDav3510`（修改）

**文件**：`device_sche.h`
**超时**：20分钟退出，每10秒警告

```cpp
// 修改后：调用统一的WaitForCpuMaskReady
int AllocThreadIdxForDav3510(DeviceArgs* devArgs, int cpu, int& curThreadIdx, std::atomic<int>& threadIdx)
{
    // ... threadIdx分配逻辑 ...
    
    cpumask_.fetch_or(1 << cpu, std::memory_order_release);
    
    int ret = WaitForCpuMaskReady(devArgs, cpu, curThreadIdx);
    if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
        return ret;
    }

    DEV_INFO("Thread alloc success: physicalCpu=%d, threadIdx=%d.", cpu, curThreadIdx);
    return DEVICE_MACHINE_OK;
}
```

---

#### #12 `ProcessTaskLoop`（修改）

**文件**：`aicore_manager.h`
**超时**：10秒退出，每1秒警告

```cpp
// 原代码：
uint64_t start = GetCycles();
while (!deviceTaskCtx->IsCoreTaskSendFinish()) {
    int32_t ret = RunCoreTask<true>(deviceTaskCtx);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return ret;
    }
    if (deviceTaskCtx->IsParallel()) {
        deviceTaskCtx->SyncAllSchCoreTaskSent();
        isFinish = deviceTaskCtx->IsCoreTaskSendFinish();
        return ret;
    }
    DEV_IF_DEVICE {
        if (GetCycles() - start > TIMEOUT_CYCLES) {
            return DEVICE_MACHINE_TIMEOUT_CORETASK;
        }
    }
}

// 修改后：
TimeoutState timeoutState;
while (!deviceTaskCtx->IsCoreTaskSendFinish()) {
    int32_t ret = RunCoreTask<true>(deviceTaskCtx);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return ret;
    }
    if (deviceTaskCtx->IsParallel()) {
        deviceTaskCtx->SyncAllSchCoreTaskSent();
        isFinish = deviceTaskCtx->IsCoreTaskSendFinish();
        return ret;
    }
    DEV_IF_DEVICE {
        __PYPTO_TIMEOUT_CHECK(timeoutState, TIMEOUT_NS_10SEC, NSEC_PER_SEC,
            SchedErr::TASK_WAIT_TIMEOUT,
            return DEVICE_MACHINE_TIMEOUT_CORETASK,
            "#sche.task.loop: ProcessTaskLoop still waiting.",
            "#sche.task.loop: ProcessTaskLoop timeout.");
    }
}
```

---

#### #13 `RunManager`（修改）

**文件**：`aicore_manager.h`
**超时**：10秒退出，每1秒警告

```cpp
// 原代码：
uint64_t start_cycles = GetCycles();
while (ret == 0) {
    FillParallelDevtaskCtx();
    ret = ProcessParallelDevTasks();
    if (ret != DEVICE_MACHINE_OK)
        break;
    // ... 其他逻辑 ...
    DEV_IF_DEVICE {
        if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
            ret = ToUnderlying(SchedErr::SCH_PARALLEL_DEVTASK_TIMEOUT);
            DEV_ERROR(ret, "Schedule prallel devtask timeout, dequeueFinish=%d.", taskCtrlDequeFinish);
            break;
        }
    }
}

// 修改后：
TimeoutState timeoutState;
while (ret == 0) {
    FillParallelDevtaskCtx();
    ret = ProcessParallelDevTasks();
    if (ret != DEVICE_MACHINE_OK)
        break;
    // ... 其他逻辑 ...
    DEV_IF_DEVICE {
        __PYPTO_TIMEOUT_CHECK(timeoutState, TIMEOUT_NS_10SEC, NSEC_PER_SEC,
            SchedErr::SCH_PARALLEL_DEVTASK_TIMEOUT,
            { ret = ToUnderlying(SchedErr::SCH_PARALLEL_DEVTASK_TIMEOUT); break; },
            "#sche.parallel.devtask: Schedule parallel devtask still waiting, dequeueFinish=%d.",
            "#sche.parallel.devtask: Schedule parallel devtask timeout, dequeueFinish=%d.",
            taskCtrlDequeFinish);
    }
}
```

---

#### #14 `SyncAicoreDevTaskFinish`（修改）

**文件**：`aicore_manager.h`
**超时**：10秒退出，每1秒警告

```cpp
// 原代码：
uint64_t start_cycles = GetCycles();
while (devTaskCtx->coreFinishedNum < mngCoreNum) {
    // ... 检查每个核心是否完成 ...
    DEV_IF_DEVICE {
        if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
            DumpDfxWhenCoreNotStop(devTaskCtx);
            DEV_ERROR(SchedErr::TASK_WAIT_TIMEOUT,
                "#sche.task.end.sync.timeout: SyncAicoreDevTaskFinish timeout notstopNum=%u.",
                mngCoreNum - devTaskCtx->coreFinishedNum);
            return DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH;
        }
    }
}

// 修改后：
TimeoutState timeoutState;
while (devTaskCtx->coreFinishedNum < mngCoreNum) {
    // ... 检查每个核心是否完成 ...
    DEV_IF_DEVICE {
        __PYPTO_TIMEOUT_CHECK(timeoutState, TIMEOUT_NS_10SEC, NSEC_PER_SEC,
            SchedErr::TASK_WAIT_TIMEOUT,
            { DumpDfxWhenCoreNotStop(devTaskCtx); return DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH; },
            "#sche.task.end.sync: SyncAicoreDevTaskFinish still waiting, notstopNum=%u.",
            "#sche.task.end.sync: SyncAicoreDevTaskFinish timeout, notstopNum=%u.",
            mngCoreNum - devTaskCtx->coreFinishedNum);
    }
}
```

---

#### #15 `HandShakeByGmWithPreSendTask`（修改）

**文件**：`aicore_manager.h`
**超时**：16分钟退出，每1分钟警告

```cpp
// 原代码：
uint64_t start_cycles = GetCycles();
while (handShakeNum < mngAicoreNum) {
    // ... 尝试与每个AiCore握手 ...
    if (unlikely(GetCycles() - start_cycles > HAND_SHAKE_TIMEOUT)) {
        DumpAicoreStatusWhenTimeout(handFlag);
        DEV_ERROR(SchedErr::HANDSHAKE_TIMEOUT,
            "#sche.handshake.timeout: HandShakeByGmWithPreSendTask timeout notHandshakeNum=%d.",
            mngAicoreNum - handShakeNum);
        return DEVICE_MACHINE_ERROR;
    }
}

// 修改后：
TimeoutState timeoutState;
while (handShakeNum < mngAicoreNum) {
    // ... 尝试与每个AiCore握手 ...
    __PYPTO_TIMEOUT_CHECK(timeoutState, HAND_SHAKE_TIMEOUT_NS, TIMEOUT_NS_1MIN,
        SchedErr::HANDSHAKE_TIMEOUT,
        { DumpAicoreStatusWhenTimeout(handFlag); return DEVICE_MACHINE_ERROR; },
        "#sche.handshake: HandShakeByGmWithPreSendTask still waiting, notHandshakeNum=%d.",
        "#sche.handshake: HandShakeByGmWithPreSendTask timeout, notHandshakeNum=%d.",
        mngAicoreNum - handShakeNum);
}
```

---

#### #16 `SyncAicpuTaskFinish`（修改）

**文件**：`aicpu_task_manager.h`
**超时**：10秒退出，每1秒警告

```cpp
// 原代码：
int64_t start_cycles = GetCycles();
while (!Finished()) {
    auto ret = TaskPoll(aiCoreManager);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return ret;
    }
    if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
        DEV_ERROR(DistributedErrorCode::AICPU_TASK_TIMEOUT,
            "#sche.task.end.sync.timeout: SyncAicpuTaskFinish timeout.");
        return DEVICE_MACHINE_TIMEOUT_SYNC_AICPU_FINISH;
    }
}

// 修改后：
TimeoutState timeoutState;
while (!Finished()) {
    auto ret = TaskPoll(aiCoreManager);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return ret;
    }
    __PYPTO_TIMEOUT_CHECK(timeoutState, TIMEOUT_NS_10SEC, NSEC_PER_SEC,
        DistributedErrorCode::AICPU_TASK_TIMEOUT,
        return ToUnderlying(DistributedErrorCode::AICPU_TASK_TIMEOUT),
        "#sche.task.end.sync: SyncAicpuTaskFinish still waiting.",
        "#sche.task.end.sync: SyncAicpuTaskFinish timeout.");
}
```

---

#### #17 `GetMetrics`（修改）

**文件**：`aicore_hal.h`
**超时**：10秒退出，每1秒警告

```cpp
// 原代码：
uint64_t cycles_start = GetCycles();
while (metric->isMetricStop != 1) {
    if (GetCycles() - cycles_start > PROF_DUMP_TIMEOUT_CYCLES) {
        DEV_ERROR(DevCommonErr::NULLPTR, "#sche.prof.aicore.wait_finish: wait metrics done timeout !!!.");
        return nullptr;
    }
};

// 修改后：
TimeoutState timeoutState;
volatile int stopFlag = metric->isMetricStop;
while (stopFlag != 1) {
    __PYPTO_TIMEOUT_CHECK(timeoutState, TIMEOUT_NS_10SEC, NSEC_PER_SEC,
        DevCommonErr::NULLPTR,
        return nullptr,
        "#sche.prof.aicore.wait_finish: wait metrics done still waiting.",
        "#sche.prof.aicore.wait_finish: wait metrics done timeout.");
    stopFlag = metric->isMetricStop;
}
```

---

### 2.4 保持不变（7处）

以下代码保持不变，使用原代码：

#### #18 `GetNextLeafTask`

**文件**：`aicore_entry.h`
**超时**：250ms退出

```cpp
uint64_t t0 = get_sys_cnt();
uint64_t loop_count = 0;
do {
    regValue = GetDataMainBase();
    nextLowIdx = regValue & 0xFFFFFFFF;
    nextLowIdx -= 1;
    ++loop_count;
    if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > AICORE_LEAF_TASK_WAIT_TIMEOUT)) {
        return AICORE_TASK_STOP;
    }
} while (nextLowIdx == lastTaskIdx);
```

---

#### #19 `GetRegHighValue`

**文件**：`aicore_entry.h`
**超时**：250ms退出

```cpp
uint64_t t0 = get_sys_cnt();
uint64_t loop_count = 0;
do {
    regValue = GetDataMainBase();
    nextHighIdx = (regValue >> 32) & 0xFFFFFFFF;
    ++loop_count;
    if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > AICORE_LEAF_TASK_WAIT_TIMEOUT)) {
        return AICORE_TASK_STOP;
    }
} while (nextHighIdx == lastTaskIdx);
```

---

#### #20 `GetCoreFuncionData`

**文件**：`aicore_entry.h`
**超时**：250ms退出

```cpp
uint64_t t0 = get_sys_cnt();
uint64_t loop_count = 0;
while (true) {
    // ... 等待并行任务数据 ...
    if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > AICORE_DEVICE_TASK_WAIT_TIME_OUT)) {
        SetStatus(args, STAGE_GET_PARALLEL_DEVTASK_TIMEOUT);
        return nullptr;
    }
    // ... dcci等待element指针 ...
    do {
        dcci(&parallelDevTask->elements[idx], SINGLE_CACHE_LINE, CACHELINE_OUT);
        elemPtr = parallelDevTask->elements[idx];
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > AICORE_DEVICE_TASK_WAIT_TIME_OUT)) {
            SetStatus(args, STAGE_GET_PARALLEL_DEVTASK_TIMEOUT);
            return nullptr;
        }
    } while (elemPtr == 0);
}
```

---

#### #21 `AiCoreEntry` 外层循环

**文件**：`aicore_entry.h`
**超时**：3秒退出

```cpp
uint64_t t0 = get_sys_cnt();
uint64_t loop_count = 0;
while (true) {
    ++loop_count;
    if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > AICORE_DEVICE_TASK_RUN_TIMEOUT)) {
        break;
    }
    // ... 获取任务、执行 ...
}
```

---

#### #22 `AiCoreEntry` 内层循环

**文件**：`aicore_entry.h`
**超时**：3秒退出

```cpp
uint64_t t1 = get_sys_cnt();
uint64_t inner_loop_count = 0;
while (true) {
    ++inner_loop_count;
    if ((inner_loop_count % 1000 == 0) && (get_sys_cnt() - t1 > AICORE_LEAF_TASK_RUN_TIMEOUT)) {
        break;
    }
    // ... 获取下一个叶任务 ...
}
```

---

#### #23 `WaitWaveSignal`

**文件**：`aicore_entry.h`
**超时**：50ms退出

```cpp
uint64_t t2 = get_sys_cnt();
while (true) {
    dcci(waveBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
    if (*waveBuffer == AICORE_SAY_GOODBYE) { return; }
    if ((get_sys_cnt() - t2 > AICORE_GM_DCCI_TIMEOUT)) { return; }
}
```

---

#### #24 `RefreshParallelDevTaskByModifyFlag`

**文件**：`aicore_entry.h`
**超时**：250ms退出

```cpp
uint64_t t3 = get_sys_cnt();
uint64_t loop_count = 0;
while (true) {
    dcci(modifyFlagBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
    if (*modifyFlagBuffer == 0) { break; }
    ++loop_count;
    if ((loop_count % 1000 == 0) && (get_sys_cnt() - t3 > AICORE_DEVICE_TASK_WAIT_TIME_OUT)) {
        return AICORE_TASK_STOP;
    }
}
```

---

### 2.5 删除（2处）

#### #25 `WaitFinQueue`（删除）

**文件**：`aicore_hal.h`
**原因**：void返回，调用方无法感知超时，无实际使用

```cpp
// 原代码（已删除）：
inline void WaitFinQueue(int coreStart, int coreEnd, uint64_t val)
{
    for (int idx = coreStart; idx < coreEnd; idx++) {
        uint64_t startCycle = GetCycles();
        while (*finishRegQueues_[GetPhyIdByBlockId(idx)] != val) {
            if (GetCycles() - startCycle > TIMEOUT_CYCLES) {
                DEV_ERROR(SchedErr::TASK_WAIT_TIMEOUT,
                    "#sche.aicore.wait_finish: CoreId=%d cannot get finish Flag", idx);
                return;  // void return
            }
        }
    }
}
```

---

#### #26 `device_channel.h`（删除）

**文件**：`device_channel.h`
**原因**：阻塞接口已不使用，删除整个文件（218行）

```cpp
// 原代码（已删除）：
class DeviceTaskSender {
    int send(int64_t taskId, int64_t taskData) {
        int slotId = INVALID_SLOT_ID;
        while (!trySend(taskId, taskData, slotId)) {
            cpuRelax();  // 阻塞等待
        }
        return slotId;
    }
    
    void sync(int slotId) {
        while (!trySync(slotId)) {
            cpuRelax();  // 阻塞等待
        }
    }
};

class DeviceTaskReceiver {
    int recv(int64_t& taskId, int64_t& taskData) {
        int slot = INVALID_SLOT_ID;
        while (!TryRecv(taskId, taskData, slot)) {
            cpuRelax();  // 阻塞等待
        }
        return slot;
    }
};
```

---

## 三、超时阈值汇总表

| 函数类别 | 函数名 | 超时阈值 | 警告间隔 |
|---------|--------|---------|---------|
| 线程初始化 | WaitForCpuMaskReady | 20分钟 | 10秒 |
| 任务同步 | ProcessTaskLoop | 10秒 | 1秒 |
| 任务同步 | RunManager | 10秒 | 1秒 |
| 任务同步 | SyncAicoreDevTaskFinish | 10秒 | 1秒 |
| 任务同步 | SyncAicpuTaskFinish | 10秒 | 1秒 |
| 任务同步 | GetMetrics | 10秒 | 1秒 |
| AiCore握手 | HandShakeByGmWithPreSendTask | 16分钟 | 1分钟 |
| 通用等待 | ScheWait | 1分钟 | 10秒 |
| 通用等待 | AllocNewTaskCtrl | 1分钟 | 10秒 |
| 内存分配 | SlabAlloc (外层) | 20分钟 | 2分钟 |
| 内存分配 | SlabAlloc (内层) | 20分钟 | 1分钟 |
| 内存分配 | SlabStageAllocMemSubmmit | 20分钟 | 1分钟 |
| 正常阻塞 | AllocateWait | 永不退出 | 10分钟 |
| 正常阻塞 | ProcessKeepTailCase | 永不退出 | 10分钟 |