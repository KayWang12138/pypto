### Background（背景信息）

MACHINE 是 PyPTO 设备侧执行引擎的核心组件，运行在 AICPU 上，负责 AiCore 任务调度、Workspace 管理、设备间通信、多线程资源分配等。代码路径为 `framework/src/machine/`（AICPU 侧调度逻辑，119 个 .h/.cpp 文件）和 `framework/src/interface/machine/`（AiCore 入口 + Host 侧执行器，16 个 .h/.cpp 文件）。

执行路径中存在大量轮询等待和同步操作（`while (!condition)`、CAS 自旋锁、SPSC 队列阻塞等）。部分位置存在设计缺陷（伪超时、裸 mutex 无 RAII），可能导致线程永久阻塞（卡死），进程无法退出。

### Origin（信息来源）

内部代码审计 + 线上卡死问题反馈 + 补充排查遗漏文件

### Benefit / Necessity（价值/作用）

消除设计缺陷导致的无限阻塞，修复伪超时机制，规范 mutex 使用，为阻塞等待接口提供超时版本供调用者选择。

### Design（设计方案）

> **排查范围**：`framework/src/machine/`（119 个文件）+ `framework/src/interface/machine/`（16 个文件）= **共 135 个 .h/.cpp 文件**，**114 个 while 循环**逐个审查

以下每个位置均基于实际源码逐行验证。每个位置展示：当前代码 → 问题 → 修复后代码。

---

# 第一部分：当前已有超时检测的代码（20 处）及问题分析

这些位置已经有某种形式的超时检测，但存在阈值不统一、用法不规范、语义混乱等问题。

按代码层级分组：AICPU 侧（`framework/src/machine/`）和 AiCore 入口侧（`framework/src/interface/machine/device/`）。

---

## AICPU 侧已有超时（12 处）

### 已有超时 1：`device_sche.h` L189-198 `AllocThreadIdxForDav3510()`

**当前代码：**

```cpp
cpumask_.fetch_or(1 << cpu, std::memory_order_release);
uint64_t start = GetCycles();
while (__builtin_popcount(cpumask_.load(std::memory_order_acquire)) != static_cast<int>(devArgs->nrAicpu)) {
    if (GetCycles() - start > TIMEOUT_CYCLES) {
        DEV_ERROR(ThreadErr::THREAD_CPU_ALLOC_FAILED,
                  "#sche.thread.init: Thread alloc timeout: threadIdx=%d, physicalCpu=%d.", curThreadIdx, cpu);
        return DEVICE_MACHINE_ERROR;
    }
    sched_yield();
}
```

**使用的超时常量：** `TIMEOUT_CYCLES`（aarch64 上 500M cycles ≈ 250ms @2GHz）

**问题：** 超时阈值过小。这是线程初始化操作，涉及所有 AICPU 线程同步，250ms 可能不够。

---

### 已有超时 2：`aicore_manager.h` L321-339 `ProcessTaskLoop()`

**当前代码：**

```cpp
uint64_t start = GetCycles();
while (!deviceTaskCtx->IsCoreTaskSendFinish()) {
    int32_t ret = RunCoreTask<true>(deviceTaskCtx);
    if (unlikely(ret != DEVICE_MACHINE_OK)) { return ret; }
    if (deviceTaskCtx->IsParallel()) { /* ... return */ }

    DEV_IF_DEVICE
    {
        if (GetCycles() - start > TIMEOUT_CYCLES) {
            return DEVICE_MACHINE_TIMEOUT_CORETASK;
        }
    }
}
```

**使用的超时常量：** `TIMEOUT_CYCLES`（250ms）

**问题：**
1. 阈值过小，复杂算子核心任务循环可能需要更长时间
2. 被 `DEV_IF_DEVICE` 包裹——sim 模式和 ESL 模型下**完全不生效**
3. 没有打日志，直接 return，难以定位问题

---

### 已有超时 3：`aicore_manager.h` L449-476 `RunManager()` 主循环

**当前代码：**

```cpp
uint64_t start_cycles = GetCycles();
while (ret == 0) {
    FillParallelDevtaskCtx();
    ret = ProcessParallelDevTasks();
    if (ret != DEVICE_MACHINE_OK) break;
    // ...
    DEV_IF_DEVICE {
        if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
            ret = ToUnderlying(SchedErr::SCH_PARALLEL_DEVTASK_TIMEOUT);
            DEV_ERROR(ret, "Schedule prallel devtask timeout, dequeueFinish=%d.", taskCtrlDequeFinish);
            break;
        }
    }
}
```

**使用的超时常量：** `TIMEOUT_CYCLES`（250ms）

**问题：**
1. `start_cycles` 在循环外初始化一次，永不再重置——测的是"整个 RunManager 循环的总执行时间"，正常运行很多小任务累计超过 250ms 也会误触发
2. 被 `DEV_IF_DEVICE` 包裹
3. 250ms 对于整个调度循环生命周期来说可能过短

---

### 已有超时 4：`aicore_manager.h` L749-793 `SyncAicoreDevTaskFinish()`

**当前代码：**

```cpp
uint64_t start_cycles = GetCycles();
while (devTaskCtx->coreFinishedNum < mngCoreNum) {
    // ... 检查每个 AiCore 是否完成 ...
    DEV_IF_DEVICE
    {
        if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
            DumpDfxWhenCoreNotStop(devTaskCtx);
            DEV_ERROR(SchedErr::TASK_WAIT_TIMEOUT,
                      "#sche.task.end.sync.timeout: SyncAicoreDevTaskFinish timeout notstopNum=%u.",
                      mngCoreNum - devTaskCtx->coreFinishedNum);
            return DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH;
        }
    }
}
```

**使用的超时常量：** `TIMEOUT_CYCLES`（250ms）

**问题：** 同样 `start_cycles` 不重置；被 `DEV_IF_DEVICE` 包裹；等待所有 AiCore 核心完成，250ms 明显不足。

---

### 已有超时 5：`aicore_manager.h` L1857-1931 `HandShakeByGmWithPreSendTask()`

**当前代码：**

```cpp
uint64_t start_cycles = GetCycles();
while (handShakeNum < mngAicoreNum) {
    // ... 尝试与每个 AiCore 握手 ...
    if (unlikely(GetCycles() - start_cycles > HAND_SHAKE_TIMEOUT)) {
        DumpAicoreStatusWhenTimeout(handFlag);
        DEV_ERROR(SchedErr::HANDSHAKE_TIMEOUT,
                  "#sche.handshake.timeout: timeout notHandshakeNum=%d.", mngAicoreNum - handShakeNum);
        return DEVICE_MACHINE_ERROR;
    }
}
```

**使用的超时常量：** `HAND_SHAKE_TIMEOUT`（48000000000 ns = 48 秒）

**状态：** ✅ 规范。整个 AICPU 侧方案中**唯一正确使用基于时间（ns）定义的超时常量**的位置。无 `DEV_IF_DEVICE` 包裹。可作为其他超时的参考模式。

---

### 已有超时 6：`aicore_hal.h` L243-255 `WaitFinQueue()`：void 返回

**当前代码：**

```cpp
inline void WaitFinQueue(int coreStart, int coreEnd, uint64_t val)
{
    for (int idx = coreStart; idx < coreEnd; idx++) {
        uint64_t startCycle = GetCycles();
        while (*finishRegQueues_[GetPhyIdByBlockId(idx)] != val) {
            if (GetCycles() - startCycle > TIMEOUT_CYCLES) {
                DEV_ERROR(SchedErr::TASK_WAIT_TIMEOUT,
                          "#sche.aicore.wait_finish: CoreId=%d cannot get finish Flag", idx);
                return;    // ← void return，调用方不知道超时了
            }
        }
    }
}
```

**问题：** 返回类型 `void`，调用方完全无法感知超时。每个 core 的 `startCycle` 独立重置（正确）。

---

### 已有超时 7：`aicore_hal.h` L341-347 `GetMetrics()` 等待 metrics

**当前代码：**

```cpp
uint64_t cycles_start = GetCycles();
while (metric->isMetricStop != 1) {
    if (GetCycles() - cycles_start > PROF_DUMP_TIMEOUT_CYCLES) {
        DEV_ERROR(DevCommonErr::NULLPTR, "#sche.prof.aicore.wait_finish: wait metrics done timeout !!!.");
        return nullptr;
    }
};
```

**问题：** `PROF_DUMP_TIMEOUT_CYCLES = TIMEOUT_CYCLES`（250ms），对性能数据采集偏短。其余写法规范。

---

### 已有超时 8：`aicpu_task_manager.h` L101-117 `SyncAicpuTaskFinish()`

**当前代码：**

```cpp
int64_t start_cycles = GetCycles();
while (!Finished()) {
    auto ret = TaskPoll(aiCoreManager);
    if (unlikely(ret != DEVICE_MACHINE_OK)) { return ret; }
    if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
        DEV_ERROR(DistributedErrorCode::AICPU_TASK_TIMEOUT,
                  "#sche.task.end.sync.timeout: SyncAicpuTaskFinish timeout.");
        return DEVICE_MACHINE_TIMEOUT_SYNC_AICPU_FINISH;
    }
}
```

**问题：** `start_cycles` 不重置，测累计时间。其余写法规范。

---

### 已有超时 9：`dev_workspace.h` L754-760 `SlabAlloc()` 内层等待：伪超时 ❌ 设计缺陷

**当前代码：**

```cpp
uint64_t ttlstart = GetCycles();
while (!DeviceTaskMemTryRecycle()) {
    if (GetCycles() - ttlstart > TIMEOUT_CYCLES) {
        ttlstart = GetCycles();       // ← 重置计时器，继续等
        DEV_WARN("Waiting for device task finished for too long.");
    }
};
```

**问题：** **伪超时**——超时后重置计时器继续循环，配合外层 `do {} while(true)` 永不退出。这是设计缺陷，需要修复。

---

### 已有超时 10：`device_sche.h` L208-214 `AllocThreadIdxForDav2201()`：`TIMEOUT_CHECK_AND_RESET` 伪超时 ❌ 设计缺陷

**当前代码：**

```cpp
cpumask_.fetch_or(1 << cpu, std::memory_order_release);
TIMEOUT_CHECK_START();
while (__builtin_popcount(cpumask_.load(std::memory_order_acquire)) != static_cast<int>(devArgs->nrAicpu)) {
    TIMEOUT_CHECK_AND_RESET(
        TIMEOUT_ONE_MINUTE, ThreadErr::THREAD_CPU_ALLOC_FAILED,
        "#sche.thread.init: Thread alloc timeout over 1 min: threadIdx=%d, physicalCpu=%d.", curThreadIdx, cpu);
    sched_yield();
}
```

**使用的超时常量：** `TIMEOUT_ONE_MINUTE`（3000000000 ns = 3 秒），通过 `TIMEOUT_CHECK_AND_RESET` 宏使用。

**问题：** 使用 `TIMEOUT_CHECK_AND_RESET` 宏（定义于 `device_utils.h` L293-299），该宏超时后**仅打印日志并重置 start 继续循环**，永远不会真正退出——是**伪超时**。与已有超时 1（`AllocThreadIdxForDav3510`）功能相同但超时策略截然不同。此外 `TIMEOUT_ONE_MINUTE` 实际值为 3 秒而非一分钟，命名具有误导性。

---

### 已有超时 11：`device_sche.h` L655-671 `ScheWait()`：两处 while 均为 `TIMEOUT_CHECK_AND_RESET` 伪超时 ❌ 设计缺陷

**当前代码：**

```cpp
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
```

**使用的超时常量：** `TIMEOUT_ONE_MINUTE`（3 秒），通过 `TIMEOUT_CHECK_AND_RESET` 宏使用。

**问题：** 两个 while 循环均使用 `TIMEOUT_CHECK_AND_RESET` 宏——伪超时，永远不退出。这是调度器启动关键路径：第一个 while 等待 ctrl 线程初始化环形缓冲区，第二个 while 等待数据就绪。若 ctrl 线程异常，调度线程永久阻塞。

---

### 已有超时 12：`device_ctrl.h` L60-73 `AllocNewTaskCtrl()`：`TIMEOUT_CHECK_AND_RESET` 伪超时 + 死循环 ❌ 设计缺陷

**当前代码：**

```cpp
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
```

**使用的超时常量：** `TIMEOUT_ONE_MINUTE`（3 秒），通过 `TIMEOUT_CHECK_AND_RESET` 宏使用。

**问题：** `while (true)` 死循环 + `TIMEOUT_CHECK_AND_RESET` 伪超时。当所有 `MAX_DEVICE_TASK_NUM`（1024）个 TaskCtrl 均被占用且无法释放时，该函数永远不会返回。

---

### `TIMEOUT_CHECK_AND_RESET` 宏的系统性问题 ❌ 设计缺陷

`device_utils.h` L291-299 中定义的两个宏：

```cpp
#define TIMEOUT_CHECK_START() uint64_t start = GetCycles()

#define TIMEOUT_CHECK_AND_RESET(timeout, ...)  \
    do {                                       \
        if (GetCycles() - start > (timeout)) { \
            DEV_ERROR(__VA_ARGS__);            \
            start = GetCycles();               \
        }                                      \
    } while (0)
```

**核心问题：** `TIMEOUT_CHECK_AND_RESET` 超时后重置 `start` 并继续循环，**永远不会导致循环退出**。所有使用该宏的位置（已有超时 10、11、12）实际上与无超时保护的死循环等效，只是多了周期性日志打印。

---

## AiCore 入口侧已有超时（8 处）

以下位于 `framework/src/interface/machine/device/tilefwk/aicore_entry.h`，运行在 AiCore 核心上（非 AICPU），超时常量以**纳秒**定义，使用 `get_sys_cnt()` 计时。

### AiCore 入口超时常量（`aicore_entry.h` L52-56）

```cpp
#define AICORE_DEVICE_TASK_RUN_TIMEOUT 3000000000   // 3 秒
#define AICORE_DEVICE_TASK_WAIT_TIME_OUT 250000000   // 250ms
#define AICORE_LEAF_TASK_RUN_TIMEOUT 3000000000      // 3 秒
#define AICORE_LEAF_TASK_WAIT_TIMEOUT 250000000      // 250ms
#define AICORE_GM_DCCI_TIMEOUT 50000000              // 50ms
```

这些常量与 AICPU 侧的 `TIMEOUT_CYCLES`（裸 cycles）是**完全不同的时间体系**。

---

### 已有超时 13：`aicore_entry.h` L114-123 `GetNextLeafTask()`

**当前代码：**

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

**状态：** ✅ 规范。每 1000 次迭代检查超时，超时返回 `AICORE_TASK_STOP`。

---

### 已有超时 14：`aicore_entry.h` L134-142 `GetRegHighValue()`

**当前代码：**

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

**状态：** ✅ 规范。同已有超时 13 的模式。

---

### 已有超时 15：`aicore_entry.h` L295-343 `GetCoreFuncionData()`

**当前代码（简化）：**

```cpp
uint64_t t0 = get_sys_cnt();
uint64_t loop_count = 0;
while (true) {
    // ... 等待并行任务数据 ...
    if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > AICORE_DEVICE_TASK_WAIT_TIME_OUT)) {
        SetStatus(args, STAGE_GET_PARALLEL_DEVTASK_TIMEOUT);
        return nullptr;
    }
    // ... dcci 等待 element 指针 ...
    do {
        dcci(&parallelDevTask->elements[idx], SINGLE_CACHE_LINE, CACHELINE_OUT);
        elemPtr = parallelDevTask->elements[idx];
        ++loop_count;
        if ((loop_count % 1000 == 0) && (get_sys_cnt() - t0 > AICORE_DEVICE_TASK_WAIT_TIME_OUT)) {
            SetStatus(args, STAGE_GET_PARALLEL_DEVTASK_TIMEOUT);
            return nullptr;
        }
    } while (elemPtr == 0);
    // ...
}
```

**状态：** ✅ 规范。多处超时检查，超时后设置状态并返回。

---

### 已有超时 16：`aicore_entry.h` L549-563 `AiCoreEntry()` 外层主循环

**当前代码：**

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

**状态：** ✅ 规范。超时 break 退出。

---

### 已有超时 17：`aicore_entry.h` L569-573 `AiCoreEntry()` 内层叶任务循环

**当前代码：**

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

**状态：** ✅ 规范。

---

### 已有超时 18：`aicore_entry.h` L446-454 `WaitWaveSignal()`

**当前代码：**

```cpp
uint64_t t2 = get_sys_cnt();
while (true) {
    dcci(waveBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
    if (*waveBuffer == AICORE_SAY_GOODBYE) { return; }
    if ((get_sys_cnt() - t2 > AICORE_GM_DCCI_TIMEOUT)) { return; }
}
```

**状态：** ✅ 规范。

---

### 已有超时 19：`aicore_entry.h` L466-477 `RefreshParallelDevTaskByModifyFlag()`

**当前代码：**

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

**状态：** ✅ 规范。dcci 等待新任务数据，超时返回 `AICORE_TASK_STOP`。

---

### 已有超时 20：`aikernel_runtime.h` L35-44 `WaitAicoreStart()`：超时后不设错误状态

**当前代码：**

```cpp
constexpr uint64_t SYNC_TIMEOUT = 48000000000;  // 48 秒

__always_inline void WaitAicoreStart([[maybe_unused]] npu::tile_fwk::DevStartArgsBase* startArgs)
{
#if defined(__aarch64__) && defined(__DEVICE__)
    uint64_t start = GetCycles();
    while (startArgs->syncFlag != 1) {
        if (GetCycles() - start > SYNC_TIMEOUT) {
            break;           // ← 超时后仅 break，不设任何错误状态
        }
    }
#endif
}
```

**问题：** 超时后 break 退出循环，但没有设置任何错误标志，调用方和 AICPU 侧都不知道同步失败。后续 AiCore 核心可能在未同步的状态下开始执行任务，导致未定义行为。

**修复后：**

```cpp
__always_inline int32_t WaitAicoreStart([[maybe_unused]] npu::tile_fwk::DevStartArgsBase* startArgs)
{
#if defined(__aarch64__) && defined(__DEVICE__)
    uint64_t start = GetCycles();
    while (startArgs->syncFlag != 1) {
        if (GetCycles() - start > SYNC_TIMEOUT) {
            return DEVICE_MACHINE_ERROR;
        }
    }
#endif
    return DEVICE_MACHINE_OK;
}
```

---

## 已有超时检测汇总问题

| 问题类型 | 涉及位置 | 影响 |
|---------|---------|------|
| **AICPU 侧阈值全部使用 `TIMEOUT_CYCLES`（250ms）** | 已有超时 1-8 | 不区分初始化/通用/长操作，一刀切 250ms |
| **`DEV_IF_DEVICE` 包裹** | 已有超时 2、3、4 | sim 模式下超时检测完全失效 |
| **`start` 不重置** | 已有超时 3、4、8 | 测的是累计时间，可能误触发 |
| **void 返回** | 已有超时 6 | 调用方无法感知超时 |
| **伪超时（重置继续）** | 已有超时 9、10、11、12 | ❌ 设计缺陷，永不退出 |
| **`TIMEOUT_ONE_MINUTE` 命名误导** | 已有超时 10、11、12 | 实际值 3 秒 |
| **超时后不设错误状态** | 已有超时 20 | AiCore 未同步就执行任务 |

### 两层超时体系对比

| 维度 | AICPU 侧（`framework/src/machine/`） | AiCore 入口侧（`interface/machine/device/`） |
|------|--------------------------------------|----------------------------------------------|
| 计时函数 | `GetCycles()` → cycles (aarch64) / ns (x86) | `get_sys_cnt()` → 系统计数器 cycles |
| 超时单位 | 裸 cycles（不一致） | 纳秒（一致） |
| 超时常量 | `TIMEOUT_CYCLES=500M` / `HAND_SHAKE_TIMEOUT=48s` / `TIMEOUT_ONE_MINUTE=3s` | `AICORE_*_TIMEOUT` 全部以纳秒定义 |
| 超时模式 | 混乱：伪超时、`DEV_IF_DEVICE`、void 返回 | 统一：`loop_count % 1000` + 返回错误码 |
| 规范程度 | ❌ 不规范（部分位置有设计缺陷） | ✅ 规范（可作为 AICPU 侧改造参考） |

---

# 第二部分：设计缺陷与阻塞等待设计分析

按问题性质分为两类：
- **❌ 设计缺陷**：代码逻辑存在明显错误（伪超时、裸 mutex 无 RAII），需要修复
- **✅ 阻塞等待设计**：设计意图明确的阻塞接口，正常情况下等待时间极短，仅在异常场景下可能卡死

---

## ❌ 设计缺陷（需要修复）

---

### 缺陷 1：`SlabAlloc()` 外层 `do {} while(true)` + 内层伪超时重置

**文件：** `dev_workspace.h` L728-766

**当前代码：**

```cpp
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
        if (ptr != nullptr) { break; }              // ← 唯一正常退出

        uint64_t ttlstart = GetCycles();
        while (!DeviceTaskMemTryRecycle()) {
            if (GetCycles() - ttlstart > TIMEOUT_CYCLES) {
                ttlstart = GetCycles();              // ← 重置计时器，继续等
                DEV_WARN("Waiting for device task finished for too long.");
            }
        };
    } while (true);                                 // ← 永不退出

    WsAllocation allocation;
    allocation.ptr = reinterpret_cast<uintdevptr_t>(ptr);
    return allocation;
}
```

**问题：** 外层 `do {} while (true)` 只有一个退出点——分配成功。内层超时后重置计时器继续循环，配合外层形成**双层死循环结构**，不存在任何失败退出路径。

**修复后：**

```cpp
constexpr uint32_t MAX_SLAB_ALLOC_RETRY = 100;

WsAllocation SlabAlloc(uint32_t objSize, WsAicpuSlabMemType type)
{
    void* ptr = nullptr;
    SlabTryDynAddCache(type, objSize);
    uint32_t retryCount = 0;
    do {
        if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
            ptr = metadataAllocators_.generalSlab.Alloc(ToUnderlying(type));
        } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
            ptr = metadataAllocators_.stitchSlab.Alloc(ToUnderlying(type));
        }
        if (ptr != nullptr) { break; }

        uint64_t ttlstart = GetCycles();
        while (!DeviceTaskMemTryRecycle()) {
            if (GetCycles() - ttlstart > TIMEOUT_CYCLES_NORMAL) {
                break;                              // ← 跳出内层，不再重置
            }
        };

        if (++retryCount > MAX_SLAB_ALLOC_RETRY) {
            DEV_ERROR(WsErr::SLAB_ADD_CACHE_FAILED,
                      "#workspace.alloc.timeout: SlabAlloc retry exceeded %u.", MAX_SLAB_ALLOC_RETRY);
            return {0};                             // ← 返回空分配
        }
    } while (true);

    WsAllocation allocation;
    allocation.ptr = reinterpret_cast<uintdevptr_t>(ptr);
    return allocation;
}
```

---

### 缺陷 2：裸 `mutex_.lock()` 无 RAII（2 处）

#### 位置 2.1：`device_sche.h` L425 `RunUnifiedCtrlInit()`

**当前代码：**

```cpp
int RunUnifiedCtrlInit(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry)
{
    int ret = DEVICE_MACHINE_OK;
    mutex_.lock();
    if (!initCtrl_.load()) {
        initCtrl_.store(true);
        ret = RunCtrlInitNoLock(kargs, entry);
    }
    mutex_.unlock();
    return ret;
}
```

**问题：** 裸 `mutex_.lock()` 无 RAII，若 `RunCtrlInitNoLock` 抛异常或内部卡死，`mutex_.unlock()` 永不执行。

**修复后：**

```cpp
int RunUnifiedCtrlInit(DeviceKernelArgs* kargs, const KernelCtrlEntry& entry)
{
    std::lock_guard<std::mutex> lock(mutex_);  // RAII 自动释放
    if (!initCtrl_.load()) {
        initCtrl_.store(true);
        return RunCtrlInitNoLock(kargs, entry);
    }
    return DEVICE_MACHINE_OK;
}
```

---

#### 位置 2.2：`device_perf.h` L58 `allocRecord()`

**当前代码：**

```cpp
Record* allocRecord(int tid)
{
    if (trunk_[tid] == nullptr || trunk_[tid]->Full()) {
        trunk_[tid] = new Trunk;
        mutex_.lock();
        trunks_.push_back(trunk_[tid]);  // ← 可能抛 bad_alloc
        mutex_.unlock();
    }
    return trunk_[tid]->Alloc();
}
```

**问题：** `push_back` 可能抛出 `std::bad_alloc`，此时 `mutex_.unlock()` 永不执行。

**修复后：**

```cpp
Record* allocRecord(int tid)
{
    if (trunk_[tid] == nullptr || trunk_[tid]->Full()) {
        trunk_[tid] = new Trunk;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            trunks_.push_back(trunk_[tid]);
        }
    }
    return trunk_[tid]->Alloc();
}
```

---

## ✅ 阻塞等待设计（正常设计，建议提供超时版本）

以下位置是**阻塞式等待接口**，设计意图明确：等待某个条件满足后返回。正常情况下等待时间极短（微秒~毫秒级），仅在异常场景（对端线程异常退出或硬件挂死）下才会卡死。建议添加带超时的版本供需要超时保护的调用者选择，保留现有阻塞接口不变。

---

### 阻塞接口 1：`spsc_queue.h` `Enqueue()` / `Dequeue()`

> **设计意图说明**：这是 SPSC 队列的**阻塞式入队/出队接口**，设计上与 `BlockingQueue.put()/take()` 类似。队列同时提供了非阻塞版本 `TryEnqueue()`/`TryDequeue()`，调用者可根据场景选择。正常情况下生产者-消费者协作迅速完成等待，等待时间极短。

#### L27-31 `Enqueue()`：阻塞式入队（队列满时等待）

```cpp
inline void Enqueue(const T& val)
{
    while (!TryEnqueue(val))
        ;
}
```

**异常卡死条件：** 消费者线程异常退出不再消费 → 队列持续满 → 生产者线程永久阻塞。

**建议优化：** 添加 `EnqueueWithTimeout()` 版本供调用者选择。

---

#### L45-51 `Dequeue()`：阻塞式出队（队列空时等待）

```cpp
inline T Dequeue()
{
    T val;
    while (!TryDequeue(val))
        ;
    return val;
}
```

**异常卡死条件：** 生产者线程异常退出不再生产 → 队列持续空 → 消费者线程永久阻塞。

**建议优化：** 添加 `DequeueWithTimeout()` 版本供调用者选择。

---

### 阻塞接口 2：`device_channel.h` `send()` / `sync()` / `recv()`

> **设计意图说明**：这是 device_channel 的**阻塞式任务通信接口**，设计上与阻塞 I/O 类似。channel 同时提供了非阻塞版本 `trySend()`/`trySync()`/`TryRecv()`，调用者可根据场景选择。正常情况下对端（AiCore 硬件）迅速响应，等待时间极短。

#### L78-85 `DeviceTaskSender::send()`：阻塞式发送（slot 全忙时等待）

```cpp
int send(int64_t taskId, int64_t taskData)
{
    int slotId = INVALID_SLOT_ID;
    while (!trySend(taskId, taskData, slotId)) {
        cpuRelax();
    }
    return slotId;
}
```

**异常卡死条件：** AiCore 核心异常未消费 slot → slot 持续全忙 → 发送方永久阻塞。

**建议优化：** 添加 `sendWithTimeout()` 版本供调用者选择。

---

#### L105-114 `DeviceTaskSender::sync()`：阻塞式同步（等待任务完成）

```cpp
void sync(int slotId)
{
    while (!trySync(slotId)) {
        cpuRelax();
    }
    // ...
}
```

**异常卡死条件：** AiCore 核心异常不返回 FIN → sync 永久阻塞。

**建议优化：** 添加 `syncWithTimeout()` 版本供调用者选择。

---

#### L183-190 `DeviceTaskReceiver::recv()`：阻塞式接收（等待任务到达）

```cpp
int recv(int64_t& taskId, int64_t& taskData)
{
    int slot = INVALID_SLOT_ID;
    while (!TryRecv(taskId, taskData, slot)) {
        cpuRelax();
    }
    return slot;
}
```

**异常卡死条件：** 发送方异常不再发送任务 → recv 永久阻塞。

**建议优化：** 添加 `recvWithTimeout()` 版本供调用者选择。

---

### 阻塞接口 3：`dev_start_args.h` `AllocateWait()`

> **设计意图说明**：这是 Ring buffer 的**阻塞式等待接口**，设计上等待 ring buffer 出现空位。正常情况下消费者线程持续消费，空位迅速出现，等待时间极短。仅在异常场景（消费者线程异常退出）下才会卡死。

#### L171-176 `AllocateWait()`：Ring buffer 满时等待

```cpp
void AllocateWait()
{
    while (Full()) {
        RuntimeYield();
    }
}
```

**异常卡死条件：** 消费者线程异常退出不再释放槽位 → ring buffer 持续满 → 等待方永久阻塞。

**建议优化：** 添加 `AllocateWaitWithTimeout()` 版本供调用者选择。

---

### 阻塞接口 4：`device_task.h` `Finish()` runcnt 等待

> **设计意图说明**：这是多线程同步完成逻辑，`runcnt` 是原子计数器记录还有多少线程需要完成该任务。当 `syncWait=true` 时，本线程等待其他线程全部完成。正常情况下其他线程会迅速执行 `Finish()` 减少计数，等待时间极短。

#### L169-187 `Finish()` 中 runcnt 等待

```cpp
bool Finish(bool syncWait)
{
    if (runcnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (syncWait) { SetFree(); }
        return true;
    } else {
        if (syncWait) {
            while (runcnt.load(std::memory_order_acquire) != 0) {}  // ← 多线程同步等待
            return true;
        }
        // ...
    }
}
```

**异常卡死条件：** 其他线程异常退出未执行 `Finish()` → 计数器永远不为 0 → 等待线程永久阻塞。

**建议优化：** 添加超时检测。

---

### 阻塞接口 5：`dev_workspace.h` `SlabStageAllocMemSubmmit()`

> **设计意图说明**：这是 Workspace 分配流程中的**阻塞式提交接口**，将任务提交到内部队列。正常情况下队列会被迅速消费，等待时间极短。

#### L778-784 `SlabStageAllocMemSubmmit()`

```cpp
void SlabStageAllocMemSubmmit(DynDeviceTask* devTask) {
    while (!submmitTaskQueue_.TryEnqueue(devTask)) {
        DeviceTaskMemTryRecycle();
    }
}
```

**异常卡死条件：** 所有任务因 AiCore 挂死无法完成 → 队列持续满无法入队 → 永久阻塞。

**建议优化：** 添加带超时的版本。

---

### 阻塞接口 6：`aicore_manager.h` 状态机 `WAIT_ALL_SCH_FINISH` 阶段

> **设计意图说明**：这是设备任务状态机的一个阶段，等待所有 AICPU 调度线程完成任务调度。通过检查 `runcnt == 0` 判断是否完成。正常情况下其他线程会迅速完成，等待时间极短。

#### L248-254 `RunTask()` 状态机阶段

```cpp
case DevTaskExecStage::WAIT_ALL_SCH_FINISH: {
    isStageFinish = deviceTaskCtx->GetDeviceTaskCtrl()->TryWaitAllSchFinish();
    if (isStageFinish) {
        deviceTaskCtx->EntryStage(DevTaskExecStage::FINISH);
    }
    break;
}
```

其中 `TryWaitAllSchFinish()` 实现：

```cpp
bool TryWaitAllSchFinish()
{
    if (runcnt.load(std::memory_order_acquire) == 0) {
        return true;
    }
    return false;
}
```

**异常卡死条件：** 其他线程异常退出未减少 `runcnt` → 状态机永久停留在该阶段。

**建议优化：** 为该阶段增加超时检测。

---

## ✅ 自旋锁设计（临界区极短，正常情况下不会卡死）

以下位置是**自旋锁的标准实现**，临界区极短（微秒级），正常情况下锁持有时间极短。仅在极低概率的极端异常场景（持有锁的线程被硬件异常中断且永远无法恢复）下才可能死锁。

---

### 自旋锁 1：`device_channel.h` `SpinLock::lock()`

#### L37-41

```cpp
void lock()
{
    while (flag.test_and_set(std::memory_order_acquire))
        cpuRelax();
}
```

**临界区：** slot 数组遍历分配（极短）

**建议优化：** 可添加 `lockWithTimeout()` 方法供需要超时保护的场景使用。

---

### 自旋锁 2：`machine_ws_intf.h` `ReadyQueueLock()` / `ReadyQueueUnLock()`

#### L65-69 `ReadyQueueLock()`

```cpp
inline void ReadyQueueLock(ReadyCoreFunctionQueue* rq)
{
    while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {}
}
```

**临界区：** 队列 head/tail 指针操作（极短）

---

### 自旋锁 3：`wrap_manager.h` `WrapInfoQueueLock()` / `WrapInfoQueueUnLock()`

#### L33-37 `WrapInfoQueueLock()`

```cpp
inline void WrapInfoQueueLock(WrapInfoQueue* rq)
{
    while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {}
}
```

**临界区：** 队列 head/tail 操作和 WrapInfo 元素赋值（极短）

---

### 自旋锁 4：`aicpu_task_manager.h` `ReadyQueueLock()` / `ReadyQueueUnLock()`

#### L120-124 `ReadyQueueLock()`

```cpp
inline void ReadyQueueLock()
{
    while (!__sync_bool_compare_and_swap(&readyQueue_->lock, 0, 1))
        ;
}
```

**临界区：** 队列 head/tail 指针操作（极短）

---

## ✅ 有明确退出条件的循环（正常设计）

以下位置有明确的退出条件，正常情况下会很快退出。

---

### 循环 1：`aicore_manager.h` `TrySendTaskDirectly()` 遍历找空闲核心

> **设计意图说明**：采用**基于计数器保障的遍历找空闲核心**设计。进入 while 循环前有前置检查：只有 `corePendReadyCnt_ > 0`（有空闲核心计数）时才进入循环。计数器维护空闲核心数量，遍历一圈后理论上能找到空闲核心。

#### L1441-1443

```cpp
// 进入循环前的前置检查
if (context_->corePendReadyCnt_[coreType] == 0) {
    return false;  // ← 计数器为0时不进入循环
}

while (pendingIds_[idx] != AICORE_TASK_INIT || !context_->wrapCoreAvail_[idx]) {
    idx = startIdx + (idx - startIdx + 1) % (coreNum);
}
context_->corePendReadyCnt_[coreType]--;  // ← 找到后减少计数器
```

**异常卡死条件：** 计数器与实际核心状态不一致（bug 场景）→ 循环遍历一圈后找不到空闲核心。概率极低。

**建议优化：** 可添加遍历次数上限作为边界保护。

---

### 循环 2：`aicore_manager.h` `BatchSendTask()` 核心遍历

> **设计意图说明**：采用**基于计数器保障的批量发送**设计。循环有两个明确的退出条件：`corePendReadyCnt_ == 0` 或 `sendCnt >= taskCount`。

#### L910-919

```cpp
while (context_->corePendReadyCnt_[static_cast<int>(type)] > 0 && sendCnt < taskCount) {
    if (pendingIds_[idx] == AICORE_TASK_INIT && context_->wrapCoreAvail_[idx]) {
        SendTaskToAiCore(devTaskCtx, type, idx, ...);
        sendCnt++;
        context_->corePendReadyCnt_[static_cast<int>(type)]--;
    }
    idx = coreIdxStart + (idx - coreIdxStart + 1) % coreNum;
}
```

**两个明确的退出条件保障了循环不会无限执行。**

---

### 循环 3：`aicore_manager.h` `PostRun()` 清理循环

> **设计意图说明**：清理任务队列，队列清空后退出。有 `IsEmpty()` 检查作为退出条件。

#### L403-407

```cpp
DeviceTaskCtrl* taskCtrl = nullptr;
while (!taskQueue_->IsEmpty()) {
    if ((taskCtrl = taskQueue_->Dequeue())) {
        taskCtrl->Finish(true);
    }
}
```

**退出条件：** `taskQueue_->IsEmpty()` 返回 true。

---

### 循环 4：`device_execute_context.cpp` `StitchExecute()` 内存分配循环

> **设计意图说明**：Stitch 执行路径中等待内存分配，有错误检查退出条件。

#### L472

```cpp
while (!workspace.TryAllocateFunctionMemory(currDevRootDup, slotContext.GetSlotList())) {
    ret = SubmitToAicoreAndRecycleMemory(true);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return RUNTIME_FUNCKEY_ERROR;  // ← 有错误退出条件！
    }
}
```

**退出条件：** 分配成功或 `SubmitToAicoreAndRecycleMemory` 返回错误。

---

## ⚠️ 边界保护问题（非设计缺陷，建议增强健壮性）

以下位置的循环正常情况下会很快退出，但缺少边界保护。在极端异常场景（内存损坏导致链表环路）下可能无限循环。

---

### 边界保护 1：`slab_ws_allocator.h` 阶段缓存链表遍历

#### L244-246

```cpp
void* temp = caches_[i].stageAllocHead;
while (*static_cast<void**>(temp) != caches_[i].stageAllocTail) {
    temp = *static_cast<void**>(temp);
}
```

**问题：** 若链表结构出现环（内存损坏）或 `stageAllocTail` 指针异常，while 循环可能无限遍历。

**建议优化：** 添加遍历步数上限作为边界保护。

---

### 边界保护 2：`shmem_wait_until.h` 链表遍历

#### L89 `FindTask()`

```cpp
while (current != nullptr) {
    if (current->taskId_ == taskId) { return current; }
    current = current->next;
}
```

**问题：** 若链表出现环（内存损坏），while 循环可能无限遍历。

**建议优化：** 添加遍历步数上限作为边界保护。

---

## ⚠️ Host 侧级联阻塞问题

---

### Host 侧阻塞：`host_machine.cpp` `WaitTaskFinish()`

**当前代码：**

```cpp
void HostMachine::WaitTaskFinish()
{
    while (curTaskId_ != finishQueue_.Size()) {
        usleep(1000);
    }
    // ...
}
```

**问题：** 这是 Host 侧等待 Device 任务完成的核心函数。当 Device 侧因上述任何原因卡死时，Host 侧也会随之永久阻塞。

**建议优化：** 添加超时退出。

```cpp
void HostMachine::WaitTaskFinish()
{
    constexpr uint64_t HOST_WAIT_TIMEOUT_US = 300 * 1000 * 1000;  // 300s
    uint64_t waited = 0;
    while (curTaskId_ != finishQueue_.Size()) {
        usleep(1000);
        waited += 1000;
        if (waited > HOST_WAIT_TIMEOUT_US) {
            finishQueue_.Clear();
            throw std::runtime_error("HostMachine: WaitTaskFinish timeout after 300s.");
        }
    }
    // ...
}
```

---

## 其他安全文件

以下文件中的 while 循环经审查后确认为**数据处理或非阻塞逻辑**，无需修复。每个位置展示具体代码及安全原因分析。

---

### 1. `wrap_manager.h` — 核心遍历找空闲（有边界条件和内部 break）

**文件路径：** `framework/src/machine/device/dynamic/wrap_manager.h`

#### 位置 L233-246（MIX_1C1V 场景）

```cpp
case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
    while (idx < aicReadyCnt) {                    // ← 边界条件：idx < aicReadyCnt
        uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
        uint32_t aivIdx0 = aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
        idx++;
        if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION) {
            CheckCoreIdxInitStatus(aicIdx);
            CheckCoreIdxInitStatus(aivIdx0);
            aicoreIdxList[WRAP_IDX_AIC] = aicIdx;
            aicoreIdxList[WRAP_IDX_AIV0] = aivIdx0;
            aicoreIdxList[WRAP_IDX_AIV1] = aivIdx0;
            validReadyCnt++;
            break;                                   // ← 内部 break 退出
        }
    }
    break;
```

#### 位置 L249-266（MIX_1C2V 场景）

```cpp
case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
    while (idx < aicReadyCnt) {                    // ← 边界条件：idx < aicReadyCnt
        uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
        uint32_t aivIdx0 = aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
        uint32_t aivIdx1 = aivIdx0 + 1;
        idx++;
        if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION &&
            coreIdxPosition_[aivIdx1] != INVALID_COREIDX_POSITION) {
            CheckCoreIdxInitStatus(aicIdx);
            CheckCoreIdxInitStatus(aivIdx0);
            CheckCoreIdxInitStatus(aivIdx1);
            aicoreIdxList[WRAP_IDX_AIC] = aicIdx;
            aicoreIdxList[WRAP_IDX_AIV0] = aivIdx0;
            aicoreIdxList[WRAP_IDX_AIV1] = aivIdx1;
            validReadyCnt++;
            break;                                   // ← 内部 break 退出
        }
    }
    break;
```

**安全原因：**
- ✅ 有明确的边界条件 `idx < aicReadyCnt`（aicReadyCnt 为就绪核心数量，有限值）
- ✅ 找到匹配核心后有内部 `break` 退出
- ✅ `idx` 每次迭代自增，保证循环进展，遍历完就绪核心列表后必然结束

---

### 2. `memory_pool.h` — STL 容器遍历（有明确结束条件）

**文件路径：** `framework/src/machine/runtime/memory_pool.h`

#### 位置 L332-340

```cpp
void DynamicRecycle()
{
    auto it = memoryBlocks_.begin();
    while (it != memoryBlocks_.end()) {            // ← STL 迭代器结束条件
        if ((*it)->used_size == 0) {
            MACHINE_LOGI("Recycling empty block: addr=%p", (*it)->base_addr);
            FreeMemBlock(*it);
            it = memoryBlocks_.erase(it);
        } else {
            ++it;                                   // ← 迭代器自增
        }
    }
}
```

**安全原因：**
- ✅ 使用 STL 迭代器，有明确的结束条件 `it != memoryBlocks_.end()`
- ✅ 每次迭代要么 `erase(it)` 返回新迭代器，要么 `++it`
- ✅ 容器大小有限，遍历必然结束

---

### 3. `device_perf.h` — 分批输出（有明确退出条件）

**文件路径：** `framework/src/machine/device/dynamic/device_perf.h`

#### 位置 L298-303

```cpp
void DumpPerfTrace(uint32_t scheCpuNum, std::string file = "")
{
    // ...
    uint32_t totalLength = str.length();
    uint32_t startPos = 0;
    uint32_t batchSize = 600;
    while (startPos < totalLength) {               // ← 明确的退出条件
        uint32_t endPos = std::min(startPos + batchSize, totalLength);
        std::string batch = str.substr(startPos, endPos - startPos);
        DEV_INFO("tile_fwk aicpu prof:%s", batch.c_str());
        startPos = endPos;                         // ← startPos 递增
    }
    // ...
}
```

**安全原因：**
- ✅ 有明确的退出条件 `startPos < totalLength`（已知字符串长度）
- ✅ `startPos` 每次迭代更新为 `endPos`，递增量至少为 1
- ✅ 处理的是已知长度的字符串，必然结束

---

### 4. `aicore_print.h` — 浮点数规范化、字符串处理、Ring buffer 读

**文件路径：** `framework/src/interface/machine/device/tilefwk/aicore_print.h`

#### 位置 L275-278（FP16 subnormal 规范化）

```cpp
// FP16 subnormal 规范化
while ((normMant & NORM_HIDDEN) == 0) {
    normMant <<= 1;
    --exp32;
}
```

#### 位置 L321-324（FP8 subnormal 规范化）

```cpp
// FP8 subnormal 规范化
while ((normMant & MantHidden) == 0) {
    normMant <<= 1;
    --exp32;
}
```

**安全原因：**
- ✅ 处理浮点数的 subnormal 规范化
- ✅ `normMant` 每次左移 1 位，最多 10-23 次必然满足条件（尾数位数有限）
- ✅ 算法保证有限步数结束

---

#### 位置 L673-693（Ring buffer 读）

```cpp
int Read(char* buf, size_t maxSize) {
    // ...
    while (tail_ != head_) {                       // ← Ring buffer 有数据时读
        DataType type = static_cast<DataType>(ReadByte(tail_++));
        
        if (type == DataType::End) {
            if (totalWritten > 0) {
                return static_cast<int>(totalWritten);
            }
            continue;
        }
        // ...
    }
    return 0;
}
```

**安全原因：**
- ✅ 读取 Ring buffer，有明确的退出条件 `tail_ != head_`
- ✅ Ring buffer 数据量有限，读完必然结束

---

#### 位置 L646-649（dcci 同步缓存行）

```cpp
__aicore__ void Sync() {
    // ...
    while (off < head_) {                          // ← 明确的退出条件
        dcci(&data_[off % size_], SINGLE_CACHE_LINE, CACHELINE_OUT);
        off += CACHE_LINE_SIZE;
    }
    // ...
}
```

**安全原因：**
- ✅ 同步缓存行，有明确退出条件 `off < head_`
- ✅ `off` 每次增加固定的 `CACHE_LINE_SIZE`（64 字节）
- ✅ `head_` 为已知偏移量，必然结束

---

#### 位置 L837-846（格式解析：查找 '%'）

```cpp
while (fmt[idx]) {                                // ← 遇到 '\0' 结束
    if (fmt[idx] == '%') {
        if (fmt[idx + 1] == '%') {
            idx += 2;
        } else {
            break;
        }
    } else {
        idx++;
    }
}
```

#### 位置 L852-858（格式解析：跳过标志字符）

```cpp
while (fmt[idx]) {                                // ← 遇到 '\0' 结束
    if (fmt[idx] != '0' && fmt[idx] != '+' && fmt[idx] != '-' && 
        fmt[idx] != ' ' && fmt[idx] != '#') {
        break;
    }
    idx++;
}
```

#### 位置 L860-864（格式解析：跳过数字）

```cpp
while (IsDigit(fmt[idx])) idx++;                  // ← 非数字字符结束
```

**安全原因：**
- ✅ 解析格式字符串，遇到 `\0` 或特定字符结束
- ✅ 格式字符串长度有限，`idx` 递增保证结束

---

#### 位置 L877-878（字符串长度计算）

```cpp
INLINE size_t StringLength(__gm__ const char* str) {
    size_t n = 0;
    while (*str++) { n++; }                       // ← 遇到 '\0' 结束
    return n;
}
```

#### 位置 L901-906（读取字符串）

```cpp
std::string ReadString(int64_t off) {
    std::string result;
    result.reserve(64);
    while (off < head_) {                         // ← 明确的退出条件
        char c = ReadValue<char>(off++);
        if (c == '\0') break;                     // ← 遇到终止符退出
        result.push_back(c);
    }
    return result;
}
```

#### 位置 L1327-1328（计算名称长度）

```cpp
int64_t nameLen = 0;
while (name[nameLen]) ++nameLen;                  // ← 遇到 '\0' 结束
nameLen += 1;
```

**安全原因：**
- ✅ 字符串操作，遇到 `\0` 终止符自然结束
- ✅ 字符串长度有限

---

#### 位置 L790-794（SkipToEndMarker）

```cpp
__aicore__ void SkipToEndMarker() {
    while (ReadByte(tail_) != static_cast<uint8_t>(DataType::End)) {
        DataType segType = static_cast<DataType>(ReadByte(tail_++));
        SkipRecord(segType);
    }
    tail_++;
}
```

**安全原因：**
- ✅ 跳过 Ring buffer 记录直到 End 标记
- ✅ Ring buffer 数据量有限
- ✅ 编码时保证每个记录序列有 End 标记

---

### 5. `device_switch.h` — `do{}while(0)` 宏模式

**文件路径：** `framework/src/machine/utils/device_switch.h`

#### 位置 L99-102（PROF_START）

```cpp
#define PROF_START(...)                             \
    do {                                            \
        BAREMETAL_RAW_START();                      \
        DEV_PROF("[baremetal]start: " __VA_ARGS__); \
    } while (0)
```

#### 位置 L103-108（PROF_STAGE_BEGIN_DYN）

```cpp
#define PROF_STAGE_BEGIN_DYN(perfkey, ...)        \
    do {                                          \
        DEV_PROF("[baremetal]get: " __VA_ARGS__); \
        BAREMETAL_RAW_GET_PMU();                  \
        PerfBegin(perfkey);                       \
    } while (0)
```

#### 位置 L109-114（PROF_STAGE_END_DYN）

```cpp
#define PROF_STAGE_END_DYN(perfkey, ...)          \
    do {                                          \
        PerfEnd(perfkey);                         \
        BAREMETAL_RAW_GET_PMU();                  \
        DEV_PROF("[baremetal]get: " __VA_ARGS__); \
    } while (0)
```

**安全原因：**
- ✅ `do{}while(0)` 是常见的宏定义模式
- ✅ 循环条件为 `0`，**只执行一次**，等效于单次代码块
- ✅ 用于创建局部作用域，防止宏展开时的变量作用域问题

---

### 6. `device_log.h` — `do{}while(false)` / `do{}while(0)` 宏模式

**文件路径：** `framework/src/machine/utils/device_log.h`

#### 位置 L128-133（DEV_DEBUG_SPLIT）

```cpp
#define DEV_DEBUG_SPLIT(fmt, ...)                                             \
    do {                                                                      \
        if (unlikely(!HardBranchTrue(verboseDebug) || IsLogEnableDebug())) {  \
            DeviceLogSplitDebug(__FUNCTION__, fmt, ##__VA_ARGS__);            \
        }                                                                     \
    } while (false)
```

#### 位置 L135-140（DEV_DEBUG）

```cpp
#define DEV_DEBUG(fmt, ...)                                                                    \
    do {                                                                                       \
        if (unlikely(!HardBranchTrue(verboseDebug) || IsLogEnableDebug())) {                   \
            dlog_debug(LOG_MOD_ID, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);  \
        }                                                                                      \
    } while (false)
```

#### 位置 L142-147（DEV_INFO）

```cpp
#define DEV_INFO(fmt, ...)                                                                     \
    do {                                                                                       \
        if (unlikely(!HardBranchTrue(verboseInfo) || IsLogEnableInfo())) {                     \
            dlog_info(LOG_MOD_ID, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);   \
        }                                                                                      \
    } while (false)
```

#### 位置 L149-154（DEV_WARN）

```cpp
#define DEV_WARN(fmt, ...)                                                                     \
    do {                                                                                       \
        if (IsLogEnableWarn()) {                                                               \
            dlog_warn(LOG_MOD_ID, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);   \
        }                                                                                      \
    } while (false)
```

#### 位置 L156-162（DEV_ERROR）

```cpp
#define DEV_ERROR(errCode, fmt, ...)                                                           \
    do {                                                                                       \
        if (IsLogEnableError()) {                                                              \
            dlog_error(LOG_MOD_ID, "%lu %s\nErrCode: F%05X! " #fmt , GET_TID(), __FUNCTION__,  \
                        static_cast<uint32_t>(errCode) & 0xFFFFF, ##__VA_ARGS__);               \
        }                                                                                      \
    } while (false)
```

#### 位置 L164-169（DEV_VERBOSE_DEBUG）

```cpp
#define DEV_VERBOSE_DEBUG(fmt, ...)                                 \
    do {                                                            \
        DEV_IF_VERBOSE_LOG {                                        \
            DEV_DEBUG(fmt, ##__VA_ARGS__);                          \
        }                                                           \
    } while(0)
```

#### 位置 L171-176（DEV_VERBOSE_INFO）

```cpp
#define DEV_VERBOSE_INFO(fmt, ...)                                  \
    do {                                                            \
        DEV_IF_VERBOSE_LOG {                                        \
            DEV_INFO(fmt, ##__VA_ARGS__);                           \
        }                                                           \
    } while(0)
```

#### 位置 L178-184（DEV_ASSERT_MSG）

```cpp
#define DEV_ASSERT_MSG(errCode, expr, fmt, args...)                           \
    do {                                                                      \
        if (!(expr)) {                                                        \
            DEV_ERROR(errCode, "Assertion failed (%s): " fmt, #expr, ##args); \
            assert(0);                                                        \
        }                                                                     \
    } while (0)
```

#### 位置 L186-192（DEV_ASSERT）

```cpp
#define DEV_ASSERT(errCode, expr)                               \
    do {                                                        \
        if (!(expr)) {                                          \
            DEV_ERROR(errCode, "Assertion failed (%s)", #expr); \
            assert(0);                                          \
        }                                                       \
    } while (0)
```

**安全原因：**
- ✅ `do{}while(false)` 或 `do{}while(0)` 是常见的宏定义模式
- ✅ 循环条件为 `false` 或 `0`，**只执行一次**
- ✅ 用于创建局部作用域，防止宏展开时变量作用域问题，以及支持嵌套 `if-else` 语句

---

## 其他安全文件汇总表

| 文件 | 行号 | while 类型 | 退出机制 | 安全原因 |
|------|------|------------|----------|----------|
| `wrap_manager.h` | L233, L249 | 核心遍历找空闲 | `idx < aicReadyCnt` + 内部 `break` | 有边界条件，找到即退出 |
| `memory_pool.h` | L332 | STL 容器遍历 | `it != memoryBlocks_.end()` | STL 迭代器保证结束 |
| `device_perf.h` | L298 | 分批输出 | `startPos < totalLength` | 已知长度字符串处理 |
| `aicore_print.h` | L275, L321 | 浮点规范化 | `normMant & NORM_HIDDEN` | 左移有限次数必然满足 |
| `aicore_print.h` | L673 | Ring buffer 读 | `tail_ != head_` | 数据量有限 |
| `aicore_print.h` | L646 | dcci 同步 | `off < head_` | 固定步长递增 |
| `aicore_print.h` | L790 | SkipToEndMarker | `DataType::End` | End 标记必然存在 |
| `aicore_print.h` | L837, L852, L860 | 格式解析 | `fmt[idx] == '\0'` 或特定字符 | 字符串有限 |
| `aicore_print.h` | L877, L901, L1327 | 字符串操作 | `*str == '\0'` | 字符串终止符 |
| `device_switch.h` | L102, L108, L114 | `do{}while(0)` | 条件恒为 0 | 宏模式，只执行一次 |
| `device_log.h` | 多处 | `do{}while(false)` | 条件恒为 false | 宏模式，只执行一次 |

---

# 修复前后总览

| 类别 | 修复前 | 修复后 |
|------|--------|--------|
| **❌ 设计缺陷（5 处）** | 伪超时（4处）+ 裸 mutex 无 RAII（2处） | 伪超时改为真超时，mutex 改用 RAII |
| **已有超时问题（15 处）** | 阈值一刀切 250ms、`DEV_IF_DEVICE` 包裹、void 返回、超时后不设错误状态 | 统一使用分层超时常量、去掉包裹、void 改 int32_t |
| **✅ 阻塞接口设计（6 类）** | 无超时版本可选 | 建议添加带超时的版本供调用者选择 |
| **✅ 自旋锁（4 处）** | 无超时版本 | 建议添加 `lockWithTimeout()` 方法 |
| **⚠️ 边界保护问题（2 处）** | 无边界保护 | 建议添加遍历步数上限 |
| **⚠️ Host 侧级联（1 处）** | Device 卡死导致 Host 永久阻塞 | 建议添加超时退出 |

---

# 涉及文件

**`framework/src/machine/`（AICPU 侧）：**
`device_utils.h`、`spsc_queue.h`、`device_channel.h`、`machine_ws_intf.h`、`wrap_manager.h`、`aicpu_task_manager.h`、`aicore_hal.h`、`aicore_manager.h`、`device_ctrl.h`、`device_sche.h`、`dev_workspace.h`、`slab_ws_allocator.h`、`dev_start_args.h`、`device_task.h`、`device_execute_context.cpp`、`device_perf.h`、`shmem_wait_until.h`

**`framework/src/interface/machine/`（AiCore 入口 + Host 侧）：**
`aicore_entry.h`、`aikernel_runtime.h`、`host_machine.cpp`

---

# 实施顺序

Phase 0 → 超时常量基础设施（`device_utils.h`）：统一时间基准，定义分层超时常量
Phase 1 → ❌ 设计缺陷修复：伪超时改为真超时 + 裸 mutex RAII 规范化
Phase 2 → 已有超时规范化：去掉 `DEV_IF_DEVICE` 包裹，void 返回改为 int32_t，调整阈值
Phase 3 → ✅ 阻塞接口优化：为阻塞等待接口添加带超时的版本（建议性质）
Phase 4 → ✅ 自旋锁优化：为自旋锁添加 `lockWithTimeout()` 方法（建议性质）
Phase 5 → ⚠️ 边界保护：为链表遍历添加步数上限（建议性质）
Phase 6 → ⚠️ Host 侧优化：为 `WaitTaskFinish()` 添加超时退出（建议性质）

> **说明**：
> - Phase 0、Phase 1、Phase 2 为**必须修复项**
> - Phase 3-6 为**建议性质**，增强健壮性，供需要超时保护的调用者选择，保留现有阻塞接口不变