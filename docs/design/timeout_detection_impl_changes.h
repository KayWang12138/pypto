/**
 * Timeout Detection Implementation Changes
 * Based on issue_1217 analysis and review requirements
 */

// ========== 1. SlabAlloc() Modification ==========
// File: framework/src/machine/utils/dynamic/dev_workspace.h
// Lines: 728-766
// Problem: Pseudo-timeout (reset and continue), never exits
// Fix: True timeout exit (20 minutes)

WsAllocation SlabAlloc(uint32_t objSize, WsAicpuSlabMemType type)
{
    void* ptr = nullptr;
    DEV_VERBOSE_DEBUG("SlabAlloc type = %u, size = %u.", ToUnderlying(type), objSize);
    SlabTryDynAddCache(type, objSize);
    
    TimeoutState state;  // NEW: Initialize timeout state
    
    do {
        if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
            ptr = metadataAllocators_.generalSlab.Alloc(ToUnderlying(type));
        } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
            ptr = metadataAllocators_.stitchSlab.Alloc(ToUnderlying(type));
        }
        if (ptr != nullptr) {
            break;
        }

        if (submmitTaskQueue_.IsEmpty()) {
            metadataAllocators_.generalSlab.DumpMemoryStatusWhenAbnormal("SlabAlloc null");
            metadataAllocators_.stitchSlab.DumpMemoryStatusWhenAbnormal("SlabAlloc null");
            DEV_ERROR(
                WsErr::SLAB_ADD_CACHE_FAILED, "#workspace.init.check: Slab alloc null,type=%u,objsize=%u.",
                ToUnderlying(type), objSize);
            DEV_ASSERT_MSG(
                WsErr::SLAB_ADD_CACHE_FAILED, false, "Slab alloc null,type=%u,objsize=%u.", ToUnderlying(type),
                objSize);
        }
        
        // NEW: Inner timeout (10 minutes for inner wait)
        TimeoutState innerState;
        while (!DeviceTaskMemTryRecycle()) {
            TIMEOUT_CHECK_EXIT(innerState, TIMEOUT_NS_10MIN,
                               DEVICE_MACHINE_ERROR,
                               "#workspace.alloc.inner_timeout: Inner recycle wait 10min, type=%u, objSize=%u.",
                               ToUnderlying(type), objSize);
        };
        
        // NEW: Outer timeout check (20 minutes)
        TIMEOUT_CHECK_EXIT(state, TIMEOUT_NS_20MIN,
                           DEVICE_MACHINE_TIMEOUT_SLAB_ALLOC,
                           "#workspace.alloc.timeout: SlabAlloc timeout 20min, type=%u, objSize=%u.",
                           ToUnderlying(type), objSize);
    } while (true);

    WsAllocation allocation;
    allocation.ptr = reinterpret_cast<uintdevptr_t>(ptr);
    return allocation;
}


// ========== 2. AllocThreadIdxForDav2201() Modification ==========
// File: framework/src/machine/device/dynamic/device_sche.h
// Lines: 205-237
// Problem: TIMEOUT_CHECK_AND_RESET pseudo-timeout
// Fix: True timeout exit (20 minutes)

int AllocThreadIdxForDav2201(DeviceArgs* devArgs, int cpu, int& curThreadIdx, std::atomic<int>& threadIdx)
{
    cpumask_.fetch_or(1 << cpu, std::memory_order_release);
    
    TimeoutState state;  // NEW
    
    while (__builtin_popcount(cpumask_.load(std::memory_order_acquire)) != static_cast<int>(devArgs->nrAicpu)) {
        // NEW: True timeout exit (20 minutes)
        TIMEOUT_CHECK_EXIT(state, TIMEOUT_NS_20MIN,
                           DEVICE_MACHINE_TIMEOUT_THREAD_ALLOC,
                           "#sche.thread.init: Thread alloc timeout 20min: threadIdx=%d, physicalCpu=%d.",
                           curThreadIdx, cpu);
        sched_yield();
    }

    auto maskval = cpumask_.load(std::memory_order_relaxed);
    int cpuoff = 0;
    int clus_id = -1;
    for (int index = 0; index < static_cast<int>(sizeof(uint64_t)); ++index) {
        int mask = (maskval >> cpuoff) & 0xF;
        if (__builtin_popcount(static_cast<uint32_t>(mask)) >= static_cast<int>(devArgs->scheCpuNum)) {
            clus_id = index;
            break;
        }
        cpuoff += CPUS_PER_CLUSTER;
    }
    if (clus_id == -1) {
        curThreadIdx = ++threadIdx;
        return npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
    }
    if (cpu < cpuoff || cpu >= (cpuoff + CPUS_PER_CLUSTER)) {
        curThreadIdx = -1;
        return npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
    }
    curThreadIdx = ++threadIdx;
    return npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
}


// ========== 3. AllocThreadIdxForDav3510() Modification ==========
// File: framework/src/machine/device/dynamic/device_sche.h
// Lines: 172-203 (L189-198 is the timeout part)
// Problem: TIMEOUT_CYCLES (250ms) is too small
// Fix: True timeout exit (20 minutes)

int AllocThreadIdxForDav3510(DeviceArgs* devArgs, int cpu, int& curThreadIdx, std::atomic<int>& threadIdx)
{
    int die0MaxCpuid = static_cast<int>(devArgs->maxAicpuNum >> 1);
    int die0MaxCpuNum = static_cast<int>(devArgs->scheCpuNum >> 1);
    int die1MaxCpuNum = static_cast<int>(devArgs->scheCpuNum) - die0MaxCpuNum;

    if (cpu <= die0MaxCpuid) {
        SetCurThreadIdxForDav3510(die0MaxCpuNum, SCHE_THREAD_START_IDX, curThreadIdx, die0ThreadIdx_);
    } else {
        SetCurThreadIdxForDav3510(
            die1MaxCpuNum, die0MaxCpuNum + SCHE_THREAD_START_IDX, curThreadIdx, die1ThreadIdx_);
    }

    cpumask_.fetch_or(1 << cpu, std::memory_order_release);
    
    TimeoutState state;  // NEW
    
    while (__builtin_popcount(cpumask_.load(std::memory_order_acquire)) != static_cast<int>(devArgs->nrAicpu)) {
        // NEW: True timeout exit (20 minutes), consistent with Dav2201
        TIMEOUT_CHECK_EXIT(state, TIMEOUT_NS_20MIN,
                           DEVICE_MACHINE_TIMEOUT_THREAD_ALLOC,
                           "#sche.thread.init: Thread alloc timeout 20min: threadIdx=%d, physicalCpu=%d.",
                           curThreadIdx, cpu);
        sched_yield();
    }

    DEV_INFO("Thread alloc success: physicalCpu=%d, threadIdx=%d.", cpu, curThreadIdx);
    threadIdx = curThreadIdx;
    return npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
}


// ========== 4. ScheWait() Modification ==========
// File: framework/src/machine/device/dynamic/device_sche.h
// Lines: 657-674
// Problem: TIMEOUT_CHECK_AND_RESET pseudo-timeout in both loops
// Fix: Warn at 30s, exit at 60s (MODE_WARN_THEN_EXIT)

void ScheWait(DevAscendProgram* devProg)
{
    // First loop: wait for ring buffer initialization
    TimeoutState state1;  // NEW
    
    while (unlikely(!devProg->runtimeDataRingBufferInited)) {
        RuntimeYield(0);
        // NEW: Warn at 30s, exit at 60s
        TIMEOUT_CHECK_WARN_EXIT(state1, TIMEOUT_NS_1MIN, TIMEOUT_NS_1MIN / 2,
                                SchedErr::RINGBUFFER_WAIT_TIMEOUT,
                                "#sche.wait: RingBuffer init waiting 30s, may timeout soon.",
                                "#sche.wait: RingBuffer init timeout 1min.");
    }
    
    RuntimeDataRingBufferHead* ringBufferHead = devProg->GetRuntimeDataList();
    
    // Second loop: wait for ring buffer data ready
    TimeoutState state2;  // NEW
    
    while (unlikely(ringBufferHead->Empty())) {
        RuntimeYield(0);
        // NEW: Warn at 30s, exit at 60s
        TIMEOUT_CHECK_WARN_EXIT(state2, TIMEOUT_NS_1MIN, TIMEOUT_NS_1MIN / 2,
                                SchedErr::RINGBUFFER_WAIT_TIMEOUT,
                                "#sche.wait: RingBuffer data waiting 30s, may timeout soon.",
                                "#sche.wait: RingBuffer data timeout 1min.");
    }
}


// ========== 5. AllocNewTaskCtrl() Modification ==========
// File: framework/src/machine/device/dynamic/device_ctrl.h
// Lines: 60-73
// Problem: while(true) + TIMEOUT_CHECK_AND_RESET pseudo-timeout = infinite loop
// Fix: Warn at 30s, exit at 60s (MODE_WARN_THEN_EXIT)

int AllocNewTaskCtrl()
{
    uint32_t& taskCtrlIndex = devStartArgs_->devCtrlState.taskCtrlIndex;
    
    TimeoutState state;  // NEW
    
    while (true) {
        if (taskCtrlIndex == MAX_DEVICE_TASK_NUM) {
            taskCtrlIndex = 0;
        }
        if (!GetTaskCtrlInPool(taskCtrlIndex).IsNotFree()) {
            return taskCtrlIndex++;
        }
        taskCtrlIndex++;
        
        // NEW: Warn at 30s, exit at 60s
        TIMEOUT_CHECK_WARN_EXIT(state, TIMEOUT_NS_1MIN, TIMEOUT_NS_1MIN / 2,
                                DEVICE_MACHINE_TIMEOUT_CTRL_ALLOC,
                                "#ctrl.alloc: AllocNewTaskCtrl waiting 30s, taskCtrlIndex=%u.",
                                "#ctrl.alloc: AllocNewTaskCtrl timeout 1min.",
                                taskCtrlIndex);
    }
}


// ========== 6. AllocateWait() Modification ==========
// File: framework/src/machine/utils/dynamic/dev_start_args.h
// Lines: 171-176
// Problem: No timeout detection
// Fix: Periodic warning (10 minutes), no exit (MODE_WARN)

void AllocateWait()
{
    TimeoutState state;  // NEW
    
    while (Full()) {
        RuntimeYield();
        
        // NEW: Periodic warning every 10 minutes, no exit
        TIMEOUT_CHECK_WARN(state, TIMEOUT_NS_10MIN,
                           "#ringbuffer.alloc: AllocateWait waiting 10min, ring buffer full.");
    }
}


// ========== 7. SlabStageAllocMemSubmmit() Modification ==========
// File: framework/src/machine/utils/dynamic/dev_workspace.h
// Lines: 778-784
// Problem: No timeout detection
// Fix: True timeout exit (10 minutes)

void SlabStageAllocMemSubmmit(DynDeviceTask* devTask)
{
    TimeoutState state;  // NEW
    
    while (!submmitTaskQueue_.TryEnqueue(devTask)) {
        DeviceTaskMemTryRecycle();
        
        // NEW: Timeout exit (10 minutes)
        TIMEOUT_CHECK_EXIT(state, TIMEOUT_NS_10MIN,
                           DEVICE_MACHINE_TIMEOUT_SLAB_ALLOC,
                           "#workspace.submit: SlabStageAllocMemSubmmit timeout 10min.");
    }
    return;
}


// ========== 8. Host-side WaitTaskFinish() Modification ==========
// File: framework/src/machine/runtime/host_machine.cpp
// Problem: No timeout, blocks forever when device hangs
// Fix: Host-side timeout (5 minutes), throws exception

void HostMachine::WaitTaskFinish()
{
    constexpr uint64_t HOST_WAIT_TIMEOUT_US = 300ULL * 1000ULL * 1000ULL;  // 300s = 5min
    uint64_t waited = 0;
    
    while (curTaskId_ != finishQueue_.Size()) {
        usleep(1000);
        waited += 1000;
        
        if (waited > HOST_WAIT_TIMEOUT_US) {
            DEV_ERROR(HostErr::WAIT_TASK_TIMEOUT,
                      "#host.wait: WaitTaskFinish timeout 5min, curTaskId=%lu.",
                      curTaskId_);
            finishQueue_.Clear();
            throw std::runtime_error("HostMachine: WaitTaskFinish timeout after 5min.");
        }
    }
    // ... rest of the function
}


// ========== 9. slab_ws_allocator.h Linked List Traversal ==========
// File: framework/src/machine/utils/dynamic/allocator/slab_ws_allocator.h
// Lines: 244-246 (from issue body)
// Problem: No boundary protection, potential infinite loop if memory corrupted
// Fix: Periodic warning (10 minutes), add traversal step limit

void* temp = caches_[i].stageAllocHead;

TimeoutState state;  // NEW
uint32_t stepCount = 0;  // NEW: traversal step counter

while (*static_cast<void**>(temp) != caches_[i].stageAllocTail) {
    temp = *static_cast<void**>(temp);
    
    // NEW: Periodic warning every 10 minutes
    TIMEOUT_CHECK_WARN(state, TIMEOUT_NS_10MIN,
                       "#slab.allocator: StageAllocHead traversal waiting 10min, cache_idx=%u.",
                       i);
    
    // NEW: Optional - add traversal step limit for extra safety
    stepCount++;
    if (stepCount > MAX_TRAVERSAL_STEPS) {
        DEV_ERROR(WsErr::SLAB_TRAVERSAL_LIMIT_EXCEEDED,
                  "#slab.allocator: Traversal step limit exceeded %u, possible memory corruption.",
                  MAX_TRAVERSAL_STEPS);
        break;
    }
}


// ========== 10. aicore_manager.h - Remove DEV_IF_DEVICE Wrapping ==========
// File: framework/src/machine/device/dynamic/aicore_manager.h
// Lines: L321-339 (ProcessTaskLoop), L449-476 (RunManager), L749-793 (SyncAicoreDevTaskFinish)
// Problem: DEV_IF_DEVICE wrapping causes timeout disabled in sim/ESL mode
// Fix: Remove DEV_IF_DEVICE, use direct timeout check

// Example: ProcessTaskLoop (L321-339)
uint64_t start = GetCycles();
while (!deviceTaskCtx->IsCoreTaskSendFinish()) {
    int32_t ret = RunCoreTask<true>(deviceTaskCtx);
    if (unlikely(ret != DEVICE_MACHINE_OK)) { return ret; }
    if (deviceTaskCtx->IsParallel()) { /* ... return */ }
    
    // NEW: Remove DEV_IF_DEVICE, direct check (10 minutes timeout)
    TimeoutState state;
    TIMEOUT_CHECK_EXIT(state, TIMEOUT_NS_10MIN,
                       DEVICE_MACHINE_TIMEOUT_CORETASK,
                       "#sche.task.coretask: CoreTask timeout 10min.");
}


// ========== DEPRECATED CODE TO DELETE ==========

// 1. aicore_hal.h::WaitFinQueue() - DELETE entire function
//    File: framework/src/machine/device/dynamic/aicore_hal.h
//    Lines: 243-255
//    Reason: Deprecated, no longer used

// 2. device_channel.h - DELETE entire file
//    File: framework/src/machine/utils/dynamic/device_channel.h
//    Reason: Deprecated, no longer used

// 3. device_sche.h::RunUnifiedCtrlInit() - DELETE function
//    File: framework/src/machine/device/dynamic/device_sche.h
//    Lines: 425 (approximate)
//    Reason: Deprecated, use RAII std::lock_guard instead

// 4. TIMEOUT_CHECK_AND_RESET macro - Keep for backward compatibility
//    but mark as DEPRECATED in documentation