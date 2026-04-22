# Timeout Detection Usage Guide

## Overview

This document describes how to use the unified timeout detection macros added to `device_utils.h` for PyPTO MACHINE module.

## Problem Statement

The original timeout detection code had several issues:
1. `TIMEOUT_CHECK_AND_RESET` was a pseudo-timeout that reset and continued looping, never exiting
2. Timeout constants used inconsistent units (cycles vs nanoseconds)
3. Platform frequency differences (A2/A3: 50MHz, A5: 1000MHz) were not properly handled
4. Some loops had no timeout protection, causing potential deadlocks

## Solution: Unified Timeout Detection Macros

### Core Design

One unified macro `TIMEOUT_CHECK` supports three modes:
- **MODE_EXIT**: Exit immediately on timeout
- **MODE_WARN**: Periodic warnings without exit
- **MODE_WARN_THEN_EXIT**: Warn at threshold, exit at final timeout

### Timeout Constants (Nanosecond-based)

```cpp
constexpr uint64_t TIMEOUT_NS_1MIN  = 60ULL  * 1000ULL * 1000ULL * 1000ULL;  // 1 min
constexpr uint64_t TIMEOUT_NS_10MIN = 600ULL * 1000ULL * 1000ULL * 1000ULL;  // 10 min
constexpr uint64_t TIMEOUT_NS_20MIN = 1200ULL * 1000ULL * 1000ULL * 1000ULL; // 20 min
```

These constants automatically adapt to platform frequency via `TimeoutState::NsToCycles()`.

### Usage Patterns

#### Pattern 1: Timeout with Exit (20 minutes)

Used for critical allocation loops like `SlabAlloc`, `AllocThreadIdxForDav2201`.

```cpp
WsAllocation SlabAlloc(uint32_t objSize, WsAicpuSlabMemType type)
{
    void* ptr = nullptr;
    SlabTryDynAddCache(type, objSize);
    
    TimeoutState state;  // Initialize timeout state
    
    do {
        // ... allocation logic ...
        if (ptr != nullptr) { break; }
        
        // Check timeout: exit after 20 minutes
        TIMEOUT_CHECK_EXIT(state, TIMEOUT_NS_20MIN, 
                           DEVICE_MACHINE_TIMEOUT_SLAB_ALLOC,
                           "#workspace.alloc.timeout: SlabAlloc timeout 20min, objSize=%u, type=%d.",
                           objSize, ToUnderlying(type));
    } while (true);
    
    WsAllocation allocation;
    allocation.ptr = reinterpret_cast<uintdevptr_t>(ptr);
    return allocation;
}
```

#### Pattern 2: Timeout with Warning at N/2, then Exit (1 minute)

Used for scheduler initialization like `ScheWait`, `AllocNewTaskCtrl`.

```cpp
int AllocNewTaskCtrl()
{
    uint32_t& taskCtrlIndex = devStartArgs_->devCtrlState.taskCtrlIndex;
    
    TimeoutState state;
    
    while (true) {
        if (taskCtrlIndex == MAX_DEVICE_TASK_NUM) {
            taskCtrlIndex = 0;
        }
        if (!GetTaskCtrlInPool(taskCtrlIndex).IsNotFree()) {
            return taskCtrlIndex++;
        }
        taskCtrlIndex++;
        
        // Warn at 30s, exit at 60s
        TIMEOUT_CHECK_WARN_EXIT(state, TIMEOUT_NS_1MIN, TIMEOUT_NS_1MIN / 2,
                                DEVICE_MACHINE_TIMEOUT_CTRL_ALLOC,
                                "#ctrl.alloc: AllocNewTaskCtrl waiting 30s, taskCtrlIndex=%u.",
                                "#ctrl.alloc: AllocNewTaskCtrl timeout 1min.",
                                taskCtrlIndex);
    }
}
```

#### Pattern 3: Periodic Warning without Exit (10 minutes)

Used for ring buffer waits like `AllocateWait`.

```cpp
void AllocateWait()
{
    TimeoutState state;
    
    while (Full()) {
        RuntimeYield();
        
        // Periodic warning every 10 minutes, no exit
        TIMEOUT_CHECK_WARN(state, TIMEOUT_NS_10MIN,
                           "#ringbuffer.alloc: AllocateWait waiting 10min, ring buffer full.");
    }
}
```

## Platform Frequency Verification

To verify the platform frequency:

```cpp
uint64_t start = GetCycles();
usleep(1000000);  // Sleep 1 second
uint64_t end = GetCycles();
uint64_t cycles_per_sec = end - start;

DEV_INFO("Platform frequency: %lu cycles/sec = %lu MHz",
         cycles_per_sec, cycles_per_sec / 1000000);

// Expected:
// A2/A3: ~50,000,000 cycles/sec (50 MHz)
// A5:    ~1,000,000,000 cycles/sec (1000 MHz)
```

## Migration Guide

### From TIMEOUT_CHECK_AND_RESET (Legacy)

Old code (pseudo-timeout, never exits):
```cpp
TIMEOUT_CHECK_START();
while (condition) {
    TIMEOUT_CHECK_AND_RESET(TIMEOUT_ONE_MINUTE, err_code, "timeout message");
    // ... continues forever ...
}
```

New code (true timeout):
```cpp
TimeoutState state;
while (condition) {
    TIMEOUT_CHECK_EXIT(state, TIMEOUT_NS_1MIN, err_code, "timeout message");
    // ... exits on timeout ...
}
```

### From DEV_IF_DEVICE Wrapped Timeouts

Old code (timeout disabled in sim mode):
```cpp
DEV_IF_DEVICE {
    if (GetCycles() - start > TIMEOUT_CYCLES) {
        return DEVICE_MACHINE_ERROR;
    }
}
```

New code (timeout works in all modes):
```cpp
TimeoutState state;
TIMEOUT_CHECK_EXIT(state, TIMEOUT_NS_10MIN, DEVICE_MACHINE_ERROR, "timeout");
```

## Implementation Files to Modify

Based on issue_1217 analysis:

| File | Function | Timeout | Mode |
|------|----------|---------|------|
| `dev_workspace.h` | `SlabAlloc` | 20min | MODE_EXIT |
| `device_sche.h` | `AllocThreadIdxForDav2201` | 20min | MODE_EXIT |
| `device_sche.h` | `AllocThreadIdxForDav3510` | 20min | MODE_EXIT |
| `device_sche.h` | `ScheWait` | 1min | MODE_WARN_THEN_EXIT |
| `device_ctrl.h` | `AllocNewTaskCtrl` | 1min | MODE_WARN_THEN_EXIT |
| `dev_start_args.h` | `AllocateWait` | 10min | MODE_WARN |
| `dev_workspace.h` | `SlabStageAllocMemSubmmit` | 10min | MODE_EXIT |
| `slab_ws_allocator.h` | Linked list traversal | 10min | MODE_WARN |
| `aicore_manager.h` | `ProcessTaskLoop` | 10min | MODE_EXIT |
| `aicore_manager.h` | `SyncAicoreDevTaskFinish` | 10min | MODE_EXIT |

## Deprecated Code to Delete

Per review requirements:
- `aicore_hal.h::WaitFinQueue()` - deprecated, delete
- `device_channel.h` entire file - deprecated, delete
- `device_sche.h::RunUnifiedCtrlInit()` - deprecated, delete

## Testing

1. Enable verbose debug: `debug_runtime_mode = 1`
2. Run ST tests and verify no false timeout warnings
3. Verify performance profile (swimlane) still generated after removing deprecated code