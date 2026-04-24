#pragma once

#include <atomic>
#include <cstdint>
#include <stdint.h>
#include <chrono>
#include "machine/runtime/e2e_host_sim/host_core_context.h"
#include "machine/runtime/e2e_host_sim/host_reg_bus.h"

#ifndef UINT64_MAX
typedef unsigned long long uint64_t;
typedef long long int64_t;
#endif

namespace npu::tile_fwk {
struct CoreFuncParam;
}

#ifndef mem_dsb_t
typedef uint64_t mem_dsb_t;
#endif

#ifndef dsb
#define dsb(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

#ifndef dcci
#define dcci(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

#ifndef set_flag
#define set_flag(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

#ifndef wait_flag
#define wait_flag(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

#ifndef set_mask_norm
#define set_mask_norm(...) std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

static inline uint64_t get_sys_cnt()
{
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

static inline int64_t get_coreid() { return npu::tile_fwk::dynamic::HostCoreCtx::Current().phyId; }

static inline int get_block_idx() { return npu::tile_fwk::dynamic::HostCoreCtx::Current().blockId; }

static inline int get_subblockdim() { return 0; }

static inline int get_subblockid() { return 0; }

static inline int get_block_num() { return 0; }

static inline void set_cond(uint64_t v)
{
    const auto& ctx = npu::tile_fwk::dynamic::HostCoreCtx::Current();
    npu::tile_fwk::dynamic::HostRegBus::Global().WriteCond(static_cast<size_t>(ctx.phyId), v);
}

static inline uint64_t GetDataMainBase()
{
    const auto& ctx = npu::tile_fwk::dynamic::HostCoreCtx::Current();
    return npu::tile_fwk::dynamic::HostRegBus::Global().ReadMainBase(static_cast<size_t>(ctx.phyId));
}

static inline void CallSubFuncTask(uint64_t, npu::tile_fwk::CoreFuncParam*, int64_t, int64_t*)
{
    std::this_thread::sleep_for(std::chrono::microseconds(5)); // fixed 5us modeling
}
