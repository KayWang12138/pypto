#pragma once

#include <atomic>
#include <cstdio>
#include <cstdint>
#include "machine/runtime/e2e_host_sim/host_core_context.h"
#include "machine/runtime/e2e_host_sim/host_sim_clock.h"
#include "machine/runtime/e2e_host_sim/host_reg_bus.h"

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

static inline uint64_t get_sys_cnt() { return npu::tile_fwk::dynamic::HostSimClock::NowNs(); }

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
    uint64_t regVal = npu::tile_fwk::dynamic::HostRegBus::Global().ReadMainBase(static_cast<size_t>(ctx.phyId));
    if (ctx.blockId == 13 || ctx.blockId == 63) {
        thread_local uint64_t lastMainBase = UINT64_MAX;
        thread_local uint64_t stableReadCount = 0;
        if (regVal != lastMainBase) {
            std::fprintf(
                stderr,
                "[HOST_SIM_READ_REG_CHANGE] blockId=%d phyId=%d mainBase=0x%llx high=0x%x low=0x%x\n",
                ctx.blockId,
                ctx.phyId,
                static_cast<unsigned long long>(regVal),
                static_cast<unsigned int>(regVal >> 32),
                static_cast<unsigned int>(regVal & 0xFFFFFFFFULL));
            lastMainBase = regVal;
            stableReadCount = 0;
        } else {
            ++stableReadCount;
            if ((stableReadCount % 50000ULL) == 0ULL) {
                std::fprintf(
                    stderr,
                    "[HOST_SIM_READ_REG_STABLE] blockId=%d phyId=%d mainBase=0x%llx stableReads=%llu\n",
                    ctx.blockId,
                    ctx.phyId,
                    static_cast<unsigned long long>(regVal),
                    static_cast<unsigned long long>(stableReadCount));
            }
        }
    }
    return regVal;
}

static inline void CallSubFuncTask(uint64_t, npu::tile_fwk::CoreFuncParam*, int64_t, int64_t*)
{
    npu::tile_fwk::dynamic::HostSimClock::AdvanceNs(5000); // fixed 5us modeling
}
