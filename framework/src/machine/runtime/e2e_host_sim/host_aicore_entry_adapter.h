#pragma once

#include <cstdint>
#include "machine/runtime/e2e_host_sim/host_core_context.h"
#include "machine/runtime/e2e_host_sim/host_sim_clock.h"
#include "machine/runtime/e2e_host_sim/host_reg_bus.h"

namespace npu::tile_fwk {
struct CoreFuncParam;
}

#ifndef dcci
#define dcci(...)
#endif

#ifndef set_flag
#define set_flag(...)
#endif

#ifndef wait_flag
#define wait_flag(...)
#endif

#ifndef set_mask_norm
#define set_mask_norm(...)
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
 	      npu::tile_fwk::dynamic::HostRegBus::Global().WriteCond(static_cast<size_t>(ctx.blockId), v);
}

static inline uint64_t GetDataMainBase() 
{  
    const auto& ctx = npu::tile_fwk::dynamic::HostCoreCtx::Current();
 	      return npu::tile_fwk::dynamic::HostRegBus::Global().ReadMainBase(static_cast<size_t>(ctx.blockId));
}

static inline void CallSubFuncTask(uint64_t, npu::tile_fwk::CoreFuncParam*, int64_t, int64_t*) 
{
    npu::tile_fwk::dynamic::HostSimClock::AdvanceNs(5000); // fixed 5us modeling
}
