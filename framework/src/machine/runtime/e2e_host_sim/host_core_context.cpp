#include "machine/runtime/e2e_host_sim/host_core_context.h"

namespace npu::tile_fwk::dynamic {

namespace {
thread_local HostCoreContext g_ctx{};
}

const HostCoreContext& HostCoreCtx::Current() { return g_ctx; }

void HostCoreCtx::SetCurrent(const HostCoreContext& ctx) { g_ctx = ctx; }

} // namespace npu::tile_fwk::dynamic
