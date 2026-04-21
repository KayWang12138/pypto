#pragma once

#include <cstdint>

namespace npu::tile_fwk::dynamic {

struct HostCoreContext {
    int32_t blockId{-1};
    int32_t phyId{-1};
};

class HostCoreCtx {
public:
    static const HostCoreContext& Current();
    static void SetCurrent(const HostCoreContext& ctx);
};

} // namespace npu::tile_fwk::dynamic
