#pragma once

#include <string>
#include <vector>

#include "gpu_vk/ir/gpu_vk_dispatch_ir.h"
#include "gpu_vk/ir/gpu_vk_tensor_ir.h"

namespace npu::tile_fwk::gpu_vk {

struct GpuVkShaderOp {
    GpuVkOpKind op{GpuVkOpKind::UNKNOWN};
    std::string expr;
};

class GpuVkShaderFunction {
public:
    void SetName(const std::string &name);
    const std::string &Name() const;

    void SetDispatch(const GpuVkDispatchParam &dispatch);
    const GpuVkDispatchParam &Dispatch() const;

    void AddOp(const GpuVkShaderOp &op);
    const std::vector<GpuVkShaderOp> &Ops() const;

private:
    std::string name_;
    GpuVkDispatchParam dispatch_;
    std::vector<GpuVkShaderOp> ops_;
};

} // namespace npu::tile_fwk::gpu_vk
