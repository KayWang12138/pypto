#include "gpu_vk/ir/gpu_vk_shader_ir.h"

namespace npu::tile_fwk::gpu_vk {

void GpuVkShaderFunction::SetName(const std::string &name) {
    name_ = name;
}

const std::string &GpuVkShaderFunction::Name() const {
    return name_;
}

void GpuVkShaderFunction::SetDispatch(const GpuVkDispatchParam &dispatch) {
    dispatch_ = dispatch;
}

const GpuVkDispatchParam &GpuVkShaderFunction::Dispatch() const {
    return dispatch_;
}

void GpuVkShaderFunction::AddOp(const GpuVkShaderOp &op) {
    ops_.push_back(op);
}

const std::vector<GpuVkShaderOp> &GpuVkShaderFunction::Ops() const {
    return ops_;
}

} // namespace npu::tile_fwk::gpu_vk
