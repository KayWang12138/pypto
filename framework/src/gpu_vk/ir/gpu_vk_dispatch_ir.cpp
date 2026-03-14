#include "gpu_vk/ir/gpu_vk_dispatch_ir.h"

namespace npu::tile_fwk::gpu_vk {

void GpuVkDispatchGraph::SetKernelName(const std::string &kernelName) {
    kernelName_ = kernelName;
}

const std::string &GpuVkDispatchGraph::KernelName() const {
    return kernelName_;
}

void GpuVkDispatchGraph::AddBinding(const GpuVkBufferBinding &binding) {
    bindings_.push_back(binding);
}

const std::vector<GpuVkBufferBinding> &GpuVkDispatchGraph::Bindings() const {
    return bindings_;
}

void GpuVkDispatchGraph::AddLoweredOp(const std::string &expr) {
    loweredOps_.push_back(expr);
}

const std::vector<std::string> &GpuVkDispatchGraph::LoweredOps() const {
    return loweredOps_;
}

void GpuVkDispatchGraph::SetDispatch(const GpuVkDispatchParam &dispatch) {
    dispatch_ = dispatch;
}

const GpuVkDispatchParam &GpuVkDispatchGraph::Dispatch() const {
    return dispatch_;
}

} // namespace npu::tile_fwk::gpu_vk
