#include "gpu_vk/passes/layout_normalize.h"

#include <algorithm>

namespace npu::tile_fwk::gpu_vk {

namespace {

std::vector<std::int64_t> ComputeStride(const std::vector<std::int64_t> &shape) {
    std::vector<std::int64_t> stride(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 1; --i) {
        stride[i - 2] = stride[i - 1] * std::max<std::int64_t>(shape[i - 1], 1);
    }
    return stride;
}

} // namespace

const char *LayoutNormalizePass::Name() const {
    return "layout_normalize";
}

VkStatus LayoutNormalizePass::Run(GpuVkTensorGraph &graph) {
    for (auto &value : graph.MutableValues()) {
        if (value.stride.empty() && !value.shape.empty()) {
            value.stride = ComputeStride(value.shape);
        }
    }
    return VkStatus::kSuccess;
}

} // namespace npu::tile_fwk::gpu_vk
