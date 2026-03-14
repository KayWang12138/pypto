#include "gpu_vk/passes/tile_lowering.h"

#include <algorithm>

namespace npu::tile_fwk::gpu_vk {

namespace {

std::uint32_t ComputeElementCount(const GpuVkValue *value) {
    if (value == nullptr || value->shape.empty()) {
        return 1;
    }
    std::int64_t total = 1;
    for (auto dim : value->shape) {
        total *= std::max<std::int64_t>(dim, 1);
    }
    return static_cast<std::uint32_t>(std::max<std::int64_t>(1, total));
}

} // namespace

VkStatus TileLoweringPass::Run(const GpuVkTensorGraph &graph, GpuVkDispatchGraph &dispatchGraph) const {
    if (graph.Empty()) {
        return VkStatus::INVALID_ARGUMENT;
    }

    dispatchGraph.SetKernelName(graph.Name().empty() ? "gpu_vk_kernel" : graph.Name());
    std::uint32_t bindingIndex = 0;
    for (const auto &name : graph.InputNames()) {
        dispatchGraph.AddBinding(GpuVkBufferBinding{name, bindingIndex++, false});
    }
    for (const auto &name : graph.OutputNames()) {
        dispatchGraph.AddBinding(GpuVkBufferBinding{name, bindingIndex++, true});
    }

    const GpuVkValue *outputValue = nullptr;
    if (!graph.OutputNames().empty()) {
        outputValue = graph.FindValue(graph.OutputNames().front());
    }

    const std::uint32_t elementCount = ComputeElementCount(outputValue);
    GpuVkDispatchParam dispatch;
    dispatch.localX = 64;
    dispatch.groupX = std::max<std::uint32_t>(1, (elementCount + dispatch.localX - 1) / dispatch.localX);
    dispatchGraph.SetDispatch(dispatch);
    return VkStatus::SUCCESS;
}

} // namespace npu::tile_fwk::gpu_vk
