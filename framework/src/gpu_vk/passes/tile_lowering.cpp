#include "gpu_vk/passes/tile_lowering.h"

#include <algorithm>

namespace npu::tile_fwk::gpu_vk {

namespace {

struct MatmulDispatchShape {
    std::uint32_t batch{1};
    std::uint32_t rows{1};
    std::uint32_t cols{1};
};

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

bool IsMatmulGraph(const GpuVkTensorGraph &graph) {
    return graph.NodeCount() == 1 && !graph.Nodes().empty() && graph.Nodes().front().op == GpuVkOpKind::MATMUL;
}

bool TryGetMatmulDispatchShape(const GpuVkTensorGraph &graph, MatmulDispatchShape &shape) {
    if (graph.InputNames().size() != 2) {
        return false;
    }

    const GpuVkValue *lhsValue = graph.FindValue(graph.InputNames()[0]);
    const GpuVkValue *rhsValue = graph.FindValue(graph.InputNames()[1]);
    if (lhsValue == nullptr || rhsValue == nullptr) {
        return false;
    }

    const auto &lhsShape = lhsValue->shape;
    const auto &rhsShape = rhsValue->shape;
    if (lhsShape.empty() || lhsShape.size() > 3 || rhsShape.empty() || rhsShape.size() > 3) {
        return false;
    }

    const std::uint32_t lhsBatch = lhsShape.size() == 3
        ? static_cast<std::uint32_t>(std::max<std::int64_t>(1, lhsShape.front()))
        : 1;
    const std::uint32_t rhsBatch = rhsShape.size() == 3
        ? static_cast<std::uint32_t>(std::max<std::int64_t>(1, rhsShape.front()))
        : 1;
    shape.batch = std::max(lhsBatch, rhsBatch);
    shape.rows = lhsShape.size() == 1
        ? 1
        : static_cast<std::uint32_t>(std::max<std::int64_t>(1, lhsShape[lhsShape.size() - 2]));
    shape.cols = rhsShape.size() == 1
        ? 1
        : static_cast<std::uint32_t>(std::max<std::int64_t>(1, rhsShape.back()));
    return true;
}

} // namespace

VkStatus TileLoweringPass::Run(const GpuVkTensorGraph &graph, GpuVkDispatchGraph &dispatchGraph) const {
    if (graph.Empty()) {
        return VkStatus::kInvalidArgument;
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
    MatmulDispatchShape matmulShape;
    if (IsMatmulGraph(graph) && TryGetMatmulDispatchShape(graph, matmulShape)) {
        dispatch.localX = 16;
        dispatch.localY = 16;
        dispatch.groupX = std::max<std::uint32_t>(1, (matmulShape.cols + dispatch.localX - 1) / dispatch.localX);
        dispatch.groupY = std::max<std::uint32_t>(1, (matmulShape.rows + dispatch.localY - 1) / dispatch.localY);
        dispatch.groupZ = std::max<std::uint32_t>(1, matmulShape.batch);
    } else {
        dispatch.localX = 64;
        dispatch.groupX = std::max<std::uint32_t>(1, (elementCount + dispatch.localX - 1) / dispatch.localX);
    }
    dispatchGraph.SetDispatch(dispatch);
    return VkStatus::kSuccess;
}

} // namespace npu::tile_fwk::gpu_vk
