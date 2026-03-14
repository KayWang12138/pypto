#include "gpu_vk/passes/dispatch_lowering.h"

namespace npu::tile_fwk::gpu_vk {

namespace {

std::string BuildExpr(const GpuVkNode &node) {
    switch (node.op) {
        case GpuVkOpKind::ADD:
            return "input0[idx] + input1[idx]";
        case GpuVkOpKind::MUL:
            return "input0[idx] * input1[idx]";
        case GpuVkOpKind::RELU:
            return "max(input0[idx], 0.0)";
        case GpuVkOpKind::WHERE:
            return "cond[idx] != 0.0 ? input0[idx] : input1[idx]";
        case GpuVkOpKind::SUM:
            return "input0[idx]";
        case GpuVkOpKind::MATMUL:
            return "input0[idx]";
        case GpuVkOpKind::INPUT:
        case GpuVkOpKind::OUTPUT:
        case GpuVkOpKind::UNKNOWN:
        default:
            return "input0[idx]";
    }
}

} // namespace

VkStatus DispatchLoweringPass::Run(
    const GpuVkTensorGraph &graph, const GpuVkDispatchGraph &dispatchGraph, GpuVkShaderFunction &shaderFunction) const {
    if (graph.Empty()) {
        return VkStatus::INVALID_ARGUMENT;
    }

    shaderFunction.SetName(dispatchGraph.KernelName());
    shaderFunction.SetDispatch(dispatchGraph.Dispatch());
    for (const auto &node : graph.Nodes()) {
        shaderFunction.AddOp(GpuVkShaderOp{node.op, BuildExpr(node)});
    }
    return VkStatus::SUCCESS;
}

} // namespace npu::tile_fwk::gpu_vk
