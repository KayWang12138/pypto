#include "gpu_vk/passes/dispatch_lowering.h"

namespace npu::tile_fwk::gpu_vk {

namespace {

std::string LoadValue(const std::vector<std::string> &inputs, std::size_t index) {
    return index < inputs.size() ? inputs[index] + "[idx]" : "0.0";
}

std::string BuildExpr(const GpuVkNode &node) {
    switch (node.op) {
        case GpuVkOpKind::ADD:
            return LoadValue(node.inputs, 0) + " + " + LoadValue(node.inputs, 1);
        case GpuVkOpKind::MUL:
            return LoadValue(node.inputs, 0) + " * " + LoadValue(node.inputs, 1);
        case GpuVkOpKind::RELU:
            return "max(" + LoadValue(node.inputs, 0) + ", 0.0)";
        case GpuVkOpKind::WHERE:
            return LoadValue(node.inputs, 0) + " != 0.0 ? " + LoadValue(node.inputs, 1) + " : " + LoadValue(node.inputs, 2);
        case GpuVkOpKind::SUM:
            return LoadValue(node.inputs, 0);
        case GpuVkOpKind::MATMUL:
            return LoadValue(node.inputs, 0);
        case GpuVkOpKind::INPUT:
        case GpuVkOpKind::OUTPUT:
        case GpuVkOpKind::UNKNOWN:
        default:
            return LoadValue(node.inputs, 0);
    }
}

} // namespace

VkStatus DispatchLoweringPass::Run(
    const GpuVkTensorGraph &graph, const GpuVkDispatchGraph &dispatchGraph, GpuVkShaderFunction &shaderFunction) const {
    if (graph.Empty()) {
        return VkStatus::kInvalidArgument;
    }

    shaderFunction.SetName(dispatchGraph.KernelName());
    shaderFunction.SetDispatch(dispatchGraph.Dispatch());
    for (const auto &node : graph.Nodes()) {
        shaderFunction.AddOp(GpuVkShaderOp{
            node.op,
            BuildExpr(node),
            node.outputs.empty() ? std::string{} : node.outputs.front(),
        });
    }
    return VkStatus::kSuccess;
}

} // namespace npu::tile_fwk::gpu_vk
