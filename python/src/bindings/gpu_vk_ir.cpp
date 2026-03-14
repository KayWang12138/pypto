#include "pybind_common.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_vk/ir/gpu_vk_dispatch_ir.h"
#include "gpu_vk/ir/gpu_vk_shader_ir.h"
#include "gpu_vk/ir/gpu_vk_tensor_ir.h"
#include "gpu_vk/passes/gpu_vk_pass_manager.h"

namespace pypto {

using npu::tile_fwk::gpu_vk::GpuVkBufferBinding;
using npu::tile_fwk::gpu_vk::GpuVkDispatchGraph;
using npu::tile_fwk::gpu_vk::GpuVkDispatchParam;
using npu::tile_fwk::gpu_vk::GpuVkNode;
using npu::tile_fwk::gpu_vk::GpuVkOpKind;
using npu::tile_fwk::gpu_vk::GpuVkPassManager;
using npu::tile_fwk::gpu_vk::GpuVkShaderFunction;
using npu::tile_fwk::gpu_vk::GpuVkShaderOp;
using npu::tile_fwk::gpu_vk::GpuVkTensorGraph;
using npu::tile_fwk::gpu_vk::GpuVkValue;
using npu::tile_fwk::gpu_vk::ParseGpuVkOpKind;

namespace {

struct MatmulShapeInfo {
    std::size_t lhsRank{0};
    std::size_t rhsRank{0};
    std::int64_t lhsBatch{1};
    std::int64_t rhsBatch{1};
    std::int64_t batch{1};
    std::int64_t m{1};
    std::int64_t n{1};
    std::int64_t k{1};
};

MatmulShapeInfo AnalyzeMatmulShapes(
    const std::vector<std::int64_t> &lhsShape, const std::vector<std::int64_t> &rhsShape) {
    MatmulShapeInfo info;
    info.lhsRank = lhsShape.size();
    info.rhsRank = rhsShape.size();
    if (info.lhsRank < 1 || info.lhsRank > 3 || info.rhsRank < 1 || info.rhsRank > 3) {
        throw std::invalid_argument("matmul currently supports only rank-1, rank-2 or rank-3 tensors.");
    }

    info.m = info.lhsRank == 1 ? 1 : lhsShape[info.lhsRank - 2];
    info.k = lhsShape.back();
    const std::int64_t rhsK = info.rhsRank == 1 ? rhsShape.front() : rhsShape[info.rhsRank - 2];
    info.n = info.rhsRank == 1 ? 1 : rhsShape.back();
    if (info.k != rhsK) {
        if (info.lhsRank == 2 && info.rhsRank == 2) {
            throw std::invalid_argument("matmul requires lhs.shape[1] == rhs.shape[0].");
        }
        throw std::invalid_argument("matmul requires contracted dimensions to match.");
    }

    info.lhsBatch = info.lhsRank == 3 ? lhsShape.front() : 1;
    info.rhsBatch = info.rhsRank == 3 ? rhsShape.front() : 1;
    if (info.lhsBatch != info.rhsBatch && info.lhsBatch != 1 && info.rhsBatch != 1) {
        throw std::invalid_argument("batched matmul requires compatible batch dimensions.");
    }
    info.batch = std::max(info.lhsBatch, info.rhsBatch);
    return info;
}

std::vector<std::int64_t> InferOutputShape(
    const std::string &opName, const std::vector<std::vector<std::int64_t>> &inputShapes) {
    if (inputShapes.empty()) {
        return {};
    }
    if (opName == "matmul") {
        if (inputShapes.size() != 2) {
            throw std::invalid_argument("matmul expects exactly 2 input tensors.");
        }
        const auto &lhsShape = inputShapes[0];
        const auto &rhsShape = inputShapes[1];
        const MatmulShapeInfo info = AnalyzeMatmulShapes(lhsShape, rhsShape);
        if (info.lhsRank == 1 && info.rhsRank == 1) {
            return {};
        }
        if (info.lhsRank == 1 && info.rhsRank == 2) {
            return {info.n};
        }
        if (info.lhsRank == 2 && info.rhsRank == 1) {
            return {info.m};
        }
        if (info.lhsRank == 1 && info.rhsRank == 3) {
            return {info.batch, info.n};
        }
        if (info.lhsRank == 3 && info.rhsRank == 1) {
            return {info.batch, info.m};
        }
        if (info.lhsRank == 2 && info.rhsRank == 2) {
            return {info.m, info.n};
        }
        return {info.batch, info.m, info.n};
    }
    return inputShapes.front();
}

GpuVkTensorGraph BuildTensorGraphFromShapes(
    const std::string &opName, const std::vector<std::vector<std::int64_t>> &inputShapes, std::int32_t dtype) {
    GpuVkTensorGraph graph;
    graph.SetName(opName + "_graph");

    std::vector<std::string> inputNames;
    for (std::size_t i = 0; i < inputShapes.size(); ++i) {
        const std::string name = "input" + std::to_string(i);
        inputNames.push_back(name);
        graph.AddInput(GpuVkValue{name, inputShapes[i], {}, dtype, true, false});
    }

    const std::vector<std::int64_t> outputShape = InferOutputShape(opName, inputShapes);
    graph.AddOutput(GpuVkValue{"output0", outputShape, {}, dtype, false, true});
    graph.AddNode(ParseGpuVkOpKind(opName), inputNames, {"output0"});
    return graph;
}

GpuVkDispatchGraph LowerToDispatchGraph(GpuVkTensorGraph graph) {
    GpuVkPassManager passManager;
    GpuVkDispatchGraph dispatchGraph;
    GpuVkShaderFunction shaderFunction;
    passManager.LowerToDispatch(graph, dispatchGraph, shaderFunction);
    return dispatchGraph;
}

GpuVkShaderFunction LowerToShaderFunction(GpuVkTensorGraph graph) {
    GpuVkPassManager passManager;
    GpuVkDispatchGraph dispatchGraph;
    GpuVkShaderFunction shaderFunction;
    passManager.LowerToDispatch(graph, dispatchGraph, shaderFunction);
    return shaderFunction;
}

} // namespace

void BindGpuVkIr(py::module &m) {
    py::enum_<GpuVkOpKind>(m, "GpuVkOpKind")
        .value("UNKNOWN", GpuVkOpKind::UNKNOWN)
        .value("INPUT", GpuVkOpKind::INPUT)
        .value("OUTPUT", GpuVkOpKind::OUTPUT)
        .value("ADD", GpuVkOpKind::ADD)
        .value("MUL", GpuVkOpKind::MUL)
        .value("RELU", GpuVkOpKind::RELU)
        .value("WHERE", GpuVkOpKind::WHERE)
        .value("SUM", GpuVkOpKind::SUM)
        .value("MATMUL", GpuVkOpKind::MATMUL);

    py::class_<GpuVkValue>(m, "GpuVkValue")
        .def(py::init<>())
        .def_readwrite("name", &GpuVkValue::name)
        .def_readwrite("shape", &GpuVkValue::shape)
        .def_readwrite("stride", &GpuVkValue::stride)
        .def_readwrite("dtype", &GpuVkValue::dtype)
        .def_readwrite("is_input", &GpuVkValue::isInput)
        .def_readwrite("is_output", &GpuVkValue::isOutput);

    py::class_<GpuVkNode>(m, "GpuVkNode")
        .def(py::init<>())
        .def_readwrite("op", &GpuVkNode::op)
        .def_readwrite("inputs", &GpuVkNode::inputs)
        .def_readwrite("outputs", &GpuVkNode::outputs);

    py::class_<GpuVkTensorGraph>(m, "GpuVkTensorGraph")
        .def(py::init<>())
        .def("set_name", &GpuVkTensorGraph::SetName)
        .def("name", &GpuVkTensorGraph::Name)
        .def("empty", &GpuVkTensorGraph::Empty)
        .def("value_count", &GpuVkTensorGraph::ValueCount)
        .def("node_count", &GpuVkTensorGraph::NodeCount)
        .def("input_names", [](const GpuVkTensorGraph &graph) { return graph.InputNames(); })
        .def("output_names", [](const GpuVkTensorGraph &graph) { return graph.OutputNames(); })
        .def("values", [](const GpuVkTensorGraph &graph) { return graph.Values(); })
        .def("nodes", [](const GpuVkTensorGraph &graph) { return graph.Nodes(); });

    py::class_<GpuVkDispatchParam>(m, "GpuVkDispatchParam")
        .def(py::init<>())
        .def_readwrite("group_x", &GpuVkDispatchParam::groupX)
        .def_readwrite("group_y", &GpuVkDispatchParam::groupY)
        .def_readwrite("group_z", &GpuVkDispatchParam::groupZ)
        .def_readwrite("local_x", &GpuVkDispatchParam::localX)
        .def_readwrite("local_y", &GpuVkDispatchParam::localY)
        .def_readwrite("local_z", &GpuVkDispatchParam::localZ);

    py::class_<GpuVkBufferBinding>(m, "GpuVkBufferBinding")
        .def(py::init<>())
        .def_readwrite("name", &GpuVkBufferBinding::name)
        .def_readwrite("binding", &GpuVkBufferBinding::binding)
        .def_readwrite("is_output", &GpuVkBufferBinding::isOutput);

    py::class_<GpuVkDispatchGraph>(m, "GpuVkDispatchGraph")
        .def(py::init<>())
        .def("kernel_name", &GpuVkDispatchGraph::KernelName)
        .def("bindings", [](const GpuVkDispatchGraph &graph) { return graph.Bindings(); })
        .def("lowered_ops", [](const GpuVkDispatchGraph &graph) { return graph.LoweredOps(); })
        .def("dispatch", &GpuVkDispatchGraph::Dispatch);

    py::class_<GpuVkShaderOp>(m, "GpuVkShaderOp")
        .def(py::init<>())
        .def_readwrite("op", &GpuVkShaderOp::op)
        .def_readwrite("expr", &GpuVkShaderOp::expr);

    py::class_<GpuVkShaderFunction>(m, "GpuVkShaderFunction")
        .def(py::init<>())
        .def("name", &GpuVkShaderFunction::Name)
        .def("dispatch", &GpuVkShaderFunction::Dispatch)
        .def("ops", [](const GpuVkShaderFunction &shaderFunction) { return shaderFunction.Ops(); });

    py::class_<GpuVkPassManager>(m, "GpuVkPassManager")
        .def(py::init<>())
        .def("run_tensor_passes", &GpuVkPassManager::RunTensorPasses)
        .def(
            "lower_to_dispatch",
            [](GpuVkPassManager &passManager, GpuVkTensorGraph graph) {
                GpuVkDispatchGraph dispatchGraph;
                GpuVkShaderFunction shaderFunction;
                passManager.LowerToDispatch(graph, dispatchGraph, shaderFunction);
                return py::make_tuple(dispatchGraph, shaderFunction);
            });

    m.def("GpuVkParseOpKind", &ParseGpuVkOpKind, py::arg("op_name"));
    m.def("GpuVkBuildTensorGraph", &BuildTensorGraphFromShapes, py::arg("op_name"), py::arg("input_shapes"), py::arg("dtype") = 0);
    m.def("GpuVkLowerToDispatch", &LowerToDispatchGraph, py::arg("graph"));
    m.def("GpuVkLowerToShader", &LowerToShaderFunction, py::arg("graph"));
}

} // namespace pypto
