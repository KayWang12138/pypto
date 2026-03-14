#include "pybind_common.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu_vk/codegen/glsl_codegen.h"
#include "gpu_vk/codegen/gpu_vk_artifact.h"
#include "gpu_vk/codegen/shader_meta.h"
#include "gpu_vk/codegen/spirv_compiler.h"
#include "gpu_vk/ir/gpu_vk_dispatch_ir.h"
#include "gpu_vk/ir/gpu_vk_tensor_ir.h"
#include "gpu_vk/passes/gpu_vk_pass_manager.h"

namespace pypto {

using npu::tile_fwk::gpu_vk::GlslComputeCodegen;
using npu::tile_fwk::gpu_vk::GpuVkArtifact;
using npu::tile_fwk::gpu_vk::GpuVkDispatchGraph;
using npu::tile_fwk::gpu_vk::GpuVkPassManager;
using npu::tile_fwk::gpu_vk::GpuVkShaderFunction;
using npu::tile_fwk::gpu_vk::GpuVkTensorGraph;
using npu::tile_fwk::gpu_vk::PushConstantMeta;
using npu::tile_fwk::gpu_vk::ShaderBindingMeta;
using npu::tile_fwk::gpu_vk::ShaderMeta;
using npu::tile_fwk::gpu_vk::SpirvCompileOptions;
using npu::tile_fwk::gpu_vk::SpirvCompiler;
using npu::tile_fwk::gpu_vk::SpirvCompileResult;
using npu::tile_fwk::gpu_vk::VkStatus;
using npu::tile_fwk::gpu_vk::VkStatusToString;

namespace {

bool IsMatmulDispatch(const GpuVkDispatchGraph &dispatchGraph) {
    for (const auto &expr : dispatchGraph.LoweredOps()) {
        if (expr == "matmul_f32") {
            return true;
        }
    }
    return false;
}

ShaderMeta BuildShaderMeta(const GpuVkDispatchGraph &dispatchGraph, std::int32_t dtype) {
    ShaderMeta meta;
    meta.kernelName = dispatchGraph.KernelName();
    meta.dispatch = dispatchGraph.Dispatch();
    for (const auto &binding : dispatchGraph.Bindings()) {
        meta.bindings.push_back(ShaderBindingMeta{binding.binding, binding.name, dtype, binding.isOutput});
    }
    if (IsMatmulDispatch(dispatchGraph)) {
        meta.pushConstants.push_back(PushConstantMeta{"B", 0, 4});
        meta.pushConstants.push_back(PushConstantMeta{"M", 4, 4});
        meta.pushConstants.push_back(PushConstantMeta{"N", 8, 4});
        meta.pushConstants.push_back(PushConstantMeta{"K", 12, 4});
        meta.pushConstants.push_back(PushConstantMeta{"lhsBatchStride", 16, 4});
        meta.pushConstants.push_back(PushConstantMeta{"rhsBatchStride", 20, 4});
    } else {
        meta.pushConstants.push_back(PushConstantMeta{"numel", 0, 4});
    }
    return meta;
}

void ThrowCompileFailure(VkStatus status, const std::string &detail) {
    std::ostringstream builder;
    builder << "GPU Vulkan compile failed: " << VkStatusToString(status);
    if (!detail.empty()) {
        builder << ": " << detail;
    }
    throw std::runtime_error(builder.str());
}

GpuVkArtifact CompileArtifact(
    GpuVkTensorGraph graph,
    bool optimize,
    bool debugInfo,
    const std::string &validatorPath,
    bool dumpArtifacts,
    const std::string &dumpDir) {
    GpuVkPassManager passManager;
    GpuVkDispatchGraph dispatchGraph;
    GpuVkShaderFunction shaderFunction;
    passManager.LowerToDispatch(graph, dispatchGraph, shaderFunction);

    GpuVkArtifact artifact;
    artifact.shaderFunction = shaderFunction;
    artifact.meta = BuildShaderMeta(dispatchGraph, 0);

    GlslComputeCodegen codegen;
    artifact.glsl = codegen.Emit(shaderFunction, artifact.meta);

    SpirvCompiler compiler;
    SpirvCompileResult result;
    const VkStatus status = compiler.CompileGlslToSpirv(
        artifact.glsl,
        SpirvCompileOptions{optimize, debugInfo, validatorPath, dumpArtifacts, dumpDir},
        result);
    artifact.spirv = std::move(result.spirv);
    artifact.glslPath = std::move(result.glslPath);
    artifact.spirvPath = std::move(result.spirvPath);
    artifact.compileLog = std::move(result.compileLog);
    artifact.sourceHash = std::move(result.sourceHash);
    artifact.isRealSpirv = result.isRealSpirv;
    if (status != VkStatus::kSuccess) {
        ThrowCompileFailure(status, artifact.compileLog);
    }

    std::ostringstream summary;
    summary << artifact.meta.kernelName << " bindings=" << artifact.meta.bindings.size()
            << " spirv_words=" << artifact.spirv.size() << " real_spirv=" << artifact.isRealSpirv;
    artifact.debugSummary = summary.str();
    return artifact;
}

std::vector<std::uint32_t> CompileSpirv(
    const std::string &glsl,
    bool optimize,
    bool debugInfo,
    const std::string &validatorPath,
    bool dumpArtifacts,
    const std::string &dumpDir) {
    SpirvCompiler compiler;
    SpirvCompileResult result;
    const VkStatus status = compiler.CompileGlslToSpirv(
        glsl, SpirvCompileOptions{optimize, debugInfo, validatorPath, dumpArtifacts, dumpDir}, result);
    if (status != VkStatus::kSuccess) {
        ThrowCompileFailure(status, result.compileLog);
    }
    return result.spirv;
}

} // namespace

void BindGpuVkCodegen(py::module &m) {
    py::class_<ShaderBindingMeta>(m, "GpuVkShaderBindingMeta")
        .def(py::init<>())
        .def_readwrite("binding", &ShaderBindingMeta::binding)
        .def_readwrite("name", &ShaderBindingMeta::name)
        .def_readwrite("dtype", &ShaderBindingMeta::dtype)
        .def_readwrite("is_output", &ShaderBindingMeta::isOutput);

    py::class_<PushConstantMeta>(m, "GpuVkPushConstantMeta")
        .def(py::init<>())
        .def_readwrite("name", &PushConstantMeta::name)
        .def_readwrite("offset", &PushConstantMeta::offset)
        .def_readwrite("size", &PushConstantMeta::size);

    py::class_<ShaderMeta>(m, "GpuVkShaderMeta")
        .def(py::init<>())
        .def_readwrite("kernel_name", &ShaderMeta::kernelName)
        .def_readwrite("bindings", &ShaderMeta::bindings)
        .def_readwrite("push_constants", &ShaderMeta::pushConstants)
        .def_readwrite("dispatch", &ShaderMeta::dispatch);

    py::class_<SpirvCompileOptions>(m, "GpuVkSpirvCompileOptions")
        .def(py::init<>())
        .def_readwrite("optimize", &SpirvCompileOptions::optimize)
        .def_readwrite("debug_info", &SpirvCompileOptions::debugInfo)
        .def_readwrite("validator_path", &SpirvCompileOptions::validatorPath)
        .def_readwrite("dump_artifacts", &SpirvCompileOptions::dumpArtifacts)
        .def_readwrite("dump_dir", &SpirvCompileOptions::dumpDir);

    py::class_<GpuVkArtifact>(m, "GpuVkArtifact")
        .def(py::init<>())
        .def_readwrite("meta", &GpuVkArtifact::meta)
        .def_readwrite("shader_function", &GpuVkArtifact::shaderFunction)
        .def_readwrite("entry_point", &GpuVkArtifact::entryPoint)
        .def_readwrite("glsl", &GpuVkArtifact::glsl)
        .def_readwrite("spirv", &GpuVkArtifact::spirv)
        .def_readwrite("glsl_path", &GpuVkArtifact::glslPath)
        .def_readwrite("spirv_path", &GpuVkArtifact::spirvPath)
        .def_readwrite("compile_log", &GpuVkArtifact::compileLog)
        .def_readwrite("source_hash", &GpuVkArtifact::sourceHash)
        .def_readwrite("is_real_spirv", &GpuVkArtifact::isRealSpirv)
        .def_readwrite("debug_summary", &GpuVkArtifact::debugSummary)
        .def("empty", &GpuVkArtifact::Empty);

    m.def("GpuVkBuildShaderMeta", &BuildShaderMeta, py::arg("dispatch_graph"), py::arg("dtype") = 0);
    m.def(
        "GpuVkEmitGlsl",
        [](const GpuVkShaderFunction &shaderFunction, const ShaderMeta &meta) {
            GlslComputeCodegen codegen;
            return codegen.Emit(shaderFunction, meta);
        },
        py::arg("shader_function"),
        py::arg("meta"));
    m.def(
        "GpuVkCompileSpirv",
        &CompileSpirv,
        py::arg("glsl"),
        py::arg("optimize") = true,
        py::arg("debug_info") = false,
        py::arg("validator_path") = "",
        py::arg("dump_artifacts") = false,
        py::arg("dump_dir") = "");
    m.def(
        "GpuVkCompileArtifact",
        &CompileArtifact,
        py::arg("graph"),
        py::arg("optimize") = true,
        py::arg("debug_info") = false,
        py::arg("validator_path") = "",
        py::arg("dump_artifacts") = false,
        py::arg("dump_dir") = "");
}

} // namespace pypto
