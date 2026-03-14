#include "pybind_common.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_vk/codegen/gpu_vk_artifact.h"
#include "gpu_vk/common/vk_status.h"
#include "gpu_vk/runtime/vk_launcher.h"
#include "gpu_vk/runtime/vk_runner.h"

namespace pypto {

using npu::tile_fwk::gpu_vk::VkLauncher;
using npu::tile_fwk::gpu_vk::VkRunner;
using npu::tile_fwk::gpu_vk::VkStatus;
using npu::tile_fwk::gpu_vk::VkStatusToString;
using npu::tile_fwk::gpu_vk::VkTensorDesc;

namespace {

VkTensorDesc BuildTensorDescFromTorch(const py::object &tensor) {
    VkTensorDesc desc;
    for (const py::handle &item : tensor.attr("shape")) {
        desc.shape.push_back(item.cast<std::int64_t>());
    }
    desc.stride.assign(desc.shape.size(), 1);
    for (std::size_t i = desc.shape.size(); i > 1; --i) {
        desc.stride[i - 2] = desc.stride[i - 1] * std::max<std::int64_t>(desc.shape[i - 1], 1);
    }
    desc.dtype = 0;
    desc.nbytes = static_cast<std::size_t>(
        tensor.attr("numel")().cast<std::int64_t>() * tensor.attr("element_size")().cast<std::int64_t>());
    return desc;
}

std::uintptr_t GetTensorDataPtr(const py::object &tensor) {
    return tensor.attr("data_ptr")().cast<std::uintptr_t>();
}

} // namespace

void BindGpuVkRuntime(py::module &m) {
    py::enum_<VkStatus>(m, "GpuVkStatus")
        .value("SUCCESS", VkStatus::kSuccess)
        .value("UNAVAILABLE", VkStatus::kUnavailable)
        .value("INVALID_ARGUMENT", VkStatus::kInvalidArgument)
        .value("OUT_OF_MEMORY", VkStatus::kOutOfMemory)
        .value("NOT_SUPPORTED", VkStatus::kNotSupported)
        .value("INTERNAL_ERROR", VkStatus::kInternalError)
        .value("DEVICE_NOT_FOUND", VkStatus::kDeviceNotFound)
        .value("TOOL_NOT_FOUND", VkStatus::kToolNotFound)
        .value("SHADER_COMPILE_FAILED", VkStatus::kShaderCompileFailed)
        .value("SHADER_MODULE_CREATE_FAILED", VkStatus::kShaderModuleCreateFailed)
        .value("DESCRIPTOR_CREATE_FAILED", VkStatus::kDescriptorCreateFailed)
        .value("PIPELINE_LAYOUT_CREATE_FAILED", VkStatus::kPipelineLayoutCreateFailed)
        .value("PIPELINE_CREATE_FAILED", VkStatus::kPipelineCreateFailed)
        .value("BUFFER_CREATE_FAILED", VkStatus::kBufferCreateFailed)
        .value("MEMORY_ALLOC_FAILED", VkStatus::kMemoryAllocFailed)
        .value("COMMAND_RECORD_FAILED", VkStatus::kCommandRecordFailed)
        .value("QUEUE_SUBMIT_FAILED", VkStatus::kQueueSubmitFailed)
        .value("DISPATCH_FAILED", VkStatus::kDispatchFailed)
        .value("DOWNLOAD_FAILED", VkStatus::kDownloadFailed);

    m.def("GpuVkStatusToString", [](VkStatus status) { return std::string(VkStatusToString(status)); });

    py::class_<VkTensorDesc>(m, "GpuVkTensorDesc")
        .def(py::init<>())
        .def_readwrite("shape", &VkTensorDesc::shape)
        .def_readwrite("stride", &VkTensorDesc::stride)
        .def_readwrite("dtype", &VkTensorDesc::dtype)
        .def_readwrite("nbytes", &VkTensorDesc::nbytes);

    py::class_<VkLauncher>(m, "GpuVkLauncher")
        .def(py::init<>())
        .def("initialize", &VkLauncher::Initialize, py::arg("enable_validation") = false)
        .def("destroy", &VkLauncher::Destroy)
        .def("available", &VkLauncher::Available)
        .def("run_elementwise_binary",
             [](VkLauncher &launcher, const std::string &op_name, py::object input0, py::object input1) {
                 const std::uint8_t dummy = 0;
                 (void)input1;
                 const VkTensorDesc desc = BuildTensorDescFromTorch(input0);
                 return launcher.RunElementwiseBinary(op_name, &dummy, &dummy, const_cast<std::uint8_t *>(&dummy), desc);
             },
             py::arg("op_name"),
             py::arg("input0"),
             py::arg("input1"))
        .def("run_elementwise_unary",
             [](VkLauncher &launcher, const std::string &op_name, py::object input0) {
                 const std::uint8_t dummy = 0;
                 const VkTensorDesc desc = BuildTensorDescFromTorch(input0);
                 return launcher.RunElementwiseUnary(op_name, &dummy, const_cast<std::uint8_t *>(&dummy), desc);
             },
             py::arg("op_name"),
             py::arg("input0"));

    py::class_<VkRunner>(m, "GpuVkRunner")
        .def(py::init<>())
        .def("initialize", &VkRunner::Initialize, py::arg("enable_validation") = false)
        .def("destroy", &VkRunner::Destroy)
        .def("available", &VkRunner::Available)
        .def("set_enable_cache", &VkRunner::SetEnableCache, py::arg("enable_cache"))
        .def("pipeline_cache_hit_count", &VkRunner::PipelineCacheHitCount)
        .def("pipeline_cache_entry_count", &VkRunner::PipelineCacheEntryCount)
        .def(
            "run_artifact",
            [](VkRunner &runner, const npu::tile_fwk::gpu_vk::GpuVkArtifact &artifact, py::sequence inputs) -> py::tuple {
                if (inputs.size() == 0) {
                    return py::make_tuple(VkStatus::kInvalidArgument, py::none());
                }

                py::module_ torch = py::module_::import("torch");
                std::vector<VkTensorDesc> inputDescs;
                std::vector<const void *> inputPtrs;
                inputDescs.reserve(inputs.size());
                inputPtrs.reserve(inputs.size());
                for (const py::handle &input : inputs) {
                    py::object tensor = py::reinterpret_borrow<py::object>(input);
                    inputDescs.push_back(BuildTensorDescFromTorch(tensor));
                    inputPtrs.push_back(reinterpret_cast<const void *>(GetTensorDataPtr(tensor)));
                }

                py::object outputTensor = torch.attr("empty_like")(py::reinterpret_borrow<py::object>(inputs[0]));
                std::vector<VkTensorDesc> outputDescs{BuildTensorDescFromTorch(outputTensor)};
                std::vector<void *> outputPtrs{reinterpret_cast<void *>(GetTensorDataPtr(outputTensor))};
                const VkStatus status = runner.Run(artifact, inputDescs, outputDescs, inputPtrs, outputPtrs);
                if (status != VkStatus::kSuccess) {
                    return py::make_tuple(status, py::none());
                }
                return py::make_tuple(status, outputTensor);
            },
            py::arg("artifact"),
            py::arg("inputs"));
}

} // namespace pypto
