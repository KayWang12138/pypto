#include "pybind_common.h"

#include <algorithm>
#include <cstdint>
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

} // namespace

void BindGpuVkRuntime(py::module &m) {
    py::enum_<VkStatus>(m, "GpuVkStatus")
        .value("SUCCESS", VkStatus::SUCCESS)
        .value("UNAVAILABLE", VkStatus::UNAVAILABLE)
        .value("INVALID_ARGUMENT", VkStatus::INVALID_ARGUMENT)
        .value("OUT_OF_MEMORY", VkStatus::OUT_OF_MEMORY)
        .value("NOT_SUPPORTED", VkStatus::NOT_SUPPORTED)
        .value("INTERNAL_ERROR", VkStatus::INTERNAL_ERROR);

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
            [](VkRunner &runner, const npu::tile_fwk::gpu_vk::GpuVkArtifact &artifact, py::sequence inputs) {
                if (inputs.size() == 0) {
                    return VkStatus::INVALID_ARGUMENT;
                }
                std::vector<VkTensorDesc> inputDescs;
                std::vector<const void *> inputPtrs;
                inputDescs.reserve(inputs.size());
                inputPtrs.reserve(inputs.size());
                for (const py::handle &input : inputs) {
                    inputDescs.push_back(BuildTensorDescFromTorch(py::reinterpret_borrow<py::object>(input)));
                    inputPtrs.push_back(reinterpret_cast<const void *>(0x1));
                }

                std::vector<VkTensorDesc> outputDescs{BuildTensorDescFromTorch(py::reinterpret_borrow<py::object>(inputs[0]))};
                std::vector<void *> outputPtrs{reinterpret_cast<void *>(0x1)};
                return runner.Run(artifact, inputDescs, outputDescs, inputPtrs, outputPtrs);
            },
            py::arg("artifact"),
            py::arg("inputs"));
}

} // namespace pypto
