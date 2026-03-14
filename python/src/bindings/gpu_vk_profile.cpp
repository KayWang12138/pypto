#include "pybind_common.h"

#include "gpu_vk/profile/vk_profiler.h"
#include "gpu_vk/profile/workgroup_tuner.h"

namespace pypto {

using npu::tile_fwk::gpu_vk::VkKernelProfile;
using npu::tile_fwk::gpu_vk::VkProfiler;
using npu::tile_fwk::gpu_vk::WorkgroupCandidate;
using npu::tile_fwk::gpu_vk::WorkgroupTuner;

void BindGpuVkProfile(py::module &m) {
    py::class_<VkKernelProfile>(m, "GpuVkKernelProfile")
        .def(py::init<>())
        .def_readwrite("dispatch_ns", &VkKernelProfile::dispatchNs)
        .def_readwrite("group_x", &VkKernelProfile::groupX)
        .def_readwrite("group_y", &VkKernelProfile::groupY)
        .def_readwrite("group_z", &VkKernelProfile::groupZ)
        .def_readwrite("local_x", &VkKernelProfile::localX)
        .def_readwrite("local_y", &VkKernelProfile::localY)
        .def_readwrite("local_z", &VkKernelProfile::localZ);

    py::class_<WorkgroupCandidate>(m, "GpuVkWorkgroupCandidate")
        .def(py::init<>())
        .def_readwrite("local_x", &WorkgroupCandidate::localX)
        .def_readwrite("local_y", &WorkgroupCandidate::localY)
        .def_readwrite("local_z", &WorkgroupCandidate::localZ);

    py::class_<VkProfiler>(m, "GpuVkProfiler")
        .def(py::init<>())
        .def("begin", &VkProfiler::Begin)
        .def("end", &VkProfiler::End)
        .def("read", &VkProfiler::Read)
        .def("set_launch_shape", &VkProfiler::SetLaunchShape);

    py::class_<WorkgroupTuner>(m, "GpuVkWorkgroupTuner")
        .def(py::init<>())
        .def("generate_candidates", &WorkgroupTuner::GenerateCandidates)
        .def("select_best", &WorkgroupTuner::SelectBest);
}

} // namespace pypto
