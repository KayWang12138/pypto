#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_vk/common/vk_status.h"

namespace npu::tile_fwk::gpu_vk {

struct VkBindingDesc {
    std::uint32_t binding;
    std::uint32_t descriptorType;
    bool isOutput;
};

struct VkLaunchSpec {
    std::uint32_t localX{1};
    std::uint32_t localY{1};
    std::uint32_t localZ{1};
    std::uint32_t groupX{1};
    std::uint32_t groupY{1};
    std::uint32_t groupZ{1};
};

class VkComputePipelineHandle {
public:
    VkStatus Create(
        const std::string &kernelName,
        const std::vector<std::uint32_t> &spirv,
        const std::vector<VkBindingDesc> &bindings);
    void Destroy();

    bool Created() const { return created_; }
    const std::string &KernelName() const { return kernelName_; }
    const std::vector<VkBindingDesc> &Bindings() const { return bindings_; }

private:
    bool created_{false};
    std::string kernelName_;
    std::vector<VkBindingDesc> bindings_;
};

} // namespace npu::tile_fwk::gpu_vk
