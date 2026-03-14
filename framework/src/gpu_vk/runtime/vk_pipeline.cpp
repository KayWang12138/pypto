#include "gpu_vk/runtime/vk_pipeline.h"

#if defined(BUILD_WITH_VULKAN)
#include <vulkan/vulkan.h>
#endif

namespace npu::tile_fwk::gpu_vk {

VkStatus VkComputePipelineHandle::Create(
    std::uintptr_t deviceHandle,
    const std::string &kernelName,
    const std::string &entryPoint,
    const std::vector<std::uint32_t> &spirv,
    const std::vector<VkBindingDesc> &bindings,
    std::uint32_t pushConstantBytes) {
#if defined(BUILD_WITH_VULKAN)
    Destroy();
    if (deviceHandle == 0 || kernelName.empty() || entryPoint.empty() || spirv.empty()) {
        return VkStatus::kInvalidArgument;
    }

    const VkDevice device = reinterpret_cast<VkDevice>(deviceHandle);
    std::vector<VkDescriptorSetLayoutBinding> descriptorBindings;
    descriptorBindings.reserve(bindings.size());
    for (const auto &binding : bindings) {
        descriptorBindings.push_back(VkDescriptorSetLayoutBinding{
            binding.binding,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr,
        });
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = static_cast<std::uint32_t>(descriptorBindings.size());
    descriptorSetLayoutCreateInfo.pBindings = descriptorBindings.empty() ? nullptr : descriptorBindings.data();

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        return VkStatus::kDescriptorCreateFailed;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = pushConstantBytes;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    if (pushConstantBytes > 0) {
        pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    }

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        return VkStatus::kPipelineLayoutCreateFailed;
    }

    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = spirv.size() * sizeof(std::uint32_t);
    shaderModuleCreateInfo.pCode = spirv.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        return VkStatus::kShaderModuleCreateFailed;
    }

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo{};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = shaderModule;
    shaderStageCreateInfo.pName = entryPoint.c_str();

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.stage = shaderStageCreateInfo;
    computePipelineCreateInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        return VkStatus::kPipelineCreateFailed;
    }

    deviceHandle_ = deviceHandle;
    kernelName_ = kernelName;
    bindings_ = bindings;
    shaderModuleHandle_ = reinterpret_cast<std::uintptr_t>(shaderModule);
    descriptorSetLayoutHandle_ = reinterpret_cast<std::uintptr_t>(descriptorSetLayout);
    pipelineLayoutHandle_ = reinterpret_cast<std::uintptr_t>(pipelineLayout);
    pipelineHandle_ = reinterpret_cast<std::uintptr_t>(pipeline);
    created_ = true;
    return VkStatus::kSuccess;
#else
    (void)deviceHandle;
    (void)kernelName;
    (void)entryPoint;
    (void)spirv;
    (void)bindings;
    (void)pushConstantBytes;
    return VkStatus::kNotSupported;
#endif
}

void VkComputePipelineHandle::Destroy() {
#if defined(BUILD_WITH_VULKAN)
    if (deviceHandle_ != 0 && pipelineHandle_ != 0) {
        vkDestroyPipeline(reinterpret_cast<VkDevice>(deviceHandle_), reinterpret_cast<VkPipeline>(pipelineHandle_), nullptr);
    }
    if (deviceHandle_ != 0 && pipelineLayoutHandle_ != 0) {
        vkDestroyPipelineLayout(
            reinterpret_cast<VkDevice>(deviceHandle_), reinterpret_cast<VkPipelineLayout>(pipelineLayoutHandle_), nullptr);
    }
    if (deviceHandle_ != 0 && descriptorSetLayoutHandle_ != 0) {
        vkDestroyDescriptorSetLayout(
            reinterpret_cast<VkDevice>(deviceHandle_),
            reinterpret_cast<VkDescriptorSetLayout>(descriptorSetLayoutHandle_),
            nullptr);
    }
    if (deviceHandle_ != 0 && shaderModuleHandle_ != 0) {
        vkDestroyShaderModule(
            reinterpret_cast<VkDevice>(deviceHandle_), reinterpret_cast<VkShaderModule>(shaderModuleHandle_), nullptr);
    }
#endif
    created_ = false;
    deviceHandle_ = 0;
    shaderModuleHandle_ = 0;
    descriptorSetLayoutHandle_ = 0;
    pipelineLayoutHandle_ = 0;
    pipelineHandle_ = 0;
    kernelName_.clear();
    bindings_.clear();
}

} // namespace npu::tile_fwk::gpu_vk
