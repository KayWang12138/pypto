#include "gpu_vk/runtime/vk_device.h"

#if defined(BUILD_WITH_VULKAN)
#include <vulkan/vulkan.h>

#include <vector>
#endif

namespace npu::tile_fwk::gpu_vk {

VkStatus VkDeviceContext::Initialize(std::uintptr_t instanceHandle) {
#if defined(BUILD_WITH_VULKAN)
    Destroy();
    instanceHandle_ = instanceHandle;
    if (instanceHandle == 0) {
        return VkStatus::kInvalidArgument;
    }

    auto *instance = reinterpret_cast<VkInstance>(instanceHandle);
    std::uint32_t physicalDeviceCount = 0;
    if (vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr) != VK_SUCCESS || physicalDeviceCount == 0) {
        return VkStatus::kDeviceNotFound;
    }

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    if (vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data()) != VK_SUCCESS) {
        return VkStatus::kDeviceNotFound;
    }

    VkPhysicalDevice selectedPhysicalDevice = VK_NULL_HANDLE;
    std::uint32_t selectedQueueFamily = 0;
    int selectedScore = -1;
    for (VkPhysicalDevice physicalDevice : physicalDevices) {
        std::uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        if (queueFamilyCount == 0) {
            continue;
        }

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        std::uint32_t computeQueueFamily = queueFamilyCount;
        for (std::uint32_t queueFamily = 0; queueFamily < queueFamilyCount; ++queueFamily) {
            if ((queueFamilies[queueFamily].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                computeQueueFamily = queueFamily;
                break;
            }
        }
        if (computeQueueFamily == queueFamilyCount) {
            continue;
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        int score = 10;
        switch (properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                score = 400;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                score = 300;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                score = 200;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                score = 100;
                break;
            default:
                break;
        }
        if (score > selectedScore) {
            selectedScore = score;
            selectedPhysicalDevice = physicalDevice;
            selectedQueueFamily = computeQueueFamily;
        }
    }

    if (selectedPhysicalDevice == VK_NULL_HANDLE) {
        return VkStatus::kDeviceNotFound;
    }

    const float queuePriority = 1.0F;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = selectedQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    VkDevice device = VK_NULL_HANDLE;
    const VkResult createResult = vkCreateDevice(selectedPhysicalDevice, &deviceCreateInfo, nullptr, &device);
    if (createResult != VK_SUCCESS || device == VK_NULL_HANDLE) {
        return VkStatus::kUnavailable;
    }

    VkQueue computeQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, selectedQueueFamily, 0, &computeQueue);

    available_ = true;
    nativeHandle_ = reinterpret_cast<std::uintptr_t>(device);
    physicalDeviceHandle_ = reinterpret_cast<std::uintptr_t>(selectedPhysicalDevice);
    computeQueueHandle_ = reinterpret_cast<std::uintptr_t>(computeQueue);
    computeQueueFamily_ = selectedQueueFamily;
    return VkStatus::kSuccess;
#else
    instanceHandle_ = instanceHandle;
    available_ = false;
    nativeHandle_ = 0;
    physicalDeviceHandle_ = 0;
    computeQueueHandle_ = 0;
    computeQueueFamily_ = 0;
    return instanceHandle == 0 ? VkStatus::kInvalidArgument : VkStatus::kNotSupported;
#endif
}

void VkDeviceContext::Destroy() {
#if defined(BUILD_WITH_VULKAN)
    if (nativeHandle_ != 0) {
        vkDestroyDevice(reinterpret_cast<VkDevice>(nativeHandle_), nullptr);
    }
#endif
    available_ = false;
    instanceHandle_ = 0;
    nativeHandle_ = 0;
    physicalDeviceHandle_ = 0;
    computeQueueHandle_ = 0;
    computeQueueFamily_ = 0;
}

} // namespace npu::tile_fwk::gpu_vk
