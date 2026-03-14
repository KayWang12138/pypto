#include "gpu_vk/runtime/vk_instance.h"

#if defined(BUILD_WITH_VULKAN)
#include <vulkan/vulkan.h>

#include <string>
#include <vector>
#endif

namespace npu::tile_fwk::gpu_vk {

VkStatus VkInstanceContext::Initialize(bool enableValidation) {
#if defined(BUILD_WITH_VULKAN)
    Destroy();

    std::vector<const char *> enabledLayers;
    validationEnabled_ = false;
    if (enableValidation) {
        std::uint32_t layerCount = 0;
        if (vkEnumerateInstanceLayerProperties(&layerCount, nullptr) == VK_SUCCESS && layerCount > 0) {
            std::vector<VkLayerProperties> layerProperties(layerCount);
            if (vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data()) == VK_SUCCESS) {
                for (const auto &layer : layerProperties) {
                    if (std::string(layer.layerName) == "VK_LAYER_KHRONOS_validation") {
                        enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
                        validationEnabled_ = true;
                        break;
                    }
                }
            }
        }
    } else {
        validationEnabled_ = false;
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "PyPTO GPU VK";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.pEngineName = "PyPTO";
    appInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = static_cast<std::uint32_t>(enabledLayers.size());
    createInfo.ppEnabledLayerNames = enabledLayers.empty() ? nullptr : enabledLayers.data();

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS || instance == VK_NULL_HANDLE) {
        available_ = false;
        nativeHandle_ = 0;
        return VkStatus::kUnavailable;
    }

    available_ = true;
    nativeHandle_ = reinterpret_cast<std::uintptr_t>(instance);
    return VkStatus::kSuccess;
#else
    validationEnabled_ = enableValidation;
    available_ = false;
    nativeHandle_ = 0;
    return VkStatus::kNotSupported;
#endif
}

void VkInstanceContext::Destroy() {
#if defined(BUILD_WITH_VULKAN)
    if (nativeHandle_ != 0) {
        vkDestroyInstance(reinterpret_cast<VkInstance>(nativeHandle_), nullptr);
    }
#endif
    available_ = false;
    nativeHandle_ = 0;
    validationEnabled_ = false;
}

} // namespace npu::tile_fwk::gpu_vk
