#include "gpu_vk/runtime/vk_buffer.h"

#include <cstring>

#if defined(BUILD_WITH_VULKAN)
#include <vulkan/vulkan.h>
#endif

namespace npu::tile_fwk::gpu_vk {

namespace {

#if defined(BUILD_WITH_VULKAN)
std::uint32_t FindMemoryType(
    VkPhysicalDevice physicalDevice, std::uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        const bool typeMatches = (typeFilter & (1U << index)) != 0;
        const bool propertyMatches =
            (memoryProperties.memoryTypes[index].propertyFlags & properties) == properties;
        if (typeMatches && propertyMatches) {
            return index;
        }
    }
    return UINT32_MAX;
}
#endif

} // namespace

VkStatus VkBufferHandle::Allocate(
    std::uintptr_t deviceHandle,
    std::uintptr_t physicalDeviceHandle,
    std::size_t bytes,
    VkBufferUsageKind usage) {
#if defined(BUILD_WITH_VULKAN)
    Destroy();
    if (deviceHandle == 0 || physicalDeviceHandle == 0 || bytes == 0) {
        return VkStatus::kInvalidArgument;
    }

    const VkDevice device = reinterpret_cast<VkDevice>(deviceHandle);
    const VkPhysicalDevice physicalDevice = reinterpret_cast<VkPhysicalDevice>(physicalDeviceHandle);
    const VkBufferUsageFlags usageFlags = usage == VkBufferUsageKind::STAGING
                                              ? (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                                              : (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const VkMemoryPropertyFlags memoryFlags = usage == VkBufferUsageKind::STAGING
                                                  ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                                  : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = static_cast<VkDeviceSize>(bytes);
    bufferCreateInfo.usage = usageFlags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS || buffer == VK_NULL_HANDLE) {
        return VkStatus::kBufferCreateFailed;
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
    std::uint32_t memoryTypeIndex = FindMemoryType(physicalDevice, memoryRequirements.memoryTypeBits, memoryFlags);
    if (memoryTypeIndex == UINT32_MAX && usage == VkBufferUsageKind::STORAGE) {
        memoryTypeIndex = FindMemoryType(
            physicalDevice,
            memoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (memoryTypeIndex == UINT32_MAX) {
        vkDestroyBuffer(device, buffer, nullptr);
        return VkStatus::kMemoryAllocFailed;
    }

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory) != VK_SUCCESS || memory == VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
        return VkStatus::kMemoryAllocFailed;
    }

    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return VkStatus::kMemoryAllocFailed;
    }

    void *mappedPtr = nullptr;
    if (usage == VkBufferUsageKind::STAGING) {
        if (vkMapMemory(device, memory, 0, memoryAllocateInfo.allocationSize, 0, &mappedPtr) != VK_SUCCESS) {
            vkFreeMemory(device, memory, nullptr);
            vkDestroyBuffer(device, buffer, nullptr);
            return VkStatus::kMemoryAllocFailed;
        }
    }

    usage_ = usage;
    size_ = bytes;
    deviceHandle_ = deviceHandle;
    bufferHandle_ = reinterpret_cast<std::uintptr_t>(buffer);
    memoryHandle_ = reinterpret_cast<std::uintptr_t>(memory);
    mappedPtr_ = mappedPtr;
    return VkStatus::kSuccess;
#else
    (void)deviceHandle;
    (void)physicalDeviceHandle;
    (void)bytes;
    usage_ = usage;
    return VkStatus::kNotSupported;
#endif
}

VkStatus VkBufferHandle::Upload(const void *src, std::size_t bytes) {
    if (src == nullptr) {
        return VkStatus::kInvalidArgument;
    }
    if (bytes > size_) {
        return VkStatus::kOutOfMemory;
    }
    if (mappedPtr_ == nullptr) {
        return VkStatus::kNotSupported;
    }
    if (bytes != 0) {
        std::memcpy(mappedPtr_, src, bytes);
    }
    return VkStatus::kSuccess;
}

VkStatus VkBufferHandle::Download(void *dst, std::size_t bytes) const {
    if (dst == nullptr) {
        return VkStatus::kInvalidArgument;
    }
    if (bytes > size_) {
        return VkStatus::kOutOfMemory;
    }
    if (mappedPtr_ == nullptr) {
        return VkStatus::kNotSupported;
    }
    if (bytes != 0) {
        std::memcpy(dst, mappedPtr_, bytes);
    }
    return VkStatus::kSuccess;
}

void VkBufferHandle::Destroy() {
#if defined(BUILD_WITH_VULKAN)
    if (deviceHandle_ != 0 && memoryHandle_ != 0 && mappedPtr_ != nullptr) {
        vkUnmapMemory(reinterpret_cast<VkDevice>(deviceHandle_), reinterpret_cast<VkDeviceMemory>(memoryHandle_));
    }
    if (deviceHandle_ != 0 && memoryHandle_ != 0) {
        vkFreeMemory(
            reinterpret_cast<VkDevice>(deviceHandle_), reinterpret_cast<VkDeviceMemory>(memoryHandle_), nullptr);
    }
    if (deviceHandle_ != 0 && bufferHandle_ != 0) {
        vkDestroyBuffer(reinterpret_cast<VkDevice>(deviceHandle_), reinterpret_cast<VkBuffer>(bufferHandle_), nullptr);
    }
#endif
    size_ = 0;
    deviceHandle_ = 0;
    bufferHandle_ = 0;
    memoryHandle_ = 0;
    mappedPtr_ = nullptr;
}

} // namespace npu::tile_fwk::gpu_vk
