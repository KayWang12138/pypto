#include "gpu_vk/runtime/vk_runner.h"

#include <algorithm>
#include <sstream>

#if defined(BUILD_WITH_VULKAN)
#include <vulkan/vulkan.h>
#endif

namespace npu::tile_fwk::gpu_vk {

namespace {

std::uint32_t ComputeNumel(const VkTensorDesc &desc) {
    if (!desc.shape.empty()) {
        std::int64_t numel = 1;
        for (const std::int64_t dim : desc.shape) {
            numel *= std::max<std::int64_t>(dim, 1);
        }
        return static_cast<std::uint32_t>(std::max<std::int64_t>(1, numel));
    }
    return static_cast<std::uint32_t>(std::max<std::size_t>(1, desc.nbytes / sizeof(float)));
}

std::vector<GpuVkBufferBinding> BuildDescriptorBindings(const GpuVkArtifact &artifact) {
    std::vector<GpuVkBufferBinding> bindings;
    bindings.reserve(artifact.meta.bindings.size());
    for (const auto &binding : artifact.meta.bindings) {
        bindings.push_back(GpuVkBufferBinding{binding.name, binding.binding, binding.isOutput});
    }
    return bindings;
}

} // namespace

VkStatus VkRunner::Initialize(bool enableValidation) {
    Destroy();

    validationEnabled_ = enableValidation;
    const VkStatus instanceStatus = instanceContext_.Initialize(enableValidation);
    if (instanceStatus != VkStatus::kSuccess) {
        validationEnabled_ = false;
        return instanceStatus;
    }

    const VkStatus deviceStatus = deviceContext_.Initialize(instanceContext_.NativeHandle());
    if (deviceStatus != VkStatus::kSuccess) {
        instanceContext_.Destroy();
        validationEnabled_ = false;
        return deviceStatus;
    }

    const VkStatus allocatorStatus =
        allocator_.Initialize(deviceContext_.NativeHandle(), deviceContext_.PhysicalDeviceHandle());
    if (allocatorStatus != VkStatus::kSuccess) {
        deviceContext_.Destroy();
        instanceContext_.Destroy();
        validationEnabled_ = false;
        return allocatorStatus;
    }

    validationEnabled_ = instanceContext_.ValidationEnabled();
    available_ = true;
    return VkStatus::kSuccess;
}

VkStatus VkRunner::Run(
    const GpuVkArtifact &artifact,
    const std::vector<VkTensorDesc> &inputs,
    const std::vector<VkTensorDesc> &outputs,
    const std::vector<const void *> &hostInputPtrs,
    const std::vector<void *> &hostOutputPtrs) {
    (void)outputs;
    (void)hostInputPtrs;
    (void)hostOutputPtrs;
    if (!available_) {
        return VkStatus::kUnavailable;
    }
    if (artifact.Empty() || inputs.empty()) {
        return VkStatus::kInvalidArgument;
    }

#if defined(BUILD_WITH_VULKAN)
    if (outputs.size() != 1 || hostOutputPtrs.size() != 1 || hostOutputPtrs[0] == nullptr) {
        return VkStatus::kInvalidArgument;
    }
    if (inputs.size() != hostInputPtrs.size()) {
        return VkStatus::kInvalidArgument;
    }

    const PipelineCacheKey pipelineKey{artifact.meta.kernelName, BuildShapeSignature(inputs)};
    VkComputePipelineHandle *cachedPipeline = nullptr;
    if (!pipelineCache_.Find(pipelineKey, cachedPipeline)) {
        auto pipeline = std::make_unique<VkComputePipelineHandle>();
        std::vector<VkBindingDesc> bindings;
        bindings.reserve(artifact.meta.bindings.size());
        for (const auto &binding : artifact.meta.bindings) {
            bindings.push_back(VkBindingDesc{binding.binding, 0, binding.isOutput});
        }
        const VkStatus pipelineStatus = pipeline->Create(
            deviceContext_.NativeHandle(),
            artifact.meta.kernelName,
            artifact.entryPoint,
            artifact.spirv,
            bindings,
            sizeof(std::uint32_t));
        if (pipelineStatus != VkStatus::kSuccess) {
            return pipelineStatus;
        }
        cachedPipeline = pipeline.get();
        pipelineCache_.Insert(pipelineKey, std::move(pipeline));
    }

    descriptorCache_.GetOrCreate(BuildDescriptorBindings(artifact));
    if (enableCache_) {
        const std::string artifactKey = BuildArtifactKey(artifact, inputs);
        artifactCache_.Insert(artifactKey, artifact);
        shapeCache_.Insert(artifactKey, pipelineKey.shapeSignature);
    }

    const VkDevice device = reinterpret_cast<VkDevice>(deviceContext_.NativeHandle());
    const VkQueue queue = reinterpret_cast<VkQueue>(deviceContext_.ComputeQueueHandle());

    std::vector<VkBufferHandle> inputStagingBuffers(inputs.size());
    std::vector<VkBufferHandle> inputStorageBuffers(inputs.size());
    VkBufferHandle outputStorageBuffer;
    VkBufferHandle outputStagingBuffer;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    auto finish = [&](VkStatus status) {
        if (device != VK_NULL_HANDLE) {
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, fence, nullptr);
            }
            if (commandBuffer != VK_NULL_HANDLE && commandPool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
            }
            if (commandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, commandPool, nullptr);
            }
            if (descriptorPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            }
        }
        outputStagingBuffer.Destroy();
        outputStorageBuffer.Destroy();
        for (auto &buffer : inputStorageBuffers) {
            buffer.Destroy();
        }
        for (auto &buffer : inputStagingBuffers) {
            buffer.Destroy();
        }
        return status;
    };

    for (std::size_t index = 0; index < inputs.size(); ++index) {
        VkStatus status = allocator_.AllocStagingBuffer(inputs[index].nbytes, inputStagingBuffers[index]);
        if (status != VkStatus::kSuccess) {
            return finish(status);
        }
        status = allocator_.AllocStorageBuffer(inputs[index].nbytes, inputStorageBuffers[index]);
        if (status != VkStatus::kSuccess) {
            return finish(status);
        }
        status = inputStagingBuffers[index].Upload(hostInputPtrs[index], inputs[index].nbytes);
        if (status != VkStatus::kSuccess) {
            return finish(status);
        }
    }

    VkStatus status = allocator_.AllocStorageBuffer(outputs[0].nbytes, outputStorageBuffer);
    if (status != VkStatus::kSuccess) {
        return finish(status);
    }
    status = allocator_.AllocStagingBuffer(outputs[0].nbytes, outputStagingBuffer);
    if (status != VkStatus::kSuccess) {
        return finish(status);
    }

    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = deviceContext_.ComputeQueueFamily();
    if (vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
        return finish(VkStatus::kCommandRecordFailed);
    }

    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS) {
        return finish(VkStatus::kCommandRecordFailed);
    }

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS) {
        return finish(VkStatus::kQueueSubmitFailed);
    }

    VkDescriptorPoolSize descriptorPoolSize{};
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = static_cast<std::uint32_t>(artifact.meta.bindings.size());

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 1;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
    if (vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        return finish(VkStatus::kDescriptorCreateFailed);
    }

    const VkDescriptorSetLayout descriptorSetLayout =
        reinterpret_cast<VkDescriptorSetLayout>(cachedPipeline->DescriptorSetLayoutHandle());
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;
    if (vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet) != VK_SUCCESS) {
        return finish(VkStatus::kDescriptorCreateFailed);
    }

    std::vector<VkDescriptorBufferInfo> descriptorBufferInfos;
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    descriptorBufferInfos.reserve(artifact.meta.bindings.size());
    descriptorWrites.reserve(artifact.meta.bindings.size());
    std::size_t inputIndex = 0;
    for (const auto &binding : artifact.meta.bindings) {
        const VkBufferHandle &buffer = binding.isOutput ? outputStorageBuffer : inputStorageBuffers[inputIndex++];
        descriptorBufferInfos.push_back(VkDescriptorBufferInfo{
            reinterpret_cast<VkBuffer>(buffer.NativeHandle()),
            0,
            static_cast<VkDeviceSize>(buffer.Size()),
        });
        descriptorWrites.push_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptorSet,
            binding.binding,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &descriptorBufferInfos.back(),
            nullptr,
        });
    }
    vkUpdateDescriptorSets(
        device, static_cast<std::uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

    VkCommandBufferBeginInfo commandBufferBeginInfo{};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS) {
        return finish(VkStatus::kCommandRecordFailed);
    }

    for (std::size_t index = 0; index < inputStagingBuffers.size(); ++index) {
        const VkBufferCopy copyRegion{0, 0, static_cast<VkDeviceSize>(inputs[index].nbytes)};
        vkCmdCopyBuffer(
            commandBuffer,
            reinterpret_cast<VkBuffer>(inputStagingBuffers[index].NativeHandle()),
            reinterpret_cast<VkBuffer>(inputStorageBuffers[index].NativeHandle()),
            1,
            &copyRegion);
    }

    std::vector<VkBufferMemoryBarrier> toComputeBarriers;
    toComputeBarriers.reserve(inputStorageBuffers.size() + 1);
    for (const auto &buffer : inputStorageBuffers) {
        toComputeBarriers.push_back(VkBufferMemoryBarrier{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            reinterpret_cast<VkBuffer>(buffer.NativeHandle()),
            0,
            VK_WHOLE_SIZE,
        });
    }
    toComputeBarriers.push_back(VkBufferMemoryBarrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        reinterpret_cast<VkBuffer>(outputStorageBuffer.NativeHandle()),
        0,
        VK_WHOLE_SIZE,
    });
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        static_cast<std::uint32_t>(toComputeBarriers.size()),
        toComputeBarriers.data(),
        0,
        nullptr);

    vkCmdBindPipeline(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        reinterpret_cast<VkPipeline>(cachedPipeline->NativeHandle()));
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        reinterpret_cast<VkPipelineLayout>(cachedPipeline->PipelineLayoutHandle()),
        0,
        1,
        &descriptorSet,
        0,
        nullptr);

    const std::uint32_t numel = ComputeNumel(outputs[0]);
    vkCmdPushConstants(
        commandBuffer,
        reinterpret_cast<VkPipelineLayout>(cachedPipeline->PipelineLayoutHandle()),
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(numel),
        &numel);
    vkCmdDispatch(
        commandBuffer, artifact.meta.dispatch.groupX, artifact.meta.dispatch.groupY, artifact.meta.dispatch.groupZ);

    const VkBufferMemoryBarrier outputToTransferBarrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        reinterpret_cast<VkBuffer>(outputStorageBuffer.NativeHandle()),
        0,
        VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        1,
        &outputToTransferBarrier,
        0,
        nullptr);

    const VkBufferCopy outputCopyRegion{0, 0, static_cast<VkDeviceSize>(outputs[0].nbytes)};
    vkCmdCopyBuffer(
        commandBuffer,
        reinterpret_cast<VkBuffer>(outputStorageBuffer.NativeHandle()),
        reinterpret_cast<VkBuffer>(outputStagingBuffer.NativeHandle()),
        1,
        &outputCopyRegion);

    const VkBufferMemoryBarrier outputToHostBarrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        reinterpret_cast<VkBuffer>(outputStagingBuffer.NativeHandle()),
        0,
        VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0,
        nullptr,
        1,
        &outputToHostBarrier,
        0,
        nullptr);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return finish(VkStatus::kCommandRecordFailed);
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS) {
        return finish(VkStatus::kQueueSubmitFailed);
    }
    if (vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return finish(VkStatus::kQueueSubmitFailed);
    }

    status = outputStagingBuffer.Download(hostOutputPtrs[0], outputs[0].nbytes);
    if (status != VkStatus::kSuccess) {
        return finish(status);
    }
    return finish(VkStatus::kSuccess);
#else
    if (enableCache_) {
        const std::string artifactKey = BuildArtifactKey(artifact, inputs);
        artifactCache_.Insert(artifactKey, artifact);
        shapeCache_.Insert(artifactKey, BuildShapeSignature(inputs));
    }
    return VkStatus::kNotSupported;
#endif
}

void VkRunner::Destroy() {
    allocator_.Destroy();
    deviceContext_.Destroy();
    instanceContext_.Destroy();
    available_ = false;
    validationEnabled_ = false;
}

std::string VkRunner::BuildArtifactKey(const GpuVkArtifact &artifact, const std::vector<VkTensorDesc> &inputs) const {
    std::ostringstream builder;
    builder << artifact.meta.kernelName << ":";
    for (const auto &input : inputs) {
        for (auto dim : input.shape) {
            builder << dim << ",";
        }
        builder << ";";
    }
    return builder.str();
}

std::vector<std::int64_t> VkRunner::BuildShapeSignature(const std::vector<VkTensorDesc> &inputs) const {
    std::vector<std::int64_t> signature;
    for (const auto &input : inputs) {
        signature.insert(signature.end(), input.shape.begin(), input.shape.end());
        signature.push_back(-1);
    }
    return signature;
}

} // namespace npu::tile_fwk::gpu_vk
