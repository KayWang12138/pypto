#include "gpu_vk/runtime/vk_runner.h"

#include <sstream>

namespace npu::tile_fwk::gpu_vk {

VkStatus VkRunner::Initialize(bool enableValidation) {
    validationEnabled_ = enableValidation;
    allocator_.Initialize();
    available_ = false;
    return VkStatus::UNAVAILABLE;
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
    if (artifact.Empty() || inputs.empty()) {
        return VkStatus::INVALID_ARGUMENT;
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
        pipeline->Create(artifact.meta.kernelName, artifact.spirv, bindings);
        pipelineCache_.Insert(pipelineKey, std::move(pipeline));
    }

    descriptorCache_.GetOrCreate({});
    if (enableCache_) {
        const std::string artifactKey = BuildArtifactKey(artifact, inputs);
        artifactCache_.Insert(artifactKey, artifact);
        shapeCache_.Insert(artifactKey, pipelineKey.shapeSignature);
    }
    return VkStatus::UNAVAILABLE;
}

void VkRunner::Destroy() {
    allocator_.Destroy();
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
