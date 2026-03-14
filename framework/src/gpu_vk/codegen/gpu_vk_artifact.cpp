#include "gpu_vk/codegen/gpu_vk_artifact.h"

namespace npu::tile_fwk::gpu_vk {

bool GpuVkArtifact::Empty() const {
    return glsl.empty() && spirv.empty();
}

} // namespace npu::tile_fwk::gpu_vk
