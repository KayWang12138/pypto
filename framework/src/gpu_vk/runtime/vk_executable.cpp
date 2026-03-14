#include "gpu_vk/runtime/vk_executable.h"

namespace npu::tile_fwk::gpu_vk {

bool VkExecutable::Ready() const {
    return !artifact.Empty() && pipeline != nullptr && pipeline->Created();
}

} // namespace npu::tile_fwk::gpu_vk
