#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_vk/common/vk_status.h"

namespace npu::tile_fwk::gpu_vk {

struct SpirvCompileOptions {
    bool optimize{true};
    bool debugInfo{false};
    std::string validatorPath;
    bool dumpArtifacts{false};
    std::string dumpDir;
};

struct SpirvCompileResult {
    std::vector<std::uint32_t> spirv;
    std::string glslPath;
    std::string spirvPath;
    std::string compileLog;
    std::string sourceHash;
    bool isRealSpirv{false};
};

class SpirvCompiler {
public:
    VkStatus CompileGlslToSpirv(
        const std::string &glsl, const SpirvCompileOptions &options, SpirvCompileResult &result) const;
};

} // namespace npu::tile_fwk::gpu_vk
