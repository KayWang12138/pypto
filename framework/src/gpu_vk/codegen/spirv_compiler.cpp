#include "gpu_vk/codegen/spirv_compiler.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <unistd.h>

namespace npu::tile_fwk::gpu_vk {

namespace {

constexpr std::string_view kEnvValidatorPath = "PYPTO_VK_GLSLANG_VALIDATOR";
constexpr std::string_view kDefaultTempDir = "/tmp/pypto_vk";
constexpr std::string_view kDefaultDumpDir = "./build_out/gpu_vk_dump";

std::string ComputeSourceHash(const std::string &glsl) {
    std::uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : glsl) {
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= 1099511628211ull;
    }
    std::ostringstream builder;
    builder << std::hex << std::setfill('0') << std::setw(16) << hash;
    return builder.str();
}

bool IsExecutableFile(const std::string &path) {
    return !path.empty() && access(path.c_str(), X_OK) == 0;
}

std::string ShellQuote(const std::string &value) {
    std::string quoted;
    quoted.reserve(value.size() + 2);
    quoted.push_back('\'');
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

std::string ResolveValidatorPath(const SpirvCompileOptions &options) {
    if (IsExecutableFile(options.validatorPath)) {
        return options.validatorPath;
    }

    if (const char *envValidator = std::getenv(kEnvValidatorPath.data()); envValidator != nullptr &&
        IsExecutableFile(envValidator)) {
        return envValidator;
    }

    if (const char *pathEnv = std::getenv("PATH"); pathEnv != nullptr) {
        std::stringstream pathStream(pathEnv);
        std::string directory;
        while (std::getline(pathStream, directory, ':')) {
            if (directory.empty()) {
                continue;
            }
            std::filesystem::path candidate = std::filesystem::path(directory) / "glslangValidator";
            if (IsExecutableFile(candidate.string())) {
                return candidate.string();
            }
        }
    }

    return "";
}

std::string ReadTextFile(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream.good()) {
        return "";
    }
    std::ostringstream builder;
    builder << stream.rdbuf();
    return builder.str();
}

VkStatus ReadSpirvFile(const std::filesystem::path &path, std::vector<std::uint32_t> &spirvOut) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.good()) {
        spirvOut.clear();
        return VkStatus::kShaderCompileFailed;
    }

    const std::streamsize size = stream.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(std::uint32_t)) != 0) {
        spirvOut.clear();
        return VkStatus::kShaderCompileFailed;
    }

    stream.seekg(0, std::ios::beg);
    spirvOut.resize(static_cast<std::size_t>(size) / sizeof(std::uint32_t));
    if (!stream.read(reinterpret_cast<char *>(spirvOut.data()), size)) {
        spirvOut.clear();
        return VkStatus::kShaderCompileFailed;
    }
    return VkStatus::kSuccess;
}

} // namespace

VkStatus SpirvCompiler::CompileGlslToSpirv(
    const std::string &glsl, const SpirvCompileOptions &options, SpirvCompileResult &result) const {
    result = SpirvCompileResult{};
    if (glsl.empty()) {
        return VkStatus::kInvalidArgument;
    }

    const std::string validatorPath = ResolveValidatorPath(options);
    if (validatorPath.empty()) {
        result.compileLog = "glslangValidator was not found. Set PYPTO_VK_GLSLANG_VALIDATOR or provide validatorPath.";
        return VkStatus::kToolNotFound;
    }

    result.sourceHash = ComputeSourceHash(glsl);

    std::error_code ec;
    const std::filesystem::path outputDir =
        options.dumpArtifacts ? std::filesystem::path(options.dumpDir.empty() ? std::string(kDefaultDumpDir) : options.dumpDir)
                              : std::filesystem::path(kDefaultTempDir);
    std::filesystem::create_directories(outputDir, ec);
    if (ec) {
        result.compileLog = "failed to create output directory: " + outputDir.string() + ", error=" + ec.message();
        return VkStatus::kInternalError;
    }

    result.glslPath = (outputDir / (result.sourceHash + ".comp")).string();
    result.spirvPath = (outputDir / (result.sourceHash + ".spv")).string();
    const std::filesystem::path logPath = outputDir / (result.sourceHash + ".log");

    {
        std::ofstream stream(result.glslPath, std::ios::binary | std::ios::trunc);
        if (!stream.good()) {
            result.compileLog = "failed to write GLSL source to " + result.glslPath;
            return VkStatus::kInternalError;
        }
        stream.write(glsl.data(), static_cast<std::streamsize>(glsl.size()));
        if (!stream.good()) {
            result.compileLog = "failed to flush GLSL source to " + result.glslPath;
            return VkStatus::kInternalError;
        }
    }

    std::ostringstream command;
    command << ShellQuote(validatorPath) << " -S comp -V --target-env vulkan1.2";
    if (options.debugInfo) {
        command << " -g";
    }
    command << " -o " << ShellQuote(result.spirvPath) << " " << ShellQuote(result.glslPath) << " > "
            << ShellQuote(logPath.string()) << " 2>&1";

    const int compileStatus = std::system(command.str().c_str());
    result.compileLog = ReadTextFile(logPath);
    if (compileStatus != 0) {
        if (result.compileLog.empty()) {
            result.compileLog = "glslangValidator exited with status " + std::to_string(compileStatus);
        }
        return VkStatus::kShaderCompileFailed;
    }

    const VkStatus readStatus = ReadSpirvFile(result.spirvPath, result.spirv);
    if (readStatus != VkStatus::kSuccess) {
        if (result.compileLog.empty()) {
            result.compileLog = "glslangValidator did not produce a valid SPIR-V binary.";
        }
        return readStatus;
    }

    result.isRealSpirv = true;
    return VkStatus::kSuccess;
}

} // namespace npu::tile_fwk::gpu_vk
