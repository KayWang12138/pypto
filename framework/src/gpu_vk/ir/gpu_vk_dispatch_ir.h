#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npu::tile_fwk::gpu_vk {

struct GpuVkDispatchParam {
    std::uint32_t groupX{1};
    std::uint32_t groupY{1};
    std::uint32_t groupZ{1};
    std::uint32_t localX{1};
    std::uint32_t localY{1};
    std::uint32_t localZ{1};
};

struct GpuVkBufferBinding {
    std::string name;
    std::uint32_t binding{0};
    bool isOutput{false};
};

class GpuVkDispatchGraph {
public:
    void SetKernelName(const std::string &kernelName);
    const std::string &KernelName() const;

    void AddBinding(const GpuVkBufferBinding &binding);
    const std::vector<GpuVkBufferBinding> &Bindings() const;

    void AddLoweredOp(const std::string &expr);
    const std::vector<std::string> &LoweredOps() const;

    void SetDispatch(const GpuVkDispatchParam &dispatch);
    const GpuVkDispatchParam &Dispatch() const;

private:
    std::string kernelName_;
    std::vector<GpuVkBufferBinding> bindings_;
    std::vector<std::string> loweredOps_;
    GpuVkDispatchParam dispatch_;
};

} // namespace npu::tile_fwk::gpu_vk
