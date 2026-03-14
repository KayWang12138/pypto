#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace npu::tile_fwk::gpu_vk {

enum class GpuVkOpKind {
    UNKNOWN = 0,
    INPUT = 1,
    OUTPUT = 2,
    ADD = 3,
    MUL = 4,
    RELU = 5,
    WHERE = 6,
    SUM = 7,
    MATMUL = 8,
};

const char *GpuVkOpKindToString(GpuVkOpKind opKind);
GpuVkOpKind ParseGpuVkOpKind(const std::string &opName);

struct GpuVkValue {
    std::string name;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> stride;
    std::int32_t dtype{0};
    bool isInput{false};
    bool isOutput{false};
};

struct GpuVkNode {
    GpuVkOpKind op{GpuVkOpKind::UNKNOWN};
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

class GpuVkTensorGraph {
public:
    void SetName(const std::string &name);
    const std::string &Name() const;

    std::size_t AddInput(const GpuVkValue &value);
    std::size_t AddIntermediate(const GpuVkValue &value);
    std::size_t AddOutput(const GpuVkValue &value);
    std::size_t AddNode(GpuVkOpKind op, const std::vector<std::string> &inputs, const std::vector<std::string> &outputs);

    bool Empty() const;
    std::size_t ValueCount() const;
    std::size_t NodeCount() const;

    const std::vector<GpuVkValue> &Values() const;
    std::vector<GpuVkValue> &MutableValues();
    const std::vector<GpuVkNode> &Nodes() const;
    std::vector<GpuVkNode> &MutableNodes();

    const std::vector<std::string> &InputNames() const;
    const std::vector<std::string> &OutputNames() const;

    const GpuVkValue *FindValue(const std::string &name) const;
    GpuVkValue *FindValue(const std::string &name);

private:
    std::string name_;
    std::vector<GpuVkValue> values_;
    std::vector<GpuVkNode> nodes_;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
};

} // namespace npu::tile_fwk::gpu_vk
