#include "gpu_vk/ir/gpu_vk_tensor_ir.h"

#include <algorithm>

namespace npu::tile_fwk::gpu_vk {

namespace {

std::vector<std::int64_t> MakeDefaultStride(const std::vector<std::int64_t> &shape) {
    std::vector<std::int64_t> stride(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 1; --i) {
        stride[i - 2] = stride[i - 1] * std::max<std::int64_t>(shape[i - 1], 1);
    }
    return stride;
}

GpuVkValue NormalizeValue(const GpuVkValue &value, bool isInput, bool isOutput) {
    GpuVkValue normalized = value;
    normalized.isInput = isInput;
    normalized.isOutput = isOutput;
    if (normalized.stride.empty() && !normalized.shape.empty()) {
        normalized.stride = MakeDefaultStride(normalized.shape);
    }
    return normalized;
}

} // namespace

const char *GpuVkOpKindToString(GpuVkOpKind opKind) {
    switch (opKind) {
        case GpuVkOpKind::INPUT:
            return "input";
        case GpuVkOpKind::OUTPUT:
            return "output";
        case GpuVkOpKind::ADD:
            return "add";
        case GpuVkOpKind::MUL:
            return "mul";
        case GpuVkOpKind::RELU:
            return "relu";
        case GpuVkOpKind::WHERE:
            return "where";
        case GpuVkOpKind::SUM:
            return "sum";
        case GpuVkOpKind::MATMUL:
            return "matmul";
        case GpuVkOpKind::UNKNOWN:
        default:
            return "unknown";
    }
}

GpuVkOpKind ParseGpuVkOpKind(const std::string &opName) {
    if (opName == "add") {
        return GpuVkOpKind::ADD;
    }
    if (opName == "mul") {
        return GpuVkOpKind::MUL;
    }
    if (opName == "relu") {
        return GpuVkOpKind::RELU;
    }
    if (opName == "where") {
        return GpuVkOpKind::WHERE;
    }
    if (opName == "sum") {
        return GpuVkOpKind::SUM;
    }
    if (opName == "matmul") {
        return GpuVkOpKind::MATMUL;
    }
    return GpuVkOpKind::UNKNOWN;
}

void GpuVkTensorGraph::SetName(const std::string &name) {
    name_ = name;
}

const std::string &GpuVkTensorGraph::Name() const {
    return name_;
}

std::size_t GpuVkTensorGraph::AddInput(const GpuVkValue &value) {
    values_.push_back(NormalizeValue(value, true, false));
    inputNames_.push_back(values_.back().name);
    return values_.size() - 1;
}

std::size_t GpuVkTensorGraph::AddIntermediate(const GpuVkValue &value) {
    values_.push_back(NormalizeValue(value, false, false));
    return values_.size() - 1;
}

std::size_t GpuVkTensorGraph::AddOutput(const GpuVkValue &value) {
    values_.push_back(NormalizeValue(value, false, true));
    outputNames_.push_back(values_.back().name);
    return values_.size() - 1;
}

std::size_t GpuVkTensorGraph::AddNode(
    GpuVkOpKind op, const std::vector<std::string> &inputs, const std::vector<std::string> &outputs) {
    nodes_.push_back(GpuVkNode{op, inputs, outputs});
    return nodes_.size() - 1;
}

bool GpuVkTensorGraph::Empty() const {
    return values_.empty() && nodes_.empty();
}

std::size_t GpuVkTensorGraph::ValueCount() const {
    return values_.size();
}

std::size_t GpuVkTensorGraph::NodeCount() const {
    return nodes_.size();
}

const std::vector<GpuVkValue> &GpuVkTensorGraph::Values() const {
    return values_;
}

std::vector<GpuVkValue> &GpuVkTensorGraph::MutableValues() {
    return values_;
}

const std::vector<GpuVkNode> &GpuVkTensorGraph::Nodes() const {
    return nodes_;
}

std::vector<GpuVkNode> &GpuVkTensorGraph::MutableNodes() {
    return nodes_;
}

const std::vector<std::string> &GpuVkTensorGraph::InputNames() const {
    return inputNames_;
}

const std::vector<std::string> &GpuVkTensorGraph::OutputNames() const {
    return outputNames_;
}

const GpuVkValue *GpuVkTensorGraph::FindValue(const std::string &name) const {
    auto it = std::find_if(values_.begin(), values_.end(), [&name](const GpuVkValue &value) { return value.name == name; });
    return it == values_.end() ? nullptr : &(*it);
}

GpuVkValue *GpuVkTensorGraph::FindValue(const std::string &name) {
    auto it = std::find_if(values_.begin(), values_.end(), [&name](const GpuVkValue &value) { return value.name == name; });
    return it == values_.end() ? nullptr : &(*it);
}

} // namespace npu::tile_fwk::gpu_vk
