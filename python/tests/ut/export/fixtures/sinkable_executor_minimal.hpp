// Minimal stubs for compiling pypto-generated sinkable executors / PrepareExecute only.
#pragma once

#include "gert_ge_minimal.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

namespace gert {

struct EagerOpExecutionContext;
class MockSinkableOpExecutionContext;
using SinkableOpExecutionContext = MockSinkableOpExecutionContext;

enum class SinkableOpIo : uint32_t {
    kInput = 0,
    kOutput,
};

// Matches real ``graph/custom_op.h``: ``PrepareExecute`` is the pure
// virtual the codegen overrides (with the ``override`` keyword).
class SinkableExecuteOp {
public:
    virtual ~SinkableExecuteOp() = default;
    virtual ge::graphStatus PrepareExecute(SinkableOpExecutionContext *ctx) = 0;
};

/// Mock with the same method names + signatures as the real
/// ``gert::SinkableOpExecutionContext`` (MallocWorkSpace with capital S,
/// SpecifyIoOffset, HostArgsToDevice returning ``void *``). Records the
/// arguments the generated ``PrepareExecute`` passes so tests can assert
/// on the args-array layout and offset specs.
class MockSinkableOpExecutionContext {
public:
    MockSinkableOpExecutionContext(size_t input_num, size_t output_num)
        : input_num_(input_num), output_num_(output_num) {
        inputs_.resize(input_num_);
        outputs_.resize(output_num_);
        for (size_t i = 0; i < input_num_; ++i) {
            inputs_[i].addr_ = reinterpret_cast<void *>(static_cast<std::uintptr_t>(0x1000 + i * 0x100));
        }
        for (size_t i = 0; i < output_num_; ++i) {
            outputs_[i].addr_ = reinterpret_cast<void *>(static_cast<std::uintptr_t>(0x2000 + i * 0x100));
        }
    }

    size_t GetComputeNodeInputNum() const { return input_num_; }

    size_t GetComputeNodeOutputNum() const { return output_num_; }

    Tensor *GetOutputTensor(size_t i) { return &outputs_[i]; }

    Tensor *GetInputTensor(size_t i) { return &inputs_[i]; }

    void *MallocWorkSpace(size_t /*size*/) {
        workspace_ = reinterpret_cast<void *>(static_cast<std::uintptr_t>(0x7000));
        return workspace_;
    }

    void *HostArgsToDevice(const void *args, size_t args_size) {
        last_args_.resize(args_size / sizeof(int64_t));
        if (args_size > 0 && args != nullptr) {
            std::memcpy(last_args_.data(), args, args_size);
        }
        return last_args_.data();
    }

    ge::graphStatus SpecifyIoOffset(SinkableOpIo kind, size_t offsets[], size_t count) {
        if (kind == SinkableOpIo::kInput) {
            input_offsets_.clear();
            if (offsets != nullptr && count > 0) {
                input_offsets_.assign(offsets, offsets + count);
            }
        } else if (kind == SinkableOpIo::kOutput) {
            last_output_spec_count_ = count;
            (void)offsets;
        }
        return ge::GRAPH_SUCCESS;
    }

    const std::vector<int64_t> &last_args() const { return last_args_; }

    const std::vector<size_t> &input_offsets() const { return input_offsets_; }

    size_t last_output_spec_count() const { return last_output_spec_count_; }

    void *workspace_ptr() const { return workspace_; }

private:
    size_t input_num_{};
    size_t output_num_{};
    std::vector<Tensor> inputs_;
    std::vector<Tensor> outputs_;
    void *workspace_{};
    std::vector<int64_t> last_args_;
    std::vector<size_t> input_offsets_;
    size_t last_output_spec_count_{};
};

// ``SinkableOpExecutionContext`` is aliased at the top of the namespace
// (before ``SinkableExecuteOp``) so the base class's pure-virtual
// ``PrepareExecute`` signature refers to the same type the generated
// subclass uses.

}  // namespace gert

#define REG_AUTO_MAPPING_OP(name) /* test stub */
