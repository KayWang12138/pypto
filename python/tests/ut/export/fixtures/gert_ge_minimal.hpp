// Minimal gert/ge stubs for compiling pypto export codegen tests only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

namespace ge {

/// Matches dtype tokens emitted by export codegen (`ge::DT_FLOAT`, etc.).
enum DataType : int {
    DT_FLOAT = 0,
    DT_FLOAT16 = 1,
    DT_BF16 = 2,
    DT_INT8 = 3,
    DT_INT16 = 4,
    DT_INT32 = 5,
    DT_INT64 = 6,
    DT_UINT8 = 7,
    DT_UINT16 = 8,
    DT_UINT32 = 9,
    DT_UINT64 = 10,
    DT_BOOL = 11,
};

using graphStatus = int;
inline constexpr graphStatus GRAPH_SUCCESS = 0;
inline constexpr graphStatus GRAPH_FAILED = 1;

}  // namespace ge

namespace gert {

struct Shape {
    std::vector<int64_t> dims_;

    Shape() = default;

    explicit Shape(std::initializer_list<int64_t> il) : dims_(il) {}

    void SetDimNum(const size_t dim_num) { dims_.resize(dim_num); }

    size_t GetDimNum() const { return dims_.size(); }

    int64_t &operator[](size_t i) { return dims_[i]; }

    const int64_t &operator[](const size_t i) const { return dims_[i]; }
};

/// Storage shape view; ``GetShape()`` returns the logical ``Shape`` (dims).
class StorageShape {
public:
    StorageShape() : shape_({2, 2}) {}

    explicit StorageShape(Shape shape) : shape_(std::move(shape)) {}

    const Shape &GetShape() const { return shape_; }

private:
    Shape shape_;
};

class MockSinkableOpExecutionContext;

/// Minimal tensor stub for sinkable executor compile tests.
class Tensor {
public:
    Tensor() = default;

    const void *GetAddr() const { return addr_; }

    void *GetAddr() { return addr_; }

    ge::DataType GetDataType() const { return dtype_; }

    const StorageShape &GetShape() const { return storage_shape_; }

private:
    friend class MockSinkableOpExecutionContext;

    void *addr_{};
    ge::DataType dtype_{ge::DT_FLOAT16};
    StorageShape storage_shape_{};
};

class InferShapeContext {
public:
    void SetInput(int idx, std::initializer_list<int64_t> dims) {
        const auto u = static_cast<size_t>(idx);
        if (inputs_.size() <= u) {
            inputs_.resize(u + 1);
        }
        inputs_[u] = Shape(dims);
    }

    const Shape *GetInputShape(int i) const { return &inputs_[static_cast<size_t>(i)]; }

    Shape *GetOutputShape(int i) {
        const auto u = static_cast<size_t>(i);
        if (outputs_.size() <= u) {
            outputs_.resize(u + 1);
        }
        return &outputs_[u];
    }

    const Shape &OutputShape(size_t i) const { return outputs_.at(i); }

private:
    std::vector<Shape> inputs_;
    std::vector<Shape> outputs_;
};

class InferDataTypeContext {
public:
    InferDataTypeContext() { input_dtypes_.assign(8, ge::DT_FLOAT16); }

    void SetInputDataType(int i, ge::DataType dt) {
        const auto u = static_cast<size_t>(i);
        if (input_dtypes_.size() <= u) {
            input_dtypes_.resize(u + 1, ge::DT_FLOAT16);
        }
        input_dtypes_[u] = dt;
    }

    ge::DataType GetInputDataType(int i) const {
        return input_dtypes_.at(static_cast<size_t>(i));
    }

    void SetOutputDataType(int idx, ge::DataType dtype) {
        const auto u = static_cast<size_t>(idx);
        if (output_dtypes_.size() <= u) {
            output_dtypes_.resize(u + 1);
        }
        output_dtypes_[u] = dtype;
    }

    ge::DataType OutputDataType(size_t i) const { return output_dtypes_.at(i); }

private:
    std::vector<ge::DataType> input_dtypes_;
    std::vector<ge::DataType> output_dtypes_;
};

}  // namespace gert
