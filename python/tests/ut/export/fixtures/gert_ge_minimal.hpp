// Minimal gert/ge stubs for compiling pypto export codegen tests only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

namespace ge {

/// Matches dtype tokens emitted by export codegen (`ge::DT_FLOAT`, etc.).
enum DataType {
    DT_FLOAT = 0,           // float type
    DT_FLOAT16 = 1,         // fp16 type
    DT_INT8 = 2,            // int8 type
    DT_INT16 = 6,           // int16 type
    DT_UINT16 = 7,          // uint16 type
    DT_UINT8 = 4,           // uint8 type
    DT_INT32 = 3,           // int32 type
    DT_INT64 = 9,           // int64 type
    DT_UINT32 = 8,          // unsigned int32
    DT_UINT64 = 10,         // unsigned int64
    DT_BOOL = 12,           // bool type
    DT_DOUBLE = 11,         // double type
    DT_STRING = 13,         // string type
    DT_DUAL_SUB_INT8 = 14,  // dual output int8 type
    DT_DUAL_SUB_UINT8 = 15, // dual output uint8 type
    DT_COMPLEX64 = 16,      // complex64 type
    DT_COMPLEX128 = 17,     // complex128 type
    DT_QINT8 = 18,          // qint8 type
    DT_QINT16 = 19,         // qint16 type
    DT_QINT32 = 20,         // qint32 type
    DT_QUINT8 = 21,         // quint8 type
    DT_QUINT16 = 22,        // quint16 type
    DT_RESOURCE = 23,       // resource type
    DT_STRING_REF = 24,     // string ref type
    DT_DUAL = 25,           // dual output type
    DT_BF16 = 27,           // bf16 type
    DT_UNDEFINED = 28,      // Used to indicate a DataType field has not been set.
    DT_INT4 = 29,           // int4 type
    DT_UINT1 = 30,          // uint1 type
    DT_INT2 = 31,           // int2 type
    DT_UINT2 = 32,          // uint2 type
    DT_COMPLEX32 = 33,      // complex32 type
    DT_HIFLOAT8 = 34,       // hifloat8 type
};

// Matches real Ascend ``graph/ge_error_codes.h``: ``using graphStatus = uint32_t;``.
using graphStatus = uint32_t;
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
