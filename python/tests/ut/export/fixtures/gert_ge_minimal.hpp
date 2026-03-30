// Minimal gert/ge stubs for compiling pypto export codegen tests only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

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

class InferShapeContext {
public:
    void SetInput(int idx, std::initializer_list<int64_t> dims) {
        const auto u = static_cast<size_t>(idx);
        if (inputs_.size() <= u) {
            inputs_.resize(u + 1);
        }
        inputs_[u] = Shape(dims);
    }

    const Shape* GetInputShape(int i) const { return &inputs_[static_cast<size_t>(i)]; }

    Shape* GetOutputShape(int /*i*/) { return &output_; }

    const Shape& OutputShape() const { return output_; }

private:
    std::vector<Shape> inputs_;
    Shape output_;
};

class InferDataTypeContext {
public:
    int GetInputDataType(int /*i*/) const { return 0; }

    void SetOutputDataType(int /*idx*/, int /*dtype*/) {}
};

}  // namespace gert

namespace ge {

using graphStatus = int;
inline constexpr graphStatus GRAPH_SUCCESS = 0;
inline constexpr graphStatus GRAPH_FAILED = 1;

}  // namespace ge
