// PTO-IR prototype: type system structures.
// All comments must remain in English for consistency.

#pragma once

#include "ir/object.h"

#include <cstddef>
#include <memory>
#include <ostream>
#include <vector>

namespace pto {

// Enumeration for scalar data types.
// This enum must stay aligned with the Python side DataType enum (pypto_impl.DataType).
enum class DataType {
    BOOL,

    INT4,
    INT8,
    INT16,
    INT32,
    INT64,

    UINT8,
    UINT16,
    UINT32,
    UINT64,

    FP8,
    FP16,
    BF16,
    FP32,
    FP64,

    HF4,
    HF8,

    BOTTOM,
    UNKNOWN
};

// Base class for all types in PTO-IR.
// Type represents the structure of data, separate from specific values.
class Type {
public:
    explicit Type(DataType dataType) : dataType_(dataType) {}
    virtual ~Type() = default;

    DataType GetDataType() const { return dataType_; }

    // Get the size of the data type in bytes (e.g., FP32 -> 4 bytes).
    static size_t GetDataTypeSize(DataType dataType);

    // Get the size of the data type in bytes for this type instance.
    size_t GetDataTypeSize() const { return GetDataTypeSize(dataType_); }

    // Get the total size of this type in bytes.
    // For ScalarType: returns GetDataTypeSize()
    // For TileType/TensorType: returns GetDataTypeSize() * product of shape dimensions
    virtual size_t GetTypeSize() const = 0;

    // Pretty-print the type.
    virtual void Print(std::ostream& os) const = 0;

protected:
    DataType dataType_;
};

using TypePtr = std::shared_ptr<Type>;

// Scalar type: represents a single scalar value type (e.g., int32, fp32).
class ScalarType : public Type {
public:
    explicit ScalarType(DataType dataType) : Type(dataType) {}

    size_t GetTypeSize() const override { return GetDataTypeSize(); }
    void Print(std::ostream& os) const override;
};

// Tile type: represents a tile type with static shape and element type.
// TileType represents the type of a tile with a specific (shape, dataType) combination.
class TileType : public Type {
public:
    TileType(DataType elementType, const std::vector<size_t>& shape)
        : Type(elementType), shape_(shape) {}

    const std::vector<size_t>& GetShape() const { return shape_; }
    
    size_t GetTypeSize() const override;
    void Print(std::ostream& os) const override;

private:
    std::vector<size_t> shape_;  // Static shape dimensions
};

// Tensor type: represents a tensor type with shape and element type.
// TensorType represents the type of a tensor with a specific (shape, dataType) combination.
class TensorType : public Type {
public:
    explicit TensorType(DataType dataType) : Type(dataType) {}

    size_t GetTypeSize() const override { return GetDataTypeSize(); }
    void Print(std::ostream& os) const override;
};

} // namespace pto

