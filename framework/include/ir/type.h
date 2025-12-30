// PTO-IR prototype: type system structures.
// All comments must remain in English for consistency.

#pragma once

#include "ir/utils.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <variant>
#include <cstdint>
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

// Enumeration for data type categories (Scalar/Tensor/Tile).
enum class ValueKind {
    Scalar,
    Tile,
    Tensor
};

// Enumeration for scalar value kinds: constant, symbolic expression.
enum class ScalarValueKind {
    Constant,        // Constant value known at compile time
    Symbolic,        // Symbolic expression
};

// Enumeration for cast modes.
enum class CastMode {
    CAST_NONE,       // No rounding mode specified
    CAST_RINT,       // Round to nearest integer (ties to even)
    CAST_ROUND,      // Round to nearest integer
    CAST_FLOOR,      // Round down to nearest integer
    CAST_CEIL,       // Round up to nearest integer
    CAST_TRUNC,      // Truncate towards zero
    CAST_ODD,        // Round to nearest odd integer
};

// Base class for all data types in PTO-IR.
class Value : public Object, public AttributeHolder {
public:
    explicit Value(ValueKind kind, DataType dataType, std::string name="") 
        : Object(ObjectType::Value, name), valueKind_(kind), dataType_(dataType) {}
    virtual ~Value() = default;

    const std::string GetSSAName() const {
        if (name_.empty()) {
            return "%" + std::to_string(id_);
        }
        // If tensor has a name, return it directly without adding numeric suffix
        return GetPrefixedName() + "_" + std::to_string(id_);
    }
    ObjectType GetObjectType() const override { return ObjectType::Value; }

    ValueKind GetValueKind() const { return valueKind_; }
    DataType GetDataType() const { return dataType_; }

    // Pretty-print the type with the given indentation.
    virtual void Print(std::ostream& os, int indent = 0) const = 0;

protected:
    ValueKind valueKind_;
    DataType dataType_;
};

using ValuePtr = std::shared_ptr<Value>;
using ValuePtrs = std::vector<ValuePtr>;

class StaticValue : public Value {
public:
    StaticValue(ValueKind kind, DataType dataType, std::string name="")
        : Value(kind, dataType, name) {}
    virtual ~StaticValue() = default;
    
};

using ConstantType = std::variant<bool, int, int64_t, size_t, double>;

// Scalar type: bool, int4, int8, int16, int32, int64, fp8, fp16, bf16, fp32, fp64
class Scalar : public StaticValue {
public:
    explicit Scalar(DataType type, std::string name="", ScalarValueKind valueKind = ScalarValueKind::Symbolic) 
        : StaticValue(ValueKind::Scalar, type, name), valueKind_(valueKind), constantValue_(int64_t{0}) {}
    
    explicit Scalar(std::string typeName, std::string name="", ScalarValueKind valueKind = ScalarValueKind::Symbolic) 
        : StaticValue(ValueKind::Scalar, StringToValueType(typeName), name), valueKind_(valueKind), constantValue_(int64_t{0}) {}

    explicit Scalar(DataType type, std::string name, ScalarValueKind valueKind, std::string expr, ConstantType constantVal)
        : StaticValue(ValueKind::Scalar, type, name), symbolicExpr_(expr), valueKind_(valueKind), constantValue_(constantVal) {}

    // Constant value constructors - DataType is inferred from value type
    explicit Scalar(bool value, std::string name="")
        : StaticValue(ValueKind::Scalar, DataType::BOOL, name), valueKind_(ScalarValueKind::Constant), constantValue_(value) {}
    
    explicit Scalar(int value, std::string name="")
        : StaticValue(ValueKind::Scalar, DataType::INT32, name), valueKind_(ScalarValueKind::Constant), constantValue_(value) {}

    explicit Scalar(int64_t value, std::string name="")
        : StaticValue(ValueKind::Scalar, DataType::INT64, name), valueKind_(ScalarValueKind::Constant), constantValue_(value) {}
    
    explicit Scalar(double value, std::string name="")
        : StaticValue(ValueKind::Scalar, DataType::FP64, name), valueKind_(ScalarValueKind::Constant), constantValue_(value) {}
    
    explicit Scalar(size_t value, std::string name="")
        : StaticValue(ValueKind::Scalar, DataType::UINT64, name), valueKind_(ScalarValueKind::Constant), constantValue_(value) {}

    Scalar() : StaticValue(ValueKind::Scalar, DataType::UNKNOWN) {}

    const std::string& GetSymbolicExpr() const;
    ScalarValueKind GetScalarValueKind() const { return valueKind_; }
    
    // Get constant value (only valid when valueKind_ == Constant)
    ConstantType GetConstantValue() const {
        return constantValue_;
    }
    
    bool HasConstantValue() const { return valueKind_ == ScalarValueKind::Constant; }

    // Get constant value as int64_t. Only valid when HasConstantValue() is true.
    int64_t GetInt64Value() const;

    void Print(std::ostream& os, int indent = 0) const override;

private:
    mutable std::string symbolicExpr_;  // Cached string representation
    ScalarValueKind valueKind_;       // Kind of scalar value: constant, symbolic
    ConstantType constantValue_;  // Constant value storage
};


enum class MemSpaceKind {
    UNKNOWN,
    DDR,
    L2,
    UB,
    L1,
    L0A,
    L0B,
    L0C,
    REG,
    SHMEM,
};

class Memory : public Object {
public:
    Memory(size_t byteSize) : Object(ObjectType::Memory), byteSize_(byteSize),
         space_(MemSpaceKind::UNKNOWN) {}
    
    size_t GetSize() const { return byteSize_; }
    MemSpaceKind GetSpace() const { return space_; }
    size_t GetAddr() const { return addr_; }
    
    void SetSize(const size_t newSize) { byteSize_ = newSize; }
    void SetSpace(const MemSpaceKind kind) { space_ = kind; }
    void SetAddr(const size_t newAddr) { addr_ = newAddr; }

private:
    size_t byteSize_;
    MemSpaceKind space_;
    size_t addr_;
};

// Tile: tile<validshape, tile_shapes, strides, start_offset, elem_type, memory>
class Tile : public StaticValue {
public:
    Tile(std::string name, std::vector<Scalar> validShapes, std::vector<size_t> shape, 
            std::vector<size_t> strides, Scalar startOffset, DataType elementType,
            std::shared_ptr<Memory> mem=nullptr)
        : StaticValue(ValueKind::Tile, elementType, name),
          validShapes_(validShapes),
          shape_(shape),
          strides_(strides),
          startOffset_(startOffset),
          mem_(mem) {}

    Tile(std::vector<size_t> shape, DataType elementType,
         std::vector<Scalar> validShapes, std::string name="")
        : StaticValue(ValueKind::Tile, elementType, name),
          validShapes_(validShapes), shape_(shape) { }

    Tile(std::vector<size_t> shape, DataType elementType,
         std::string name="")
        : StaticValue(ValueKind::Tile, elementType, name),
          shape_(shape){
        validShapes_.reserve(shape.size());
        for (size_t i = 0; i < shape.size(); i++) {
            validShapes_.emplace_back(Scalar(shape[i]));
        }
    }

    const std::vector<Scalar>& GetValidShape() const { return validShapes_; }
    const std::vector<size_t>& GetShape() const { return shape_; }
    const std::vector<size_t>& GetStrides() const { return strides_; }
    Scalar GetStartOffset() const { return *startOffset_; }
    const std::shared_ptr<Memory> GetMemory() const { return mem_; }

    void SetShape(const std::vector<size_t>& newShape) { shape_ = newShape; }
    void SetStrides(const std::vector<size_t>& newStrides) { strides_ = newStrides; }
    void SetStartOffset(const Scalar newStartOffset) { startOffset_ = newStartOffset; }
    void SetMemory(const std::shared_ptr<Memory> newMem) { mem_ = newMem; }

    void Print(std::ostream& os, int indent = 0) const override;

private:
    std::vector<Scalar> validShapes_; 
    std::vector<size_t> shape_;     // Must be fully static
    std::vector<size_t> strides_;
    std::optional<Scalar> startOffset_;
    std::shared_ptr<Memory> mem_;
};

// Enumeration for tile operation formats.
enum class TileOpFormat {
    TILEOP_ND = 0,  // Dense format
    TILEOP_NZ = 1   // Non-zero (sparse) format
};

class Tensor : public Value {
public:
    // Construct tensor from a vector of Scalar dimensions.
    Tensor(const std::vector<Scalar>& shape, DataType type, std::string name="",
           TileOpFormat format = TileOpFormat::TILEOP_ND) : 
        Value(ValueKind::Tensor, type, name), shape_(shape), format_(format) {}

    // Convenience constructor for static integer shapes.
    // This is mainly used by Python bindings where shapes are passed as ints.
    // The parameter order is aligned with Python Tensor(dtype, shape, name, format).
    Tensor(DataType type, const std::vector<size_t>& shape, std::string name="",
           TileOpFormat format = TileOpFormat::TILEOP_ND) :
        Value(ValueKind::Tensor, type, name), format_(format) {

        shape_.reserve(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            shape_.emplace_back(shape[i]);
        }
    }

    const std::vector<Scalar>& GetShape() const { return shape_; }
    
    TileOpFormat GetFormat() const { return format_; }
    void SetFormat(TileOpFormat format) { format_ = format; }

    void Print(std::ostream& os, int indent) const override;
private:
    std::vector<Scalar> shape_;
    TileOpFormat format_;
};

} // namespace pto

