#pragma once

#include <cassert>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>
#include "ir/value.h"
#include "ir/utils.h"

namespace pto {
enum class PayloadKind {
    None = 0,
    // Reduction
    Reduce,

    // View-like
    Reshape,
    Assemble,
    View,

    // Type conversion
    Cast,

    // Matrix operations
    Matmul,

    // Tensor creation
    TensorCreate,
};


class OpPayload {
public:
    explicit OpPayload(PayloadKind k) : kind_(k) {}
    virtual ~OpPayload() = default;

    PayloadKind Kind() const { return kind_; }

    // Payload-level verification (optional, can be no-op).
    virtual bool Verify(std::string* err) const = 0;

    // Payload-level printing (optional).
    virtual void Print(std::ostream& os) const = 0;

private:
    PayloadKind kind_;
};


enum class ReduceKind {
    RowSum,
    RowMax,
};

struct ReduceSpec {
    ReduceKind kind;
    int axis;
    bool keepDim;
};

class ReducePayload : public OpPayload {
public:
    static constexpr PayloadKind kKind = PayloadKind::Reduce;

    explicit ReducePayload(ReduceSpec spec)
        : OpPayload(kKind), spec_(std::move(spec)) {}

    const ReduceSpec& Spec() const { return spec_; }

    bool Verify(std::string* err) const override {
        if (spec_.axis < 0) {
            if (err) *err = "Reduce axis must be >= 0";
            return false;
        }
        return true;
    }

    void Print(std::ostream& os) const override {
        os << "{kind="
           << (spec_.kind == ReduceKind::RowSum ? "rowsum" : "rowmax")
           << ", axis=" << spec_.axis
           << ", keepDim=" << spec_.keepDim << "}";
    }

private:
    ReduceSpec spec_;
};

struct ReshapeSpec {
    std::vector<int64_t> shape;        // target shape, -1 allowed
    std::vector<Scalar> validShape;    // symbolic dims (optional)
    bool inplace = false;
};

class ReshapePayload final : public OpPayload {
public:
    static constexpr PayloadKind kKind = PayloadKind::Reshape;

    explicit ReshapePayload(ReshapeSpec spec)
        : OpPayload(kKind), spec_(std::move(spec)) {}

    const ReshapeSpec& Spec() const { return spec_; }

    bool Verify(std::string* err) const override {
        int unknown = 0;
        for (auto d : spec_.shape) {
            if (d == -1) {
                unknown++;
            } else if (d <= 0) {
                if (err) *err = "ReshapePayload: shape dim must be >0 or -1";
                return false;
            }
        }
        if (unknown > 1) {
            if (err) *err = "ReshapePayload: at most one -1 dim allowed";
            return false;
        }
        return true;
    }

    void Print(std::ostream& os) const override {
        os << "{shape=[";
        for (size_t i = 0; i < spec_.shape.size(); ++i) {
            os << spec_.shape[i];
            if (i + 1 < spec_.shape.size()) os << ", ";
        }
        os << "]";

        if (!spec_.validShape.empty()) {
            os << ", validShape=[";
            for (size_t i = 0; i < spec_.validShape.size(); ++i) {
                spec_.validShape[i].Print(os, /*indent=*/0);
                if (i + 1 < spec_.validShape.size()) os << ", ";
            }
            os << "]";
        }

        os << ", inplace=" << (spec_.inplace ? "true" : "false");
        os << "}";
        }

private:
    ReshapeSpec spec_;
};


struct AssembleSpec {
    std::vector<Scalar> offset;   // per-dimension offset
};

class AssemblePayload final : public OpPayload {
public:
    static constexpr PayloadKind kKind = PayloadKind::Assemble;

    explicit AssemblePayload(AssembleSpec spec)
        : OpPayload(kKind), spec_(std::move(spec)) {}

    const AssembleSpec& Spec() const { return spec_; }

    bool Verify(std::string* err) const override {
        if (spec_.offset.empty()) {
            if (err) *err = "AssemblePayload: offset must not be empty";
            return false;
        }
        return true;
    }

    void Print(std::ostream& os) const override {
        os << "{offset=[";
        for (size_t i = 0; i < spec_.offset.size(); ++i) {
            spec_.offset[i].Print(os, /*indent=*/0);
            if (i + 1 < spec_.offset.size()) os << ", ";
        }
        os << "]}";
    }


private:
    AssembleSpec spec_;
};


struct ViewSpec {
    std::vector<size_t> shape;   // static view shape
    std::vector<Scalar>  offset;     // symbolic offset
};

class ViewPayload final : public OpPayload {
public:
    static constexpr PayloadKind kKind = PayloadKind::View;

    explicit ViewPayload(ViewSpec spec)
        : OpPayload(kKind), spec_(std::move(spec)) {}

    const ViewSpec& Spec() const { return spec_; }

    bool Verify(std::string* err) const override {
        for (auto d : spec_.shape) {
            if (d <= 0) {
                if (err) *err = "ViewPayload: shape dim must be > 0";
                return false;
            }
        }
        // offset is symbolic → no numeric check
        return true;
    }

    void Print(std::ostream& os) const override {
        os << "{shape=[";
        for (size_t i = 0; i < spec_.shape.size(); ++i) {
            os << spec_.shape[i];
            if (i + 1 < spec_.shape.size()) os << ", ";
        }
        os << "], offset=[";

        for (size_t i = 0; i < spec_.offset.size(); ++i) {
            spec_.offset[i].Print(os, /*indent=*/0);
            if (i + 1 < spec_.offset.size()) os << ", ";
        }
        os << "]}";
    }

private:
    ViewSpec spec_;
};

struct CastSpec {
    DataType targetType;  // Target data type
    CastMode mode;        // Cast mode (round, floor, etc.)
};

class CastPayload final : public OpPayload {
public:
    static constexpr PayloadKind kKind = PayloadKind::Cast;

    explicit CastPayload(CastSpec spec)
        : OpPayload(kKind), spec_(std::move(spec)) {}

    const CastSpec& Spec() const { return spec_; }

    bool Verify(std::string* err) const override {
        if (spec_.targetType == DataType::UNKNOWN || spec_.targetType == DataType::BOTTOM) {
            if (err) *err = "CastPayload: target type must be valid";
            return false;
        }
        return true;
    }

    void Print(std::ostream& os) const override {
        os << "{targetType=";
        // Print target type name (simplified)
        os << static_cast<int>(spec_.targetType);
        os << ", mode=";
        switch (spec_.mode) {
            case CastMode::CAST_NONE: os << "none"; break;
            case CastMode::CAST_RINT: os << "rint"; break;
            case CastMode::CAST_ROUND: os << "round"; break;
            case CastMode::CAST_FLOOR: os << "floor"; break;
            case CastMode::CAST_CEIL: os << "ceil"; break;
            case CastMode::CAST_TRUNC: os << "trunc"; break;
            case CastMode::CAST_ODD: os << "odd"; break;
            default: os << "unknown";
        }
        os << "}";
    }

private:
    CastSpec spec_;
};

// ReLU type for matmul extended parameters
enum class ReLuType {
    NoReLu = 0,
    ReLu = 1,
};

struct MatmulSpec {
    DataType outDtype;        // Output data type
    bool aTrans = false;      // Transpose left matrix
    bool bTrans = false;      // Transpose right matrix
    bool cMatrixNz = false;   // Output in NZ format
    
    // Extended parameters (optional)
    bool hasBias = false;
    ValuePtr biasTensor;      // Bias tensor (optional)
    
    bool hasScale = false;
    double scale = 0.0;       // Scale value (optional)
    ValuePtr scaleTensor;     // Scale tensor (optional)
    
    ReLuType reluType = ReLuType::NoReLu;  // ReLU type (optional)
};

class MatmulPayload final : public OpPayload {
public:
    static constexpr PayloadKind kKind = PayloadKind::Matmul;

    explicit MatmulPayload(MatmulSpec spec)
        : OpPayload(kKind), spec_(std::move(spec)) {}

    const MatmulSpec& Spec() const { return spec_; }

    bool Verify(std::string* err) const override {
        if (spec_.outDtype == DataType::UNKNOWN || spec_.outDtype == DataType::BOTTOM) {
            if (err) *err = "MatmulPayload: output dtype must be valid";
            return false;
        }
        return true;
    }

    void Print(std::ostream& os) const override {
        os << "{outDtype=" << static_cast<int>(spec_.outDtype)
           << ", aTrans=" << (spec_.aTrans ? "true" : "false")
           << ", bTrans=" << (spec_.bTrans ? "true" : "false")
           << ", cMatrixNz=" << (spec_.cMatrixNz ? "true" : "false");
        
        if (spec_.hasBias) {
            os << ", hasBias=true";
        }
        if (spec_.hasScale) {
            os << ", scale=" << spec_.scale;
        }
        if (spec_.reluType != ReLuType::NoReLu) {
            os << ", reluType=" << static_cast<int>(spec_.reluType);
        }
        os << "}";
    }

private:
    MatmulSpec spec_;
};

struct TensorCreateSpec {
    std::vector<Scalar> shape;  // Tensor shape
    DataType dtype;             // Tensor data type
};

class TensorCreatePayload final : public OpPayload {
public:
    static constexpr PayloadKind kKind = PayloadKind::TensorCreate;

    explicit TensorCreatePayload(TensorCreateSpec spec)
        : OpPayload(kKind), spec_(std::move(spec)) {}

    const TensorCreateSpec& Spec() const { return spec_; }

    bool Verify(std::string* err) const override {
        if (spec_.dtype == DataType::UNKNOWN || spec_.dtype == DataType::BOTTOM) {
            if (err) *err = "TensorCreatePayload: dtype must be valid";
            return false;
        }
        if (spec_.shape.empty()) {
            if (err) *err = "TensorCreatePayload: shape must not be empty";
            return false;
        }
        return true;
    }

    void Print(std::ostream& os) const override {
        os << "{shape=[";
        for (size_t i = 0; i < spec_.shape.size(); ++i) {
            spec_.shape[i].Print(os, /*indent=*/0);
            if (i + 1 < spec_.shape.size()) os << ", ";
        }
        os << "], dtype=" << DataTypeToString(spec_.dtype) << "}";
    }

private:
    TensorCreateSpec spec_;
};

}