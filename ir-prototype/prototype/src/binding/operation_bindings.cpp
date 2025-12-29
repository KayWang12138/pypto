// PTO-IR prototype: Operation bindings for Python.
// All comments must remain in English for consistency.

#include "bindings.h"
#include "ir/type.h"
#include "ir/op/op_opcode.h"
#include "ir/op/op_factory.h"
#include "ir/op/op_payload.h"
#include "ir/builder/ir_builder.h"

#include <pybind11/stl.h>
#include <sstream>

namespace pto {

// Helper function to bind Opcode enum.
static void BindOpcode(py::module_ &m) {
    py::enum_<Opcode>(m, "Opcode")
        .value("OP_INVALID", Opcode::OP_INVALID)
        .value("OP_ADD", Opcode::OP_ADD)
        .value("OP_SUB", Opcode::OP_SUB)
        .value("OP_MUL", Opcode::OP_MUL)
        .value("OP_DIV", Opcode::OP_DIV)
        .value("OP_CAST", Opcode::OP_CAST)
        .value("OP_EXP", Opcode::OP_EXP)
        .value("OP_LOG", Opcode::OP_LOG)
        .value("OP_SQRT", Opcode::OP_SQRT)
        .value("OP_RSQRT", Opcode::OP_RSQRT)
        .value("OP_ROWMAX_SINGLE", Opcode::OP_ROWMAX_SINGLE)
        .value("OP_ROWSUM_SINGLE", Opcode::OP_ROWSUM_SINGLE)
        .value("OP_RESHAPE", Opcode::OP_RESHAPE)
        .value("OP_ASSEMBLE", Opcode::OP_ASSEMBLE)
        .value("OP_VIEW", Opcode::OP_VIEW)
        .value("OP_TENSOR_CREATE", Opcode::OP_TENSOR_CREATE)
        .export_values();
}

// Helper functions to create operations using IRBuilder.
// Returns ValuePtr which can be either Tensor or Tile depending on input type.
static ValuePtr CreateBinaryOp(
    Opcode opcode,
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs,
    IRBuilder& builder) {
    ValuePtrs inputs = {lhs, rhs};
    auto outputs = builder.CreateOp(opcode, inputs);
    if (outputs.empty()) {
        throw std::runtime_error("CreateBinaryOp: no outputs returned from IRBuilder");
    }
    return outputs[0];
}

// Macro to generate binary operation functions
#define DEFINE_BINARY_OP(NAME, OPCODE) \
static ValuePtr NAME( \
    const std::shared_ptr<Value>& lhs, \
    const std::shared_ptr<Value>& rhs, \
    IRBuilder& builder) { \
    return CreateBinaryOp(OPCODE, lhs, rhs, builder); \
}

DEFINE_BINARY_OP(Add, Opcode::OP_ADD)
DEFINE_BINARY_OP(Sub, Opcode::OP_SUB)
DEFINE_BINARY_OP(Mul, Opcode::OP_MUL)
DEFINE_BINARY_OP(Div, Opcode::OP_DIV)
DEFINE_BINARY_OP(Maximum, Opcode::OP_MAXIMUM)
DEFINE_BINARY_OP(Minimum, Opcode::OP_MINIMUM)

#undef DEFINE_BINARY_OP

static ValuePtr Cast(
    const std::shared_ptr<Value>& input,
    DataType targetType,
    CastMode mode,
    IRBuilder& builder) {
    // Create CastSpec and CastPayload
    CastSpec spec{targetType, mode};
    auto payload = std::make_shared<CastPayload>(spec);
    
    // Use IRBuilder to create operation
    ValuePtrs inputs = {input};
    auto outputs = builder.CreateOp(Opcode::OP_CAST, inputs, payload);
    if (outputs.empty()) {
        throw std::runtime_error("Cast: no outputs returned from IRBuilder");
    }
    return outputs[0];
}

// View operation: Extract a sub-region from input Value.
// - Tensor -> Tensor (Python frontend mode)
// - Tile -> Tile (compiler optimization mode)
static ValuePtr View(
    const std::shared_ptr<Value>& input,
    const std::vector<size_t>& shape,
    const std::vector<Scalar>& offsets,
    IRBuilder& builder) {
    // Create ViewSpec
    ViewSpec spec;
    spec.shape = shape;
    spec.offset = offsets;
    
    // Create ViewPayload
    auto payload = std::make_shared<ViewPayload>(spec);
    
    // Use IRBuilder to create operation
    // Returns Tensor if input is Tensor, Tile if input is Tile
    ValuePtrs inputs = {input};
    auto outputs = builder.CreateOp(Opcode::OP_VIEW, inputs, payload);
    if (outputs.empty()) {
        throw std::runtime_error("View: no outputs returned from IRBuilder");
    }
    return outputs[0];
}

// Assemble operation: Update a sub-region of base with patch.
// - Tensor + Tensor -> Tensor (Python frontend mode)
// - Tensor + Tile -> Tensor (mixed mode from compiler)
// - Tile + Tile -> Tile (compiler optimization mode)
static ValuePtr Assemble(
    const std::shared_ptr<Value>& patch,
    const std::vector<Scalar>& offsets,
    const std::shared_ptr<Value>& base,
    IRBuilder& builder) {
    // Create AssembleSpec
    AssembleSpec spec;
    spec.offset = offsets;
    
    // Create AssemblePayload
    auto payload = std::make_shared<AssemblePayload>(spec);
    
    // Use IRBuilder to create operation
    // Note: Assemble has (base, patch) input order in the op
    // Returns Tensor if base is Tensor, Tile if base is Tile
    ValuePtrs inputs = {base, patch};
    auto outputs = builder.CreateOp(Opcode::OP_ASSEMBLE, inputs, payload);
    if (outputs.empty()) {
        throw std::runtime_error("Assemble: no outputs returned from IRBuilder");
    }
    return outputs[0];
}

// Helper function for reduction operations (amax, amin, sum, etc.)
static ValuePtr ReduceOp(
    const std::shared_ptr<Value>& input,
    ReduceKind kind,
    int axis,
    bool keepdim,
    IRBuilder& builder) {
    // Create ReduceSpec and ReducePayload
    ReduceSpec spec{kind, axis, keepdim};
    auto payload = std::make_shared<ReducePayload>(spec);
    
    // Determine the opcode based on reduce kind
    Opcode opcode;
    switch (kind) {
        case ReduceKind::RowMax:
            opcode = Opcode::OP_ROWMAX_SINGLE;
            break;
        case ReduceKind::RowSum:
            opcode = Opcode::OP_ROWSUM_SINGLE;
            break;
        default:
            throw std::runtime_error("ReduceOp: unsupported reduce kind");
    }
    
    // Use IRBuilder to create operation
    ValuePtrs inputs = {input};
    auto outputs = builder.CreateOp(opcode, inputs, payload);
    if (outputs.empty()) {
        throw std::runtime_error("ReduceOp: no outputs returned from IRBuilder");
    }
    return outputs[0];
}

static ValuePtr Amax(
    const std::shared_ptr<Value>& input,
    int axis,
    bool keepdim,
    IRBuilder& builder) {
    return ReduceOp(input, ReduceKind::RowMax, axis, keepdim, builder);
}

static ValuePtr Sum(
    const std::shared_ptr<Value>& input,
    int axis,
    bool keepdim,
    IRBuilder& builder) {
    return ReduceOp(input, ReduceKind::RowSum, axis, keepdim, builder);
}

// Helper function for unary element-wise operations
static ValuePtr UnaryOp(
    const std::shared_ptr<Value>& input,
    Opcode opcode,
    IRBuilder& builder) {
    // Use IRBuilder to create operation
    ValuePtrs inputs = {input};
    auto outputs = builder.CreateOp(opcode, inputs);
    if (outputs.empty()) {
        throw std::runtime_error("UnaryOp: no outputs returned from IRBuilder");
    }
    return outputs[0];
}

// Macro to generate unary operation functions
#define DEFINE_UNARY_OP(NAME, OPCODE) \
static ValuePtr NAME(const std::shared_ptr<Value>& input, IRBuilder& builder) { \
    return UnaryOp(input, OPCODE, builder); \
}

DEFINE_UNARY_OP(Exp, Opcode::OP_EXP)
DEFINE_UNARY_OP(Log, Opcode::OP_LOG)
DEFINE_UNARY_OP(Sqrt, Opcode::OP_SQRT)
DEFINE_UNARY_OP(Rsqrt, Opcode::OP_RSQRT)

#undef DEFINE_UNARY_OP

// Helper struct to hold matmul extended parameters
struct MatmulExtendParam {
    std::shared_ptr<Tensor> biasTensor;
    std::shared_ptr<Tensor> scaleTensor;
    ReLuType reluType = ReLuType::NoReLu;
    double scale = 0.0;
    
    MatmulExtendParam() = default;
};

// Matmul operation (2D matrix multiplication)
static ValuePtr Matmul(
    DataType outDtype,
    const std::shared_ptr<Value>& input,
    const std::shared_ptr<Value>& mat2,
    bool aTrans,
    bool bTrans,
    bool cMatrixNz,
    const MatmulExtendParam* extendParams,
    IRBuilder& builder) {
    
    // Create MatmulSpec
    MatmulSpec spec;
    spec.outDtype = outDtype;
    spec.aTrans = aTrans;
    spec.bTrans = bTrans;
    spec.cMatrixNz = cMatrixNz;
    
    // Handle extended parameters if provided
    if (extendParams) {
        // Check if bias tensor is valid (non-null)
        if (extendParams->biasTensor) {
            spec.hasBias = true;
            spec.biasTensor = std::static_pointer_cast<Value>(extendParams->biasTensor);
        }
        
        // Check if scale tensor is valid
        if (extendParams->scaleTensor) {
            spec.scaleTensor = std::static_pointer_cast<Value>(extendParams->scaleTensor);
        }
        
        // Set scale value
        if (extendParams->scale != 0.0) {
            spec.hasScale = true;
            spec.scale = extendParams->scale;
        }
        
        // Set ReLU type
        spec.reluType = extendParams->reluType;
    }
    
    auto payload = std::make_shared<MatmulPayload>(spec);
    
    ValuePtrs inputs = {input, mat2};
    auto outputs = builder.CreateOp(Opcode::OP_MATMUL, inputs, payload);
    if (outputs.empty()) {
        throw std::runtime_error("Matmul: no outputs returned from IRBuilder");
    }
    return outputs[0];
}

// BatchMatmul operation (batched matrix multiplication)
static ValuePtr BatchMatmul(
    DataType outDtype,
    const std::shared_ptr<Value>& input,
    const std::shared_ptr<Value>& mat2,
    bool aTrans,
    bool bTrans,
    bool cMatrixNz,
    IRBuilder& builder) {
    
    // Create MatmulSpec (no extended params for batch matmul)
    MatmulSpec spec;
    spec.outDtype = outDtype;
    spec.aTrans = aTrans;
    spec.bTrans = bTrans;
    spec.cMatrixNz = cMatrixNz;
    
    auto payload = std::make_shared<MatmulPayload>(spec);
    
    ValuePtrs inputs = {input, mat2};
    auto outputs = builder.CreateOp(Opcode::OP_BATCH_MATMUL, inputs, payload);
    if (outputs.empty()) {
        throw std::runtime_error("BatchMatmul: no outputs returned from IRBuilder");
    }
    return outputs[0];
}

// TensorCreate operation: Create a new tensor with specified shape and dtype
static ValuePtr TensorCreate(
    const std::vector<Scalar>& shape,
    DataType dtype,
    IRBuilder& builder) {
    // Create TensorCreateSpec
    TensorCreateSpec spec;
    spec.shape = shape;
    spec.dtype = dtype;
    
    // Create TensorCreatePayload
    auto payload = std::make_shared<TensorCreatePayload>(spec);
    
    // Use IRBuilder to create operation (no inputs for tensor.create)
    ValuePtrs inputs = {};
    auto outputs = builder.CreateOp(Opcode::OP_TENSOR_CREATE, inputs, payload);
    if (outputs.empty()) {
        throw std::runtime_error("TensorCreate: no outputs returned from IRBuilder");
    }
    return outputs[0];
}

// Helper function to bind binary operations to Python
// This template function generates Python bindings for binary operations
template<ValuePtr (*OpFunc)(const std::shared_ptr<Value>&, const std::shared_ptr<Value>&, IRBuilder&)>
static void BindBinaryOp(py::module_& m, const char* name, 
                         const std::function<std::shared_ptr<pto::Tensor>(pto::ValuePtr)>& valuePtrToTensor) {
    // Overload 1: Tensor + Tensor
    m.def(name, [=](const std::shared_ptr<pto::Tensor>& lhs, 
                    const std::shared_ptr<pto::Tensor>& rhs, 
                    pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error(std::string(name) + ": builder is null");
        auto output_value = OpFunc(std::static_pointer_cast<pto::Value>(lhs),
                                    std::static_pointer_cast<pto::Value>(rhs), *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("lhs"), py::arg("rhs"), py::arg("builder"));
    
    // Overload 2: Tensor + Scalar
    m.def(name, [=](const std::shared_ptr<pto::Tensor>& lhs, 
                    const std::shared_ptr<pto::Scalar>& rhs, 
                    pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error(std::string(name) + ": builder is null");
        auto output_value = OpFunc(std::static_pointer_cast<pto::Value>(lhs),
                                    std::static_pointer_cast<pto::Value>(rhs), *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("lhs"), py::arg("rhs"), py::arg("builder"));
}

// Helper function to bind unary operations to Python
// This template function generates Python bindings for unary operations
template<ValuePtr (*OpFunc)(const std::shared_ptr<Value>&, IRBuilder&)>
static void BindUnaryOp(py::module_& m, const char* name,
                        const std::function<std::shared_ptr<pto::Tensor>(pto::ValuePtr)>& valuePtrToTensor) {
    m.def(name, [=](const std::shared_ptr<pto::Tensor>& input, 
                    pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error(std::string(name) + ": builder is null");
        auto output = OpFunc(std::static_pointer_cast<pto::Value>(input), *builder);
        return valuePtrToTensor(output);
    }, py::arg("input"), py::arg("builder"));
}

// Main operation bindings function
void BindOperationBindings(py::module_ &m) {
    // Opcode enum binding.
    BindOpcode(m);

    // Operation creation functions.
    // Accept Tensor (shared_ptr) and convert to ValuePtr for IR operations
    // Return Tensor (shared_ptr) for Python
    // Helper lambda to convert ValuePtr to Tensor (shared_ptr)
    auto valuePtrToTensor = [](pto::ValuePtr output_value) -> std::shared_ptr<pto::Tensor> {
        // For now, we primarily support Tensor output
        // TODO: Handle Tile output when needed
        auto output_tensor = std::dynamic_pointer_cast<pto::Tensor>(output_value);
        if (output_tensor) {
            return output_tensor;
        }
        throw std::runtime_error("Operation output must be a Tensor (Tile support TODO)");
    };
    
    // Bind binary operations using template function
    BindBinaryOp<Add>(m, "Add", valuePtrToTensor);
    BindBinaryOp<Sub>(m, "Sub", valuePtrToTensor);
    BindBinaryOp<Mul>(m, "Mul", valuePtrToTensor);
    BindBinaryOp<Div>(m, "Div", valuePtrToTensor);
    BindBinaryOp<Maximum>(m, "Maximum", valuePtrToTensor);
    BindBinaryOp<Minimum>(m, "Minimum", valuePtrToTensor);
    
    // Cast function binding
    m.def("Cast", [valuePtrToTensor](const std::shared_ptr<pto::Tensor>& input, pto::DataType targetType, pto::IRBuilder* builder, py::object mode_obj) {
        if (!builder) throw std::runtime_error("Cast: builder is null");
        pto::CastMode mode = pto::CastMode::CAST_ROUND;  // Default mode
        if (!mode_obj.is_none()) {
            mode = mode_obj.cast<pto::CastMode>();
        }
        
        auto output_value = Cast(std::static_pointer_cast<pto::Value>(input), targetType, mode, *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("input"), py::arg("targetType"), py::arg("builder"), py::arg("mode") = py::none());
    
    // View function binding
    // View now returns Tensor when input is Tensor (for Python frontend).
    // The trait rules have been updated to support Tensor -> Tensor mode.
    m.def("View", [valuePtrToTensor](const std::shared_ptr<pto::Tensor>& input, 
                     const std::vector<long long>& shape,
                     const std::vector<pto::Scalar>& offsets,
                     pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error("View: builder is null");
        std::vector<size_t> staticShape(shape.begin(), shape.end());
        auto output_value = View(std::static_pointer_cast<pto::Value>(input), staticShape, offsets, *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("input"), py::arg("shape"), py::arg("offsets"), py::arg("builder"));
    
    // Assemble function binding
    // Assemble now supports Tensor + Tensor -> Tensor (for Python frontend).
    // Returns a new Tensor with the patch assembled at the specified offsets.
    m.def("Assemble", [valuePtrToTensor](const std::shared_ptr<pto::Tensor>& patch,
                                                 const std::vector<pto::Scalar>& offsets,
                                                 const std::shared_ptr<pto::Tensor>& base,
                                                 pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error("Assemble: builder is null");
        auto output_value = Assemble(std::static_pointer_cast<pto::Value>(patch), offsets,
                                          std::static_pointer_cast<pto::Value>(base), *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("patch"), py::arg("offsets"), py::arg("base"), py::arg("builder"));
    
    // Reduction operations
    m.def("Amax", [valuePtrToTensor](const std::shared_ptr<pto::Tensor>& input, int axis, bool keepdim, pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error("Amax: builder is null");
        auto output_value = Amax(std::static_pointer_cast<pto::Value>(input), axis, keepdim, *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("input"), py::arg("axis"), py::arg("keepdim") = false, py::arg("builder"));
    
    m.def("Sum", [valuePtrToTensor](const std::shared_ptr<pto::Tensor>& input, int axis, bool keepdim, pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error("Sum: builder is null");
        auto output_value = Sum(std::static_pointer_cast<pto::Value>(input), axis, keepdim, *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("input"), py::arg("axis"), py::arg("keepdim") = false, py::arg("builder"));
    
    // Bind unary operations using valuePtrToTensor
    BindUnaryOp<Exp>(m, "Exp", valuePtrToTensor);
    BindUnaryOp<Log>(m, "Log", valuePtrToTensor);
    BindUnaryOp<Sqrt>(m, "Sqrt", valuePtrToTensor);
    BindUnaryOp<Rsqrt>(m, "Rsqrt", valuePtrToTensor);

    // MatmulExtendParam binding
    py::class_<MatmulExtendParam>(m, "MatmulExtendParam")
        .def(py::init<>())
        .def_readwrite("bias_tensor", &MatmulExtendParam::biasTensor)
        .def_readwrite("scale_tensor", &MatmulExtendParam::scaleTensor)
        .def_readwrite("relu_type", &MatmulExtendParam::reluType)
        .def_readwrite("scale", &MatmulExtendParam::scale);

    // Matmul operation binding (with optional extended parameters)
    m.def("Matmul", 
        [valuePtrToTensor](
            pto::DataType outDtype,
            const std::shared_ptr<pto::Tensor>& input,
            const std::shared_ptr<pto::Tensor>& mat2,
            bool aTrans,
            bool bTrans,
            bool cMatrixNz,
            pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error("Matmul: builder is null");
        auto output_value = Matmul(outDtype, std::static_pointer_cast<pto::Value>(input),
                                        std::static_pointer_cast<pto::Value>(mat2), 
                                        aTrans, bTrans, cMatrixNz, nullptr, *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("out_dtype"), py::arg("input"), py::arg("mat2"), 
       py::arg("a_trans"), py::arg("b_trans"), py::arg("c_matrix_nz"), py::arg("builder"));

    // Matmul operation binding with extended parameters
    m.def("Matmul", 
        [valuePtrToTensor](
            pto::DataType outDtype,
            const std::shared_ptr<pto::Tensor>& input,
            const std::shared_ptr<pto::Tensor>& mat2,
            bool aTrans,
            bool bTrans,
            bool cMatrixNz,
            const MatmulExtendParam& extendParams,
            pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error("Matmul: builder is null");
        auto output_value = Matmul(outDtype, std::static_pointer_cast<pto::Value>(input),
                                        std::static_pointer_cast<pto::Value>(mat2), 
                                        aTrans, bTrans, cMatrixNz, &extendParams, *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("out_dtype"), py::arg("input"), py::arg("mat2"), 
       py::arg("a_trans"), py::arg("b_trans"), py::arg("c_matrix_nz"), 
       py::arg("extend_params"), py::arg("builder"));

    // BatchMatmul operation binding
    m.def("BatchMatmul", 
        [valuePtrToTensor](
            pto::DataType outDtype,
            const std::shared_ptr<pto::Tensor>& input,
            const std::shared_ptr<pto::Tensor>& mat2,
            bool aTrans,
            bool bTrans,
            bool cMatrixNz,
            pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error("BatchMatmul: builder is null");
        auto output_value = BatchMatmul(outDtype, std::static_pointer_cast<pto::Value>(input),
                                             std::static_pointer_cast<pto::Value>(mat2), 
                                             aTrans, bTrans, cMatrixNz, *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("out_dtype"), py::arg("input"), py::arg("mat2"), 
       py::arg("a_trans"), py::arg("b_trans"), py::arg("c_matrix_nz"), py::arg("builder"));

    // TensorCreate operation binding
    m.def("TensorCreate",
        [valuePtrToTensor](
            const std::vector<pto::Scalar>& shape,
            pto::DataType dtype,
            pto::IRBuilder* builder) {
        if (!builder) throw std::runtime_error("TensorCreate: builder is null");
        auto output_value = TensorCreate(shape, dtype, *builder);
        return valuePtrToTensor(output_value);
    }, py::arg("shape"), py::arg("dtype"), py::arg("builder"));
}

}  // namespace pto

