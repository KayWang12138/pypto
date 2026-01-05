/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vjp_math.cpp
 * \brief VJP rules for math operations.
 *
 * Corresponds to python/pypto/autograd/rules/math.py
 */

#include "../vjp_registry.h"

#include "interface/inner/element.h"

namespace npu::tile_fwk {

namespace {

// Helper: Get output gradient
LogicalTensorPtr GetOutGrad(VJPContext& ctx) {
    return ctx.GetOutputGrad("output");
}

// Helper: Get saved input tensor
LogicalTensorPtr GetSavedInput(VJPContext& ctx, const std::string& name = "input") {
    return ctx.GetSaved(name);
}

// Helper: Create a new tensor with given shape and dtype
LogicalTensorPtr CreateTensor(VJPContext& ctx, const Shape& shape, DataType dtype) {
    if (ctx.function == nullptr) {
        return nullptr;
    }
    return std::make_shared<LogicalTensor>(*ctx.function, dtype, shape);
}

// Helper: Create neg operation
LogicalTensorPtr CreateNegOp(VJPContext& ctx, LogicalTensorPtr input) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, input->GetShape(), input->Datatype());
    Operation& op = ctx.function->AddRawOperation(Opcode::OP_MULS, {input}, {output});
    Element scalarElement(input->Datatype(), -1.0);
    op.SetAttribute(OpAttributeKey::scalar, scalarElement);
    return output;
}

// Helper: Check if alpha is effectively 1.0
bool IsAlphaOne(double alpha) {
    constexpr double kEpsilon = 1e-9;
    return std::abs(alpha - 1.0) < kEpsilon;
}

// Helper: Create mul operation
LogicalTensorPtr CreateMulOp(VJPContext& ctx, LogicalTensorPtr a, LogicalTensorPtr b) {
    if (a == nullptr || b == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, a->GetShape(), a->Datatype());
    ctx.function->AddRawOperation(Opcode::OP_MUL, {a, b}, {output});
    return output;
}

// Helper: Create div operation
LogicalTensorPtr CreateDivOp(VJPContext& ctx, LogicalTensorPtr a, LogicalTensorPtr b) {
    if (a == nullptr || b == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, a->GetShape(), a->Datatype());
    ctx.function->AddRawOperation(Opcode::OP_DIV, {a, b}, {output});
    return output;
}

// Helper: Create scalar mul operation
LogicalTensorPtr CreateMulScalarOp(VJPContext& ctx, LogicalTensorPtr input, double scalar) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, input->GetShape(), input->Datatype());
    Operation& op = ctx.function->AddRawOperation(Opcode::OP_MULS, {input}, {output});
    // Use OpAttributeKey::scalar with Element type, matching binary.h:171
    Element scalarElement(input->Datatype(), scalar);
    op.SetAttribute(OpAttributeKey::scalar, scalarElement);
    return output;
}

// Helper: Create exp operation
LogicalTensorPtr CreateExpOp(VJPContext& ctx, LogicalTensorPtr input) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, input->GetShape(), input->Datatype());
    ctx.function->AddRawOperation(Opcode::OP_EXP, {input}, {output});
    return output;
}

// Helper: Create sqrt operation
LogicalTensorPtr CreateSqrtOp(VJPContext& ctx, LogicalTensorPtr input) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, input->GetShape(), input->Datatype());
    ctx.function->AddRawOperation(Opcode::OP_SQRT, {input}, {output});
    return output;
}

// Helper: Create rsqrt operation
LogicalTensorPtr CreateRsqrtOp(VJPContext& ctx, LogicalTensorPtr input) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    auto output = CreateTensor(ctx, input->GetShape(), input->Datatype());
    ctx.function->AddRawOperation(Opcode::OP_RSQRT, {input}, {output});
    return output;
}

// Helper: Create cast operation
LogicalTensorPtr CreateCastOp(VJPContext& ctx, LogicalTensorPtr input, DataType targetDtype) {
    if (input == nullptr || ctx.function == nullptr) {
        return nullptr;
    }
    if (input->Datatype() == targetDtype) {
        return input;
    }
    auto output = CreateTensor(ctx, input->GetShape(), targetDtype);
    Operation& castOp = ctx.function->AddRawOperation(Opcode::OP_CAST, {input}, {output});
    castOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
    return output;
}

// Helper: Check if dtype is differentiable
bool IsDifferentiable(DataType dtype) {
    return dtype == DataType::DT_FP32 ||
           dtype == DataType::DT_FP16 ||
           dtype == DataType::DT_BF16;
}

/**
 * VJP for Add: d_input = d_out, d_other = alpha * d_out (unbroadcast)
 */
VJPResult vjp_add(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    double alpha = ctx.GetAttr<double>("alpha", 1.0);

    LogicalTensorPtr dInput = dOut;
    LogicalTensorPtr dOther = dOut;

    if (!IsAlphaOne(alpha)) {
        dOther = CreateMulScalarOp(ctx, dOther, alpha);
    }

    // Apply unbroadcast based on input shapes
    auto inputShape = ctx.GetInputShape("input");
    auto otherShape = ctx.GetInputShape("other");

    if (!inputShape.empty()) {
        dInput = ctx.Unbroadcast(dInput, inputShape);
    }
    if (!otherShape.empty()) {
        dOther = ctx.Unbroadcast(dOther, otherShape);
    }

    return {{"input", dInput}, {"other", dOther}};
}

/**
 * VJP for Sub: d_input = d_out, d_other = -alpha * d_out (unbroadcast)
 */
VJPResult vjp_sub(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    double alpha = ctx.GetAttr<double>("alpha", 1.0);

    LogicalTensorPtr dInput = dOut;
    LogicalTensorPtr dOther = CreateNegOp(ctx, dOut);

    if (!IsAlphaOne(alpha)) {
        dOther = CreateMulScalarOp(ctx, dOther, alpha);
    }

    // Apply unbroadcast based on input shapes
    auto inputShape = ctx.GetInputShape("input");
    auto otherShape = ctx.GetInputShape("other");

    if (!inputShape.empty()) {
        dInput = ctx.Unbroadcast(dInput, inputShape);
    }
    if (!otherShape.empty()) {
        dOther = ctx.Unbroadcast(dOther, otherShape);
    }

    return {{"input", dInput}, {"other", dOther}};
}

/**
 * VJP for Mul: d_input = d_out * other, d_other = d_out * input
 */
VJPResult vjp_mul(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto inputTensor = GetSavedInput(ctx, "input");
    auto otherTensor = GetSavedInput(ctx, "other");

    LogicalTensorPtr dInput = nullptr;
    LogicalTensorPtr dOther = nullptr;

    if (otherTensor != nullptr) {
        dInput = CreateMulOp(ctx, dOut, otherTensor);
    }
    if (inputTensor != nullptr) {
        dOther = CreateMulOp(ctx, dOut, inputTensor);
    }

    // Apply unbroadcast based on input shapes
    auto inputShape = ctx.GetInputShape("input");
    auto otherShape = ctx.GetInputShape("other");

    if (!inputShape.empty() && dInput != nullptr) {
        dInput = ctx.Unbroadcast(dInput, inputShape);
    }
    if (!otherShape.empty() && dOther != nullptr) {
        dOther = ctx.Unbroadcast(dOther, otherShape);
    }

    return {{"input", dInput}, {"other", dOther}};
}

/**
 * VJP for Div: d_input = d_out / other, d_other = -d_out * input / other^2
 */
VJPResult vjp_div(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto inputTensor = GetSavedInput(ctx, "input");
    auto otherTensor = GetSavedInput(ctx, "other");

    LogicalTensorPtr dInput = nullptr;
    LogicalTensorPtr dOther = nullptr;

    if (otherTensor != nullptr) {
        dInput = CreateDivOp(ctx, dOut, otherTensor);
    }

    if (inputTensor != nullptr && otherTensor != nullptr) {
        // d_other = -d_out * input / other^2
        auto negDOut = CreateNegOp(ctx, dOut);
        auto numerator = CreateMulOp(ctx, negDOut, inputTensor);
        auto denominator = CreateMulOp(ctx, otherTensor, otherTensor);
        dOther = CreateDivOp(ctx, numerator, denominator);
    }

    // Apply unbroadcast based on input shapes
    auto inputShape = ctx.GetInputShape("input");
    auto otherShape = ctx.GetInputShape("other");

    if (!inputShape.empty() && dInput != nullptr) {
        dInput = ctx.Unbroadcast(dInput, inputShape);
    }
    if (!otherShape.empty() && dOther != nullptr) {
        dOther = ctx.Unbroadcast(dOther, otherShape);
    }

    return {{"input", dInput}, {"other", dOther}};
}

/**
 * VJP for Neg: d_input = -d_out
 */
VJPResult vjp_neg(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto dInput = CreateNegOp(ctx, dOut);
    return {{"input", dInput}};
}

/**
 * VJP for Exp: d_input = d_out * exp(input) = d_out * output
 */
VJPResult vjp_exp(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    // Prefer using saved output to avoid recomputation
    auto output = GetSavedInput(ctx, "output");
    LogicalTensorPtr dInput = nullptr;

    if (output != nullptr) {
        dInput = CreateMulOp(ctx, dOut, output);
    } else {
        auto inputTensor = GetSavedInput(ctx, "input");
        if (inputTensor != nullptr) {
            auto expInput = CreateExpOp(ctx, inputTensor);
            dInput = CreateMulOp(ctx, dOut, expInput);
        }
    }

    return {{"input", dInput}};
}

/**
 * VJP for Log (LN): d_input = d_out / input
 */
VJPResult vjp_log(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputTensor = GetSavedInput(ctx, "input");
    LogicalTensorPtr dInput = nullptr;

    if (inputTensor != nullptr) {
        dInput = CreateDivOp(ctx, dOut, inputTensor);
    }

    return {{"input", dInput}};
}

/**
 * VJP for Sqrt: d_input = d_out / (2 * sqrt(input)) = d_out / (2 * output)
 */
VJPResult vjp_sqrt(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto output = GetSavedInput(ctx, "output");
    LogicalTensorPtr dInput = nullptr;

    if (output != nullptr) {
        auto twoOut = CreateMulScalarOp(ctx, output, 2.0);
        dInput = CreateDivOp(ctx, dOut, twoOut);
    } else {
        auto inputTensor = GetSavedInput(ctx, "input");
        if (inputTensor != nullptr) {
            auto sqrtInput = CreateSqrtOp(ctx, inputTensor);
            auto twoSqrt = CreateMulScalarOp(ctx, sqrtInput, 2.0);
            dInput = CreateDivOp(ctx, dOut, twoSqrt);
        }
    }

    return {{"input", dInput}};
}

/**
 * VJP for Rsqrt: d_input = -0.5 * d_out * rsqrt(input)^3
 */
VJPResult vjp_rsqrt(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto output = GetSavedInput(ctx, "output");
    LogicalTensorPtr dInput = nullptr;

    if (output != nullptr) {
        // rsqrt^3 = rsqrt * rsqrt * rsqrt
        auto outSquared = CreateMulOp(ctx, output, output);
        auto outCubed = CreateMulOp(ctx, outSquared, output);
        auto scaled = CreateMulScalarOp(ctx, outCubed, -0.5);
        dInput = CreateMulOp(ctx, dOut, scaled);
    } else {
        auto inputTensor = GetSavedInput(ctx, "input");
        if (inputTensor != nullptr) {
            auto rsqrtInput = CreateRsqrtOp(ctx, inputTensor);
            auto outSquared = CreateMulOp(ctx, rsqrtInput, rsqrtInput);
            auto outCubed = CreateMulOp(ctx, outSquared, rsqrtInput);
            auto scaled = CreateMulScalarOp(ctx, outCubed, -0.5);
            dInput = CreateMulOp(ctx, dOut, scaled);
        }
    }

    return {{"input", dInput}};
}

/**
 * VJP for Cast: d_input = cast(d_out, input_dtype)
 */
VJPResult vjp_cast(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputTensor = GetSavedInput(ctx, "input");
    if (inputTensor == nullptr) {
        return {{"input", nullptr}};
    }

    DataType inputDtype = inputTensor->Datatype();

    // Cannot compute gradient for non-differentiable dtypes
    if (!IsDifferentiable(inputDtype)) {
        return {{"input", nullptr}};
    }

    auto dInput = CreateCastOp(ctx, dOut, inputDtype);
    return {{"input", dInput}};
}

/**
 * VJP for Abs: d_input = d_out * sign(input)
 *
 * sign(input) = input / abs(input) = input / output
 * Note: sign(0) is undefined, we use input / output which gives NaN for 0.
 */
VJPResult vjp_abs(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputTensor = GetSavedInput(ctx, "input");
    auto outputTensor = GetSavedInput(ctx, "output");

    if (inputTensor == nullptr || outputTensor == nullptr) {
        // Cannot compute gradient without saved tensors
        return {{"input", dOut}};
    }

    // sign(x) = x / |x| = x / output
    // d_input = d_out * sign(input)
    auto signTensor = CreateDivOp(ctx, inputTensor, outputTensor);
    if (signTensor == nullptr) {
        return {{"input", dOut}};
    }

    auto dInput = CreateMulOp(ctx, dOut, signTensor);
    return {{"input", dInput}};
}

/**
 * VJP for AddS (scalar add): d_input = d_out
 *
 * The scalar doesn't have a gradient.
 */
VJPResult vjp_adds(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    return {{"input", dOut}};
}

/**
 * VJP for MulS (scalar mul): d_input = d_out * scalar
 */
VJPResult vjp_muls(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    double scalar = ctx.GetScalarAttr(1.0);
    auto dInput = CreateMulScalarOp(ctx, dOut, scalar);
    return {{"input", dInput}};
}

/**
 * VJP for DivS (scalar div): d_input = d_out / scalar
 */
VJPResult vjp_divs(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    double scalar = ctx.GetScalarAttr(1.0);
    bool reverseOperand = ctx.GetAttr<bool>("reverseOperand", false);

    if (reverseOperand) {
        // output = scalar / input => d_input = -scalar * d_out / input^2
        auto inputTensor = GetSavedInput(ctx, "input");
        if (inputTensor == nullptr) {
            return {{"input", nullptr}};
        }
        auto inputSquared = CreateMulOp(ctx, inputTensor, inputTensor);
        auto numerator = CreateMulScalarOp(ctx, dOut, -scalar);
        auto dInput = CreateDivOp(ctx, numerator, inputSquared);
        return {{"input", dInput}};
    } else {
        // output = input / scalar => d_input = d_out / scalar
        auto dInput = CreateMulScalarOp(ctx, dOut, 1.0 / scalar);
        return {{"input", dInput}};
    }
}

/**
 * VJP for Reciprocal: output = 1/input
 *
 * d_input = -d_out / input^2 = -d_out * output^2
 */
VJPResult vjp_reciprocal(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto output = GetSavedInput(ctx, "output");
    LogicalTensorPtr dInput = nullptr;

    if (output != nullptr) {
        // d_input = -d_out * output^2
        auto outSquared = CreateMulOp(ctx, output, output);
        auto negDOut = CreateNegOp(ctx, dOut);
        dInput = CreateMulOp(ctx, negDOut, outSquared);
    } else {
        auto inputTensor = GetSavedInput(ctx, "input");
        if (inputTensor != nullptr) {
            // d_input = -d_out / input^2
            auto inputSquared = CreateMulOp(ctx, inputTensor, inputTensor);
            auto negDOut = CreateNegOp(ctx, dOut);
            dInput = CreateDivOp(ctx, negDOut, inputSquared);
        }
    }

    return {{"input", dInput}};
}

/**
 * VJP for Pow: output = input^exponent
 *
 * d_input = d_out * exponent * input^(exponent-1)
 *         = d_out * exponent * output / input
 */
VJPResult vjp_pow(VJPContext& ctx) {
    auto dOut = GetOutGrad(ctx);
    if (dOut == nullptr) {
        return {{"input", nullptr}};
    }

    auto inputTensor = GetSavedInput(ctx, "input");
    auto output = GetSavedInput(ctx, "output");
    double exponent = ctx.GetScalarAttr(1.0);

    if (inputTensor == nullptr) {
        return {{"input", nullptr}};
    }

    LogicalTensorPtr dInput = nullptr;

    if (output != nullptr) {
        // d_input = exponent * d_out * output / input
        auto scaled = CreateMulScalarOp(ctx, dOut, exponent);
        auto term = CreateMulOp(ctx, scaled, output);
        dInput = CreateDivOp(ctx, term, inputTensor);
    } else {
        // Fallback: compute input^(exponent-1)
        // For simplicity, use the relationship: d/dx(x^n) = n * x^(n-1) = n * x^n / x
        // But without saved output, we'd need to recompute. Skip for now.
        return {{"input", nullptr}};
    }

    return {{"input", dInput}};
}

} // anonymous namespace

// Register all math VJP rules
REG_VJP(OP_ADD, vjp_add);
REG_VJP(OP_SUB, vjp_sub);
REG_VJP(OP_MUL, vjp_mul);
REG_VJP(OP_DIV, vjp_div);
REG_VJP(OP_NEG, vjp_neg);
REG_VJP(OP_EXP, vjp_exp);
REG_VJP(OP_LN, vjp_log);
REG_VJP(OP_SQRT, vjp_sqrt);
REG_VJP(OP_RSQRT, vjp_rsqrt);
REG_VJP(OP_CAST, vjp_cast);
REG_VJP(OP_ABS, vjp_abs);

// Register broadcast variants
REG_VJP(OP_ADD_BRC, vjp_add);
REG_VJP(OP_SUB_BRC, vjp_sub);
REG_VJP(OP_MUL_BRC, vjp_mul);
REG_VJP(OP_DIV_BRC, vjp_div);

// Register scalar operation variants
REG_VJP(OP_ADDS, vjp_adds);
REG_VJP(OP_MULS, vjp_muls);
REG_VJP(OP_DIVS, vjp_divs);

// Register additional math operations
REG_VJP(OP_RECIPROCAL, vjp_reciprocal);
REG_VJP(OP_POW, vjp_pow);

} // namespace npu::tile_fwk
