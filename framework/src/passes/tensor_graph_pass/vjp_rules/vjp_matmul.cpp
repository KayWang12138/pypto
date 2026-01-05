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
 * \file vjp_matmul.cpp
 * \brief VJP rules for matrix multiplication operations.
 *
 * Corresponds to python/pypto/autograd/rules/matmul.py
 */

#include "../vjp_registry.h"

namespace npu::tile_fwk {

namespace {

// Helper: Create tensor
LogicalTensorPtr CreateTensor(VJPContext& ctx, const Shape& shape, DataType dtype) {
    if (ctx.function == nullptr) {
        return nullptr;
    }
    return std::make_shared<LogicalTensor>(*ctx.function, dtype, shape);
}

// Helper: Create matmul operation (A @ B)
// Note: For gradient computation, output dtype should match the gradient's dtype (dOut's dtype)
// which should be the same as the original operation's output dtype.
LogicalTensorPtr CreateMatmulOp(VJPContext& ctx, LogicalTensorPtr a, LogicalTensorPtr b,
                                 bool transA, bool transB) {
    if (a == nullptr || b == nullptr || ctx.function == nullptr) {
        return nullptr;
    }

    const Shape& aShape = a->GetShape();
    const Shape& bShape = b->GetShape();

    // Compute output shape for matmul
    Shape outputShape;

    // Handle batch dimensions
    size_t aNdim = aShape.size();
    size_t bNdim = bShape.size();

    if (aNdim >= 2 && bNdim >= 2) {
        // Get M, N dimensions considering transpose (K dimensions used for validation only)
        int64_t aM = transA ? aShape[aNdim - 1] : aShape[aNdim - 2];
        int64_t bN = transB ? bShape[bNdim - 2] : bShape[bNdim - 1];

        // Broadcast batch dimensions
        size_t maxBatch = std::max(aNdim, bNdim) - 2;
        for (size_t i = 0; i < maxBatch; ++i) {
            int64_t aDim = (i < aNdim - 2) ? aShape[aNdim - 3 - i] : 1;
            int64_t bDim = (i < bNdim - 2) ? bShape[bNdim - 3 - i] : 1;
            outputShape.insert(outputShape.begin(), std::max(aDim, bDim));
        }

        outputShape.push_back(aM);
        outputShape.push_back(bN);
    } else {
        // Fallback for 1D tensors
        outputShape = aShape;
    }

    // For matmul gradients, use the dOut's dtype (which is a or b here as operand)
    // FP16/BF16 matmul usually outputs FP32, gradients should be in FP32
    DataType dtype;
    if (a->Datatype() == DataType::DT_FP32 || b->Datatype() == DataType::DT_FP32) {
        dtype = DataType::DT_FP32;
    } else {
        dtype = a->Datatype();
    }

    auto output = CreateTensor(ctx, outputShape, dtype);

    // Select appropriate matmul opcode based on transpose flags
    Opcode opcode;
    if (!transA && !transB) {
        opcode = Opcode::OP_A_MUL_B;
    } else if (!transA && transB) {
        opcode = Opcode::OP_A_MUL_BT;
    } else if (transA && !transB) {
        opcode = Opcode::OP_AT_MUL_B;
    } else {
        opcode = Opcode::OP_AT_MUL_BT;
    }

    ctx.function->AddRawOperation(opcode, {a, b}, {output});
    return output;
}

/**
 * VJP for Matmul: C = A @ B
 *
 * d_A = d_C @ B^T
 * d_B = A^T @ d_C
 */
VJPResult vjp_matmul(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto A = ctx.GetSaved("input");
    auto B = ctx.GetSaved("other");

    LogicalTensorPtr dA = nullptr;
    LogicalTensorPtr dB = nullptr;

    // d_A = d_C @ B^T
    if (B != nullptr) {
        dA = CreateMatmulOp(ctx, dOut, B, false, true);
    }

    // d_B = A^T @ d_C
    if (A != nullptr) {
        dB = CreateMatmulOp(ctx, A, dOut, true, false);
    }

    return {{"input", dA}, {"other", dB}};
}

/**
 * VJP for A @ B (no transpose)
 */
VJPResult vjp_a_mul_b(VJPContext& ctx) {
    return vjp_matmul(ctx);
}

/**
 * VJP for A @ B^T
 *
 * C = A @ B^T means we compute A @ transpose(B)
 * d_A = d_C @ B (since B was transposed, we use B directly)
 * d_B = d_C^T @ A (then transpose to match B's shape)
 */
VJPResult vjp_a_mul_bt(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto A = ctx.GetSaved("input");
    auto B = ctx.GetSaved("other");

    LogicalTensorPtr dA = nullptr;
    LogicalTensorPtr dB = nullptr;

    // d_A = d_C @ B (B is already in correct form)
    if (B != nullptr) {
        dA = CreateMatmulOp(ctx, dOut, B, false, false);
    }

    // d_B = (d_C^T @ A)^T = A^T @ d_C -> need to transpose result
    // Actually: d_B = d_C^T @ A
    if (A != nullptr) {
        dB = CreateMatmulOp(ctx, dOut, A, true, false);
    }

    return {{"input", dA}, {"other", dB}};
}

/**
 * VJP for A^T @ B
 *
 * C = A^T @ B means we compute transpose(A) @ B
 * d_A = B @ d_C^T (then transpose to match A's shape)
 * d_B = A @ d_C
 */
VJPResult vjp_at_mul_b(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto A = ctx.GetSaved("input");
    auto B = ctx.GetSaved("other");

    LogicalTensorPtr dA = nullptr;
    LogicalTensorPtr dB = nullptr;

    // d_A = B @ d_C^T
    if (B != nullptr) {
        dA = CreateMatmulOp(ctx, B, dOut, false, true);
    }

    // d_B = A @ d_C
    if (A != nullptr) {
        dB = CreateMatmulOp(ctx, A, dOut, false, false);
    }

    return {{"input", dA}, {"other", dB}};
}

/**
 * VJP for A^T @ B^T
 *
 * C = A^T @ B^T
 * d_A = B^T @ d_C^T
 * d_B = d_C^T @ A^T
 */
VJPResult vjp_at_mul_bt(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto A = ctx.GetSaved("input");
    auto B = ctx.GetSaved("other");

    LogicalTensorPtr dA = nullptr;
    LogicalTensorPtr dB = nullptr;

    // d_A = B^T @ d_C^T = (d_C @ B)^T
    if (B != nullptr) {
        dA = CreateMatmulOp(ctx, B, dOut, true, true);
    }

    // d_B = d_C^T @ A^T = (A @ d_C)^T
    if (A != nullptr) {
        dB = CreateMatmulOp(ctx, dOut, A, true, true);
    }

    return {{"input", dA}, {"other", dB}};
}

/**
 * VJP for Matmul with accumulation: C += A @ B
 */
VJPResult vjp_a_mulacc_b(VJPContext& ctx) {
    // Same as regular matmul for gradient computation
    return vjp_a_mul_b(ctx);
}

/**
 * VJP for Matmul with accumulation and transpose B: C += A @ B^T
 */
VJPResult vjp_a_mulacc_bt(VJPContext& ctx) {
    return vjp_a_mul_bt(ctx);
}

} // anonymous namespace

// Register matmul VJP rules
REG_VJP(OP_A_MUL_B, vjp_a_mul_b);
REG_VJP(OP_A_MUL_BT, vjp_a_mul_bt);
REG_VJP(OP_AT_MUL_B, vjp_at_mul_b);
REG_VJP(OP_AT_MUL_BT, vjp_at_mul_bt);
REG_VJP(OP_A_MULACC_B, vjp_a_mulacc_b);
REG_VJP(OP_A_MULACC_BT, vjp_a_mulacc_bt);

} // namespace npu::tile_fwk
