/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/verifier/shape_verify.h"

#include "ir/opcode.h"

#include <string>
#include <vector>

namespace pto {

bool TileOpShapeVisitor::IsShapeCompatibleForBinaryOp(const TileValuePtr &inputTile, const TileValuePtr &outputTile) {
    if (!inputTile || !outputTile) {
        return false;
    }

    // Validate validShape size matches shape size
    const auto &inputValidShape = inputTile->GetValidShape();
    const auto &inputShape = inputTile->GetShape();
    const auto &outputValidShape = outputTile->GetValidShape();
    const auto &outputShape = outputTile->GetShape();

    if (inputValidShape.size() != inputShape.size() || outputValidShape.size() != outputShape.size()) {
        return false;
    }

    // If shapes are exactly the same, they are compatible
    if (inputShape == outputShape) {
        return true;
    }

    // Check if input shape can broadcast to output shape
    // For broadcasting, input shape must be compatible: either same size or can be broadcasted
    size_t inputDims = inputShape.size();
    size_t outputDims = outputShape.size();

    // If input has more dimensions, it cannot broadcast to output
    if (inputDims > outputDims) {
        return false;
    }

    // Check if input can broadcast to output
    for (size_t i = 0; i < inputDims; ++i) {
        size_t inputIdx = inputDims - 1 - i;
        size_t outputIdx = outputDims - 1 - i;

        int64_t inputDim = inputShape[inputIdx];
        int64_t outputDim = outputShape[outputIdx];

        // If dimensions don't match, input dimension must be 1 for broadcasting
        if (inputDim != outputDim && inputDim != 1) {
            return false;
        }
    }

    return true;
}

std::string TileOpShapeVisitor::GetShapeStr(const std::vector<int64_t> &shape) const {
    std::string result = "[";
    if (!shape.empty()) {
        result += std::to_string(shape[0]);
        for (size_t i = 1; i < shape.size(); ++i) {
            result += ", " + std::to_string(shape[i]);
        }
    }
    result += "]";
    return result;
}

void TileOpShapeVisitor::VisitImplOp(OperationPtr &op) {
    if (!op)
        return;

    // Check Matmul operations first
    if (auto matmulLoad = std::dynamic_pointer_cast<MatmulLoadOp>(op)) {
        CheckMatmulLoadOpShape(matmulLoad);
    }
    else if (auto matmulExtract = std::dynamic_pointer_cast<MatmulExtractOp>(op)) {
        CheckMatmulExtractOpShape(matmulExtract);
    }
    else if (auto matmulMmad = std::dynamic_pointer_cast<MatmulMmadOp>(op)) {
        CheckMatmulMmadOpShape(matmulMmad);
    }
    else if (auto matmulAcc = std::dynamic_pointer_cast<MatmulAccOp>(op)) {
        CheckMatmulAccOpShape(matmulAcc);
    }
    else if (auto matmulStore = std::dynamic_pointer_cast<MatmulStoreOp>(op)) {
        CheckMatmulStoreOpShape(matmulStore);
    }
    else if (auto matmulBias = std::dynamic_pointer_cast<MatmulBiasOp>(op)) {
        CheckMatmulBiasOpShape(matmulBias);
    }
    else if (auto matmulQuant = std::dynamic_pointer_cast<MatmulQuantOp>(op)) {
        CheckMatmulQuantOpShape(matmulQuant);
    }
    // Check UnaryOp
    else if (auto unaryOp = std::dynamic_pointer_cast<UnaryOp>(op)) {
        CheckUnaryOpShape(unaryOp);
    }
    // Check BinaryOp
    else if (auto binaryOp = std::dynamic_pointer_cast<BinaryOp>(op)) {
        CheckBinaryOpShape(binaryOp);
    }
    // Check BinaryScalarMixOp
    else if (auto binaryScalarMixOp = std::dynamic_pointer_cast<BinaryScalarMixOp>(op)) {
        CheckBinaryScalarMixOpShape(binaryScalarMixOp);
    }
}

// ---- Concrete ops (auto-generated from *.def) ----
#define DEFOP(name, inherit, opcode, ...)                             \
    void TileOpShapeVisitor::VisitImplOp(name##Ptr &op) {                \
        OperationPtr opPtr = std::static_pointer_cast<Operation>(op); \
        VisitImplOp(opPtr);                                              \
    }
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

void TileOpShapeVisitor::CheckUnaryOpShape(UnaryOpPtr &unaryOp) {
    auto input = std::dynamic_pointer_cast<TileValue>(unaryOp->GetInputOperand(0));
    auto output = std::dynamic_pointer_cast<TileValue>(unaryOp->GetOutputOperand(0));

    if (input && output) {
        const auto &inputShape = input->GetShape();
        const auto &outputShape = output->GetShape();

        if (inputShape != outputShape) {
            std::string opName = GetOpcodeName(unaryOp->GetOpcode());
            std::string msg = "UnaryOp '" + opName + "': input shape " + GetShapeStr(inputShape) +
                              " != output shape " + GetShapeStr(outputShape);
            violations_.push_back(msg);
        }
    }
}

void TileOpShapeVisitor::CheckBinaryOpShape(BinaryOpPtr &binaryOp) {
    auto lhs = std::dynamic_pointer_cast<TileValue>(binaryOp->GetInputOperand(0));
    auto rhs = std::dynamic_pointer_cast<TileValue>(binaryOp->GetInputOperand(1));
    auto output = std::dynamic_pointer_cast<TileValue>(binaryOp->GetOutputOperand(0));

    if (lhs && rhs && output) {
        const auto &lhsShape = lhs->GetShape();
        const auto &rhsShape = rhs->GetShape();
        const auto &outputShape = output->GetShape();

        // Rule: "both inputs match output, OR only one input differs and only in dimensions that are 1"
        bool lhsMatches = (lhsShape == outputShape);
        bool rhsMatches = (rhsShape == outputShape);
        bool bothMatch = lhsMatches && rhsMatches;

        // Check if only one input differs and only in dimensions that are 1
        bool lhsCompatible = IsShapeCompatibleForBinaryOp(lhs, output);
        bool rhsCompatible = IsShapeCompatibleForBinaryOp(rhs, output);
        bool onlyOneDiffers =
            (lhsMatches && !rhsMatches && rhsCompatible) || (!lhsMatches && rhsMatches && lhsCompatible);

        if (!bothMatch && !onlyOneDiffers) {
            std::string opName = GetOpcodeName(binaryOp->GetOpcode());
            std::string msg = "BinaryOp '" + opName + "': lhs shape " + GetShapeStr(lhsShape) +
                              ", rhs shape " + GetShapeStr(rhsShape) +
                              " incompatible with output shape " + GetShapeStr(outputShape);
            violations_.push_back(msg);
        }
    }
}

void TileOpShapeVisitor::CheckBinaryScalarMixOpShape(BinaryScalarMixOpPtr &binaryScalarMixOp) {
    auto input = std::dynamic_pointer_cast<TileValue>(binaryScalarMixOp->GetInputOperand(0));
    auto output = std::dynamic_pointer_cast<TileValue>(binaryScalarMixOp->GetOutputOperand(0));

    if (input && output) {
        const auto &lhsShape = input->GetShape();
        const auto &outputShape = output->GetShape();

        if (lhsShape != outputShape) {
            std::string opName = GetOpcodeName(binaryScalarMixOp->GetOpcode());
            std::string msg = "BinaryScalarMixOp '" + opName + "': input tile shape " + GetShapeStr(lhsShape) +
                              " != output tile shape " + GetShapeStr(outputShape);
            violations_.push_back(msg);
        }
    }
}

// Helper functions for matmul shape verification
bool TileOpShapeVisitor::IsTransposeOpcode(Opcode opcode) {
    return opcode == Opcode::OP_L1_TO_L0_AT || opcode == Opcode::OP_L1_TO_L0_BT;
}

bool TileOpShapeVisitor::IsTransposeA(Opcode opcode) {
    return opcode == Opcode::OP_L1_TO_L0_AT;
}

bool TileOpShapeVisitor::IsTransposeB(Opcode opcode) {
    return opcode == Opcode::OP_L1_TO_L0_BT;
}

std::vector<int64_t> TileOpShapeVisitor::GetTransposedShape(const std::vector<int64_t> &shape) {
    if (shape.size() < 2) {
        return shape;
    }
    std::vector<int64_t> transposed = shape;
    std::swap(transposed[transposed.size() - 2], transposed[transposed.size() - 1]);
    return transposed;
}

// Matmul operation shape verification implementations
void TileOpShapeVisitor::CheckMatmulLoadOpShape(MatmulLoadOpPtr &op) {
    auto input = op->GetInOperand(0);
    auto output = op->GetOutOperand(0);

    if (input && output) {
        const auto &inputShape = input->GetShape();
        const auto &outputShape = output->GetShape();

        // MatmulLoadOp: input and output shapes should match (data copy operation)
        if (inputShape != outputShape) {
            std::string opName = GetOpcodeName(op->GetOpcode());
            std::string msg = "MatmulLoadOp '" + opName + "': input shape " + GetShapeStr(inputShape) +
                              " != output shape " + GetShapeStr(outputShape);
            violations_.push_back(msg);
        }
    }
}

void TileOpShapeVisitor::CheckMatmulExtractOpShape(MatmulExtractOpPtr &op) {
    auto input = op->GetInOperand(0);
    auto output = op->GetOutOperand(0);

    if (input && output) {
        const auto &inputShape = input->GetShape();
        const auto &outputShape = output->GetShape();
        Opcode opcode = op->GetOpcode();

        // Check if this is a transpose operation
        bool transposeA = IsTransposeA(opcode);
        bool transposeB = IsTransposeB(opcode);
        bool isTranspose = transposeA || transposeB;

        if (isTranspose) {
            // For transpose operations, output shape should be transpose of input shape
            std::vector<int64_t> expectedOutputShape = GetTransposedShape(inputShape);
            if (outputShape != expectedOutputShape) {
                std::string opName = GetOpcodeName(opcode);
                std::string msg = "MatmulExtractOp '" + opName + "': input shape " + GetShapeStr(inputShape) +
                                  " with transpose, expected output shape " + GetShapeStr(expectedOutputShape) +
                                  " but got " + GetShapeStr(outputShape);
                violations_.push_back(msg);
            }
        } else {
            // For non-transpose operations, input and output shapes should match
            if (inputShape != outputShape) {
                std::string opName = GetOpcodeName(opcode);
                std::string msg = "MatmulExtractOp '" + opName + "': input shape " + GetShapeStr(inputShape) +
                                  " != output shape " + GetShapeStr(outputShape);
                violations_.push_back(msg);
            }
        }
    }
}

void TileOpShapeVisitor::CheckMatmulMmadOpShape(MatmulMmadOpPtr &op) {
    auto lhs = op->GetInOperand(0);
    auto rhs = op->GetInOperand(1);
    auto output = op->GetOutOperand(0);

    if (lhs && rhs && output) {
        const auto &lhsShape = lhs->GetShape();
        const auto &rhsShape = rhs->GetShape();
        const auto &outputShape = output->GetShape();
        Opcode opcode = op->GetOpcode();

        // MatmulMmadOp only supports OP_A_MUL_B in tile_graph.def
        // Transpose information may be in attributes, but for now we assume no transpose
        // If transpose is needed, it should be handled at the MatmulExtractOp level
        bool transposeA = false;
        bool transposeB = false;

        // Matrix multiplication: C = A * B
        // If A is [M, K] and B is [K, N], then C is [M, N]
        // If A is transposed: A^T is [K, M], B is [K, N], then C is [M, N]
        // If B is transposed: A is [M, K], B^T is [N, K], then C is [M, N]
        // If both are transposed: A^T is [K, M], B^T is [N, K], then C is [M, N]

        if (lhsShape.size() < 2 || rhsShape.size() < 2 || outputShape.size() < 2) {
            std::string opName = GetOpcodeName(opcode);
            std::string msg = "MatmulMmadOp '" + opName + "': operands must have at least 2 dimensions";
            violations_.push_back(msg);
            return;
        }

        // Get effective dimensions considering transpose
        int64_t aM = transposeA ? lhsShape[lhsShape.size() - 1] : lhsShape[lhsShape.size() - 2];
        int64_t aK = transposeA ? lhsShape[lhsShape.size() - 2] : lhsShape[lhsShape.size() - 1];
        int64_t bK = transposeB ? rhsShape[rhsShape.size() - 1] : rhsShape[rhsShape.size() - 2];
        int64_t bN = transposeB ? rhsShape[rhsShape.size() - 2] : rhsShape[rhsShape.size() - 1];

        // Check K dimension matches
        if (aK != bK) {
            std::string opName = GetOpcodeName(opcode);
            std::string msg = "MatmulMmadOp '" + opName + "': K dimension mismatch, lhs K=" + std::to_string(aK) +
                              ", rhs K=" + std::to_string(bK);
            violations_.push_back(msg);
            return;
        }

        // Check output shape: [M, N]
        int64_t expectedM = aM;
        int64_t expectedN = bN;

        // Handle batch dimensions if present
        std::vector<int64_t> expectedOutputShape;
        if (lhsShape.size() > 2) {
            // Has batch dimensions, copy them from lhs (assuming batch dimensions match)
            expectedOutputShape = lhsShape;
            expectedOutputShape[expectedOutputShape.size() - 2] = expectedM;
            expectedOutputShape[expectedOutputShape.size() - 1] = expectedN;
        } else {
            expectedOutputShape = {expectedM, expectedN};
        }

        if (outputShape != expectedOutputShape) {
            std::string opName = GetOpcodeName(opcode);
            std::string msg = "MatmulMmadOp '" + opName + "': expected output shape " + GetShapeStr(expectedOutputShape) +
                              " but got " + GetShapeStr(outputShape) +
                              " (lhs=" + GetShapeStr(lhsShape) + ", rhs=" + GetShapeStr(rhsShape) +
                              ", transposeA=" + (transposeA ? "true" : "false") +
                              ", transposeB=" + (transposeB ? "true" : "false") + ")";
            violations_.push_back(msg);
        }
    }
}

void TileOpShapeVisitor::CheckMatmulAccOpShape(MatmulAccOpPtr &op) {
    auto lhs = op->GetInOperand(0);
    auto rhs = op->GetInOperand(1);
    auto output = op->GetOutOperand(0);

    if (lhs && rhs && output) {
        const auto &lhsShape = lhs->GetShape();
        const auto &rhsShape = rhs->GetShape();
        const auto &outputShape = output->GetShape();
        Opcode opcode = op->GetOpcode();

        // MatmulAccOp only supports OP_A_MULACC_B in tile_graph.def
        // Transpose information may be in attributes, but for now we assume no transpose
        // If transpose is needed, it should be handled at the MatmulExtractOp level
        bool transposeA = false;
        bool transposeB = false;

        // MatmulAccOp is similar to MatmulMmadOp but accumulates into output
        // The output shape should match the result of matrix multiplication
        if (lhsShape.size() < 2 || rhsShape.size() < 2 || outputShape.size() < 2) {
            std::string opName = GetOpcodeName(opcode);
            std::string msg = "MatmulAccOp '" + opName + "': operands must have at least 2 dimensions";
            violations_.push_back(msg);
            return;
        }

        // Get effective dimensions considering transpose
        int64_t aM = transposeA ? lhsShape[lhsShape.size() - 1] : lhsShape[lhsShape.size() - 2];
        int64_t aK = transposeA ? lhsShape[lhsShape.size() - 2] : lhsShape[lhsShape.size() - 1];
        int64_t bK = transposeB ? rhsShape[rhsShape.size() - 1] : rhsShape[rhsShape.size() - 2];
        int64_t bN = transposeB ? rhsShape[rhsShape.size() - 2] : rhsShape[rhsShape.size() - 1];

        // Check K dimension matches
        if (aK != bK) {
            std::string opName = GetOpcodeName(opcode);
            std::string msg = "MatmulAccOp '" + opName + "': K dimension mismatch, lhs K=" + std::to_string(aK) +
                              ", rhs K=" + std::to_string(bK);
            violations_.push_back(msg);
            return;
        }

        // Check output shape: [M, N] (should match the accumulation target)
        int64_t expectedM = aM;
        int64_t expectedN = bN;

        // Handle batch dimensions if present
        std::vector<int64_t> expectedOutputShape;
        if (lhsShape.size() > 2) {
            expectedOutputShape = lhsShape;
            expectedOutputShape[expectedOutputShape.size() - 2] = expectedM;
            expectedOutputShape[expectedOutputShape.size() - 1] = expectedN;
        } else {
            expectedOutputShape = {expectedM, expectedN};
        }

        if (outputShape != expectedOutputShape) {
            std::string opName = GetOpcodeName(opcode);
            std::string msg = "MatmulAccOp '" + opName + "': expected output shape " + GetShapeStr(expectedOutputShape) +
                              " but got " + GetShapeStr(outputShape) +
                              " (lhs=" + GetShapeStr(lhsShape) + ", rhs=" + GetShapeStr(rhsShape) +
                              ", transposeA=" + (transposeA ? "true" : "false") +
                              ", transposeB=" + (transposeB ? "true" : "false") + ")";
            violations_.push_back(msg);
        }
    }
}

void TileOpShapeVisitor::CheckMatmulStoreOpShape(MatmulStoreOpPtr &op) {
    auto input = op->GetInOperand(0);
    auto output = op->GetOutOperand(0);

    if (input && output) {
        const auto &inputShape = input->GetShape();
        const auto &outputShape = output->GetShape();

        // MatmulStoreOp: input and output shapes should match (data copy operation)
        if (inputShape != outputShape) {
            std::string opName = GetOpcodeName(op->GetOpcode());
            std::string msg = "MatmulStoreOp '" + opName + "': input shape " + GetShapeStr(inputShape) +
                              " != output shape " + GetShapeStr(outputShape);
            violations_.push_back(msg);
        }
    }
}

void TileOpShapeVisitor::CheckMatmulBiasOpShape(MatmulBiasOpPtr &op) {
    auto input = op->GetInOperand(0);
    auto output = op->GetOutOperand(0);

    if (input && output) {
        const auto &inputShape = input->GetShape();
        const auto &outputShape = output->GetShape();

        // MatmulBiasOp: input and output shapes should match (bias addition)
        if (inputShape != outputShape) {
            std::string opName = GetOpcodeName(op->GetOpcode());
            std::string msg = "MatmulBiasOp '" + opName + "': input shape " + GetShapeStr(inputShape) +
                              " != output shape " + GetShapeStr(outputShape);
            violations_.push_back(msg);
        }
    }
}

void TileOpShapeVisitor::CheckMatmulQuantOpShape(MatmulQuantOpPtr &op) {
    auto input = op->GetInOperand(0);
    auto output = op->GetOutOperand(0);

    if (input && output) {
        const auto &inputShape = input->GetShape();
        const auto &outputShape = output->GetShape();

        // MatmulQuantOp: input and output shapes should match (quantization operation)
        if (inputShape != outputShape) {
            std::string opName = GetOpcodeName(op->GetOpcode());
            std::string msg = "MatmulQuantOp '" + opName + "': input shape " + GetShapeStr(inputShape) +
                              " != output shape " + GetShapeStr(outputShape);
            violations_.push_back(msg);
        }
    }
}

VerifyResult VerifyOpShape(ProgramModulePtr program) {
    if (!program) {
        return {false, "ProgramModule is null, cannot verify operation shapes"};
    }

    TileOpShapeVisitor visitor;
    ProgramModulePtr programPtr = program;
    visitor.VisitProgram(programPtr);

    if (!visitor.violations_.empty()) {
        std::string errorMsg = "Operation shape verification failed - " + std::to_string(visitor.violations_.size()) +
                               " operation(s) have incompatible shapes:\n";
        for (size_t i = 0; i < visitor.violations_.size(); ++i) {
            errorMsg += "  " + std::to_string(i + 1) + ". " + visitor.violations_[i];
            if (i + 1 < visitor.violations_.size()) {
                errorMsg += "\n";
            }
        }
        return {false, errorMsg};
    }

    return {true, ""};
}

} // namespace pto
