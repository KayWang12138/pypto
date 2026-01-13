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

bool TileOpShapeVisitor::IsShapeCompatibleForBinaryOp(
    const std::vector<int64_t> &inputShape, const std::vector<int64_t> &outputShape) {
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
    // Start from the rightmost dimension
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

void TileOpShapeVisitor::VisitOp_(OperationPtr &op) {
    if (!op)
        return;

    // Check UnaryOp
    if (auto unaryOp = std::dynamic_pointer_cast<UnaryOp>(op)) {
        CheckUnaryOpShape(unaryOp, op);
    }
    // Check BinaryOp
    else if (auto binaryOp = std::dynamic_pointer_cast<BinaryOp>(op)) {
        CheckBinaryOpShape(binaryOp, op);
    }
    // Check BinaryScalarMixOp
    else if (auto binaryScalarMixOp = std::dynamic_pointer_cast<BinaryScalarMixOp>(op)) {
        CheckBinaryScalarMixOpShape(binaryScalarMixOp, op);
    }
}

// ---- Concrete ops (auto-generated from *.def) ----
#define DEFOP(name, inherit, opcode, ...)                             \
    void TileOpShapeVisitor::VisitOp_(name##Ptr &op) {                         \
        OperationPtr opPtr = std::static_pointer_cast<Operation>(op); \
        VisitOp_(opPtr);                                              \
    }
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

void TileOpShapeVisitor::CheckUnaryOpShape(UnaryOpPtr &unaryOp, OperationPtr &op) {
    auto input = std::dynamic_pointer_cast<TileValue>(unaryOp->GetInputOperand(0));
    auto output = std::dynamic_pointer_cast<TileValue>(unaryOp->GetOutputOperand(0));

    if (input && output) {
        const auto &inputShape = input->GetShape();
        const auto &outputShape = output->GetShape();

        if (inputShape != outputShape) {
            std::string opName = GetOpcodeName(op->GetOpcode());
            std::string msg = "UnaryOp '" + opName + "': input shape [";
            if (!inputShape.empty()) {
                msg += std::to_string(inputShape[0]);
                for (size_t i = 1; i < inputShape.size(); ++i) {
                    msg += ", " + std::to_string(inputShape[i]);
                }
            }
            msg += "] != output shape [";
            if (!outputShape.empty()) {
                msg += std::to_string(outputShape[0]);
                for (size_t i = 1; i < outputShape.size(); ++i) {
                    msg += ", " + std::to_string(outputShape[i]);
                }
            }
            msg += "]";
            violations_.push_back(msg);
        }
    }
}

void TileOpShapeVisitor::CheckBinaryOpShape(BinaryOpPtr &binaryOp, OperationPtr &op) {
    auto lhs = std::dynamic_pointer_cast<TileValue>(binaryOp->GetInputOperand(0));
    auto rhs = std::dynamic_pointer_cast<TileValue>(binaryOp->GetInputOperand(1));
    auto output = std::dynamic_pointer_cast<TileValue>(binaryOp->GetOutputOperand(0));

    if (lhs && rhs && output) {
        const auto &lhsShape = lhs->GetShape();
        const auto &rhsShape = rhs->GetShape();
        const auto &outputShape = output->GetShape();

        // Rule: "输入和输出的 shape 一致，或仅有一个输入的 shape 中仅有维数是1的维 shape 与输出不一致"
        // This means: both inputs match output, OR only one input differs and only in dimensions that are 1
        bool lhsMatches = (lhsShape == outputShape);
        bool rhsMatches = (rhsShape == outputShape);
        bool bothMatch = lhsMatches && rhsMatches;

        // Check if only one input differs and only in dimensions that are 1
        bool lhsCompatible = IsShapeCompatibleForBinaryOp(lhsShape, outputShape);
        bool rhsCompatible = IsShapeCompatibleForBinaryOp(rhsShape, outputShape);
        bool onlyOneDiffers =
            (lhsMatches && !rhsMatches && rhsCompatible) || (!lhsMatches && rhsMatches && lhsCompatible);

        if (!bothMatch && !onlyOneDiffers) {
            std::string opName = GetOpcodeName(op->GetOpcode());
            std::string msg = "BinaryOp '" + opName + "': lhs shape [";
            if (!lhsShape.empty()) {
                msg += std::to_string(lhsShape[0]);
                for (size_t i = 1; i < lhsShape.size(); ++i) {
                    msg += ", " + std::to_string(lhsShape[i]);
                }
            }
            msg += "], rhs shape [";
            if (!rhsShape.empty()) {
                msg += std::to_string(rhsShape[0]);
                for (size_t i = 1; i < rhsShape.size(); ++i) {
                    msg += ", " + std::to_string(rhsShape[i]);
                }
            }
            msg += "] incompatible with output shape [";
            if (!outputShape.empty()) {
                msg += std::to_string(outputShape[0]);
                for (size_t i = 1; i < outputShape.size(); ++i) {
                    msg += ", " + std::to_string(outputShape[i]);
                }
            }
            msg += "]";
            violations_.push_back(msg);
        }
    }
}

void TileOpShapeVisitor::CheckBinaryScalarMixOpShape(BinaryScalarMixOpPtr &binaryScalarMixOp, OperationPtr &op) {
    auto input = std::dynamic_pointer_cast<TileValue>(binaryScalarMixOp->GetInputOperand(0));
    auto output = std::dynamic_pointer_cast<TileValue>(binaryScalarMixOp->GetOutputOperand(0));

    if (input && output) {
        const auto &lhsShape = input->GetShape();
        const auto &outputShape = output->GetShape();

        if (lhsShape != outputShape) {
            std::string opName = GetOpcodeName(op->GetOpcode());
            std::string msg = "BinaryScalarMixOp '" + opName + "': input Tile shape [";
            if (!lhsShape.empty()) {
                msg += std::to_string(lhsShape[0]);
                for (size_t i = 1; i < lhsShape.size(); ++i) {
                    msg += ", " + std::to_string(lhsShape[i]);
                }
            }
            msg += "] != output shape [";
            if (!outputShape.empty()) {
                msg += std::to_string(outputShape[0]);
                for (size_t i = 1; i < outputShape.size(); ++i) {
                    msg += ", " + std::to_string(outputShape[i]);
                }
            }
            msg += "]";
            violations_.push_back(msg);
        }
    }
}

VerifyResult RuleVerifyOpShape(ProgramModulePtr program) {
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
