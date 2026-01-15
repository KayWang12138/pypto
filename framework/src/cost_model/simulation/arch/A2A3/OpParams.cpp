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
 * \file OpParams.cpp
 * \brief
 */

#include "OpParams.h"
#include "interface/operation/operation.h"
#include <cmath>
#include <numeric>

namespace CostModel {

using npu::tile_fwk::Opcode;

// ============================================================================
// Implementation of Calculators
// ============================================================================

// Pre-defined Strategy: Constant Latency
class Constant : public LatencyCalculator {
    int latency_;
public:
    Constant(int latency) : latency_(latency) {}
    int Calculate(const npu::tile_fwk::Operation* op) const override {
        return latency_;
    }
};

// Pre-defined Strategy: Linear function of Total Elements (Shape)
// y = k * shape_elements + b
class LinearShape : public LatencyCalculator {
    float k_;
    float b_;
public:
    LinearShape(float k, float b) : k_(k), b_(b) {}
    int Calculate(const npu::tile_fwk::Operation* op) const override {
        const npu::tile_fwk::Shape* shapePtr = nullptr;
        // Prioritize Input Shape
        if (op->GetInputOperandSize() > 0 && op->GetInputOperand(0)) {
            shapePtr = &op->GetInputOperand(0)->GetShape();
        } else if (op->GetOutputOperandSize() > 0 && op->GetOutputOperand(0)) {
            shapePtr = &op->GetOutputOperand(0)->GetShape();
        }

        int shapeElements = 1;
        if (shapePtr != nullptr && !shapePtr->empty()) {
            shapeElements = std::accumulate(shapePtr->begin(), shapePtr->end(), 1, std::multiplies<int>());
        }
        return static_cast<int>(std::ceil(shapeElements * k_ + b_));
    }
};

// Pre-defined Strategy: Linear function of Data Size (Bytes)
// y = k * size_bytes + b
class LinearSize : public LatencyCalculator {
    float k_;
    float b_;
public:
    LinearSize(float k, float b) : k_(k), b_(b) {}
    int Calculate(const npu::tile_fwk::Operation* op) const override {
        int dataSize = 0;
        // Prioritize Input Shape
        if (op->GetInputOperandSize() > 0 && op->GetInputOperand(0)) {
            dataSize = op->GetInputOperand(0)->GetDataSize();
        } else if (op->GetOutputOperandSize() > 0 && op->GetOutputOperand(0)) {
            dataSize = op->GetOutputOperand(0)->GetDataSize();
        }
        return static_cast<int>(std::ceil(dataSize * k_ + b_));
    }
};

// Pre-defined Strategy: Linear function of Aligned Data Size
// size = m * ceil(n * sizeof(dtype)/256)
// latency = k * size + b
class LinearSizeAligned : public LatencyCalculator {
    float k_;
    float b_;
public:
    LinearSizeAligned(float k, float b) : k_(k), b_(b) {}
    int Calculate(const npu::tile_fwk::Operation* op) const override {
        const npu::tile_fwk::Shape* shapePtr = nullptr;
        DataType dtype = DataType::DT_BOTTOM;

        // Prioritize Input Operand
        if (op->GetInputOperandSize() > 0 && op->GetInputOperand(0)) {
            shapePtr = &op->GetInputOperand(0)->GetShape();
            dtype = op->GetInputOperand(0)->Datatype();
        } else if (op->GetOutputOperandSize() > 0 && op->GetOutputOperand(0)) {
            shapePtr = &op->GetOutputOperand(0)->GetShape();
            dtype = op->GetOutputOperand(0)->Datatype();
        }

        if (shapePtr == nullptr || shapePtr->empty()) {
            return static_cast<int>(b_);
        }

        int n = shapePtr->back();
        int m = 1;
        if (shapePtr->size() > 1) {
             m = std::accumulate(shapePtr->begin(), shapePtr->end() - 1, 1, std::multiplies<int>());
        }

        size_t dsize = npu::tile_fwk::BytesOf(dtype);
        // size = m * ceil(n * sizeof(dtype)/256)
        double aligned_n_blocks = std::ceil((static_cast<double>(n) * dsize) / 256.0);
        double size = static_cast<double>(m) * aligned_n_blocks;

        return static_cast<int>(std::ceil(size * k_ + b_));
    }
};

// Pre-defined Strategy: Linear function of Reduction Output Elements
// Proportional to the accumulated shape excluding the summation axis
// Equivalent to Output Shape Elements
// y = k * output_elements + b
class LinearReduce : public LatencyCalculator {
    float k_;
    float b_;
public:
    LinearReduce(float k, float b) : k_(k), b_(b) {}
    int Calculate(const npu::tile_fwk::Operation* op) const override {
        const npu::tile_fwk::Shape* shapePtr = nullptr;
        if (op->GetOutputOperandSize() > 0 && op->GetOutputOperand(0)) {
            shapePtr = &op->GetOutputOperand(0)->GetShape();
        }
        
        int outputElements = 1;
        if (shapePtr != nullptr && !shapePtr->empty()) {
            outputElements = std::accumulate(shapePtr->begin(), shapePtr->end(), 1, std::multiplies<int>());
        }
        return static_cast<int>(std::ceil(outputElements * k_ + b_));
    }
};

// Pre-defined Strategy: Linear function of Matmul (m*k*n)
// Compute volume = sqrt(elementsA * elementsB * elementsC) which equals m*k*n
// y = k * volume + b
class LinearMatmul : public LatencyCalculator {
public:
    LinearMatmul() {}
    int Calculate(const npu::tile_fwk::Operation* op) const override {
        // 1. Define constants for different data types
        const float k_fp32 = 0.0097656f;
        const float b_fp32 = 21.0f;
        const float k_fp16 = 0.0024414f;
        const float b_fp16 = 21.0f;

        // 2. Select parameters based on DType
        float k = k_fp16;
        float b = b_fp16;
        
        if (op->GetInputOperandSize() > 0 && op->GetInputOperand(0)) {
            if (op->GetInputOperand(0)->Datatype() == DataType::DT_FP32) {
                k = k_fp32;
                b = b_fp32;
            }
        }

        // 3. Calculate volume m*k*n
        if (op->GetInputOperandSize() < 2 || op->GetOutputOperandSize() < 1) {
            return static_cast<int>(b);
        }
        
        const auto& input0 = op->GetInputOperand(0);
        const auto& input1 = op->GetInputOperand(1);
        const auto& output0 = op->GetOutputOperand(0);
        
        if (!input0 || !input1 || !output0) {
            return static_cast<int>(b);
        }

        auto get_elements = [](const npu::tile_fwk::LogicalTensorPtr& tensor) {
            const auto& shape = tensor->GetShape();
            if (shape.empty()) return 1LL;
            return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
        };

        int64_t elementsA = get_elements(input0);
        int64_t elementsB = get_elements(input1);
        int64_t elementsC = get_elements(output0);

        // volume = sqrt(elementsA * elementsB * elementsC) = sqrt(m^2 * k^2 * n^2) = m*k*n
        // Use double for intermediate calculation to prevent precision loss before sqrt
        double volume = std::sqrt(static_cast<double>(elementsA) * elementsB * elementsC);
        
        return static_cast<int>(std::ceil(volume * k + b));
    }
};

// ============================================================================
// Registration
// ============================================================================

// Default
OP_LATENCY_REGISTER_DEFAULT(Constant(1));

// Binary
OP_LATENCY_REGISTER(Opcode::OP_ADD, LinearSizeAligned(1.99f, 26.6f));
OP_LATENCY_REGISTER(Opcode::OP_SUB, LinearSizeAligned(1.99f, 26.6f));
OP_LATENCY_REGISTER(Opcode::OP_MUL, LinearSizeAligned(1.98f, 33.15f));
OP_LATENCY_REGISTER(Opcode::OP_DIV, LinearShape(0.0625f, 28.0f));

// Unary
OP_LATENCY_REGISTER(Opcode::OP_ADDS, LinearSize(0.0041325f, 21.0f));
OP_LATENCY_REGISTER(Opcode::OP_MULS, LinearSize(0.0041425f, 21.2f));
OP_LATENCY_REGISTER(Opcode::OP_EXP,  LinearShape(0.03125f, 27.0f));
OP_LATENCY_REGISTER(Opcode::OP_SQRT, LinearShape(0.03125f, 28.0f));

// Reduce
OP_LATENCY_REGISTER(Opcode::OP_ROWSUM_SINGLE, LinearReduce(7.0f, 36.0f));
OP_LATENCY_REGISTER(Opcode::OP_ROWMAX_SINGLE, LinearReduce(7.0f, 24.0f));
OP_LATENCY_REGISTER(Opcode::OP_ROWMIN_SINGLE, LinearReduce(7.0f, 24.0f));

// Matmul
OP_LATENCY_REGISTER(Opcode::OP_A_MUL_B, LinearMatmul());

// MTE
OP_LATENCY_REGISTER(Opcode::OP_COPY_IN, LinearSize(0.009475f, 233.4f));
OP_LATENCY_REGISTER(Opcode::OP_COPY_OUT, LinearSize(0.009475f, 233.4f));
OP_LATENCY_REGISTER(Opcode::OP_L1_COPY_IN, LinearSize(0.0067f, 764.86f));
OP_LATENCY_REGISTER(Opcode::OP_L0C_COPY_OUT, LinearSize(0.009125f, 247.8f));

} // namespace CostModel
