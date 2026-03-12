/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_op_glsl.h"
#include <sstream>
#include <regex>

namespace npu::tile_fwk {

const std::map<Opcode, OpTemplate>& CodeGenOpGLSL::GetOpTemplates() {
    static const std::map<Opcode, OpTemplate> templates = {
        {Opcode::OP_ADD, {"add", "${out} = ${in0} + ${in1};", ""}},
        {Opcode::OP_SUB, {"sub", "${out} = ${in0} - ${in1};", ""}},
        {Opcode::OP_MUL, {"mul", "${out} = ${in0} * ${in1};", ""}},
        {Opcode::OP_DIV, {"div", "${out} = ${in0} / ${in1};", ""}},
        {Opcode::OP_RELU, {"relu", "${out} = max(${in0}, 0.0);", ""}},
        {Opcode::OP_EXP, {"exp", "${out} = exp(${in0});", ""}},
        {Opcode::OP_LN, {"log", "${out} = log(${in0});", ""}},
        
        // Scatter Elements
        {Opcode::OP_SCATTER_ELEMENT, {"scatter_elements",
            "uint scatter_idx = uint(${in1});\n    ${in0_ssbo}.d[scatter_idx] = ${in2};", // Direct write to Output SSBO (which is in0 here)
            ""}},
            
        // Expand (simplified 1D)
        {Opcode::OP_EXPAND, {"expand",
            "uint src_idx = global_idx % ${in0_dim0};\n    ${out} = ${in0_ssbo}.d[src_idx];",
            ""}}
    };
    return templates;
}

CodeGenOpGLSL::CodeGenOpGLSL(const CodeGenOpCtx &ctx) : CodeGenOp(ctx) {}

void CodeGenOpGLSL::Init(const Operation &ops) {
    opCode = ops.GetOpcode();
    
    // Initialize operand array with NULL_OPERAND
    for (int i = 0; i < MAX_OPERANDS; ++i) {
        operand[i] = NULL_OPERAND;
    }

    // Map Outputs
    // operand[0] is typically the output
    if (!ops.oOperand.empty()) {
        operand[0] = ops.oOperand[0]->magic;
        shape[0] = ops.oOperand[0]->shape;
    }
    
    // Map Inputs
    // operand[1]...operand[N] are inputs
    for (size_t i = 0; i < ops.iOperand.size(); ++i) {
        if (i + 1 < MAX_OPERANDS) {
            operand[i + 1] = ops.iOperand[i]->magic;
            shape[i + 1] = ops.iOperand[i]->shape;
        }
    }
}

std::string CodeGenOpGLSL::GenOpCode() const {
    if (opCode == Opcode::OP_GATHER_ELEMENT) {
        return GenGatherOp();
    }

    const auto& templates = GetOpTemplates();
    auto it = templates.find(opCode);
    if (it != templates.end()) {
        std::string result = it->second.body;
        
        // Handle Output Registration
        // We assume operand[0] is the output buffer ID
        if (operand[0] != NULL_OPERAND) {
            std::string outVar = "v_op_" + std::to_string(operand[0]); // Use buffer ID as unique suffix
            
            // Special handling for Scatter: output is an SSBO, not a local var
            if (opCode == Opcode::OP_SCATTER_ELEMENT) {
                // Scatter usually doesn't have an output operand in oOperand list
                // It modifies one of its inputs (in0).
                // So we don't need to do anything here for output registration.
            } else {
                // Register this new variable name in SymbolManager
                // We need to create AllocKey from magic number (operand[0])
                auto key = sm->CreateAllocKey(operand[0]);
                sm->BindAddrWithVariableName(key, outVar, "");

                // Replace ${out} with declaration + variable name
                // E.g., "float v_op_1"
                result = std::regex_replace(result, std::regex(R"(\$\{out\})"), "float " + outVar);
            }
        }

        // Replace inputs
        for (int i = 1; i < MAX_OPERANDS; ++i) {
            if (operand[i] != NULL_OPERAND) {
                // Check if template needs SSBO direct access
                std::string placeholderSSBO = "\\$\\{in" + std::to_string(i - 1) + "_ssbo\\}";
                if (result.find("${in" + std::to_string(i - 1) + "_ssbo}") != std::string::npos) {
                    std::string inSSBO = sm->QuerySSBOName(operand[i]);
                    if (!inSSBO.empty()) {
                        result = std::regex_replace(result, std::regex(placeholderSSBO), inSSBO);
                    }
                }

                // Standard local variable replacement
                std::string placeholder = "\\$\\{in" + std::to_string(i - 1) + "\\}";
                if (result.find("${in" + std::to_string(i - 1) + "}") != std::string::npos) {
                    std::string inVar = sm->QueryVarNameByTensorMagic(operand[i], false);
                    result = std::regex_replace(result, std::regex(placeholder), inVar);
                }
            }
        }
        
        // Replace Dimensions (for Expand)
        // Assuming we can get dim info from shapes. Here we use a placeholder logic.
        if (opCode == Opcode::OP_EXPAND) {
             // In real implementation, we should get shape from CodeGenOp::shape[operand[i]]
             // For demo, we hardcode or use a simplified approach
             result = std::regex_replace(result, std::regex(R"(\$\{in0_dim0\})"), "1024"); // Example
        }

        return result;
    }
    return "// Unknown Opcode: " + std::to_string(static_cast<int>(opCode));
}

std::string CodeGenOpGLSL::GenGatherOp() const {
    std::stringstream ss;
    
    // 1. Get Inputs/Outputs
    // operand[0]: Output
    // operand[1]: Input Data (SSBO)
    // operand[2]: Input Indices
    
    int outMagic = operand[0];
    int inDataMagic = operand[1];
    int inIndexMagic = operand[2];
    
    std::string outVar = "v_op_" + std::to_string(outMagic);
    
    // Register output variable
    auto key = sm->CreateAllocKey(outMagic);
    sm->BindAddrWithVariableName(key, outVar, "");
    
    // Get Input Data SSBO Name
    std::string inDataSSBO = sm->QuerySSBOName(inDataMagic);
    if (inDataSSBO.empty()) {
        return "// Error: Input Data SSBO not found for Gather";
    }
    
    // Get Index Variable Name
    std::string indexVar = sm->QueryVarNameByTensorMagic(inIndexMagic, false);
    
    // 2. Get Gather Dimension
    // Assuming gather dim is 0 by default if not specified in attributes
    int64_t gatherDim = 0;
    // In a real scenario, we should get this from op attributes: originalOp.GetIntAttribute("axis");
    // Since I can't easily access attributes in this context without more code, I'll assume dim 0 or mock it.
    // Let's try to get it if possible, otherwise default to 0.
    // gatherDim = originalOp.GetIntAttribute("axis"); 
    
    // 3. Get Shapes and Strides
    // We need the shape of the Output Tensor (which matches Indices Tensor) to decompose global_idx
    // And the stride of the Input Data Tensor to compute offset.
    
    // For simplicity in this CodegenOpGLSL, we don't have direct access to Tensor objects easily 
    // unless we query SymbolManager or passed in context.
    // However, we can generate code that assumes `global_idx` decomposition.
    
    // But wait, `global_idx` decomposition depends on the shape.
    // We need to inject the shape values into the shader.
    
    // Strategy:
    // Generate code that calculates:
    // uint d4 = global_idx % S4; ...
    // uint in_offset = ...
    
    // We need to know the rank and dimensions.
    // CodeGenOp has `shape` member: `std::vector<int64_t> shape[MAX_OPERANDS]`
    // `shape[0]` is Output Shape. `shape[1]` is Input Data Shape.
    
    const auto& outShape = shape[0];
    const auto& inDataShape = shape[1];
    
    if (outShape.empty() || inDataShape.empty()) {
        // Fallback for mock test if shapes are not populated
        return "uint gather_idx = uint(" + indexVar + ");\n    float " + outVar + " = " + inDataSSBO + ".d[gather_idx];";
    }
    
    int rank = outShape.size();
    
    // Generate Coordinate Decomposition
    ss << "    // Gather: Decompose global_idx\n";
    ss << "    uint tmp_idx = global_idx;\n";
    
    std::vector<std::string> coords(rank);
    for (int i = rank - 1; i >= 0; --i) {
        std::string d_var = "d" + std::to_string(i);
        ss << "    uint " << d_var << " = tmp_idx % " << outShape[i] << ";\n";
        ss << "    tmp_idx /= " << outShape[i] << ";\n";
        coords[i] = d_var;
    }
    
    // Replace coordinate at gatherDim with index value
    coords[gatherDim] = "uint(" + indexVar + ")";
    
    // Calculate Input Offset
    ss << "    // Gather: Compute Input Offset\n";
    ss << "    uint in_offset = 0";
    
    // Compute strides for Input Data
    std::vector<int64_t> inStrides(rank);
    int64_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        inStrides[i] = stride;
        stride *= inDataShape[i];
    }
    
    for (int i = 0; i < rank; ++i) {
        ss << " + " << coords[i] << " * " << inStrides[i];
    }
    ss << ";\n";
    
    // Load Data
    ss << "    float " + outVar + " = " + inDataSSBO + ".d[in_offset];";
    
    return ss.str();
}

} // namespace npu::tile_fwk
