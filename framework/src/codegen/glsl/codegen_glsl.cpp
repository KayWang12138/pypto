/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_glsl.h"
#include "interface/utils/file_utils.h"
#include <fstream>
#include <iostream>

namespace npu::tile_fwk {

CodeGenGLSL::CodeGenGLSL(CodeGenCtx cctx) : ctx(std::move(cctx)) {
    CreateMultiLevelDir(ctx.cceDir); // Reuse cceDir for output path
}

void CodeGenGLSL::GenCode(Function &topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>> &) {
    std::string code;
    
    // 1. Collect used ops to generate helper functions
    for (auto &op : topFunc.Operations()) {
        usedOps.insert(op.GetOpcode());
    }

    // 2. Generate parts
    code += GenerateHeader();
    code += GenerateInputsOutputs(topFunc);
    code += GenerateHelperFunctions();
    code += GenerateMain(topFunc);

    // 3. Write to file
    std::string filename = ctx.cceDir + "/shader.comp";
    std::ofstream outfile(filename);
    if (outfile.is_open()) {
        outfile << code;
        outfile.close();
        std::cout << "Generated GLSL shader at: " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

std::string CodeGenGLSL::GenerateHeader() {
    return "#version 450\n"
           "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
}

std::string CodeGenGLSL::GenerateInputsOutputs(Function &func) {
    std::stringstream ss;
    int binding = 0;
    
    // Create SymbolManager if not exists (it should be part of CodeGenCtx or created here)
    // For simplicity, we assume we can create one here and reuse it in GenerateMain.
    // However, GenerateMain creates its own SymbolManager.
    // We should probably make SymbolManager a member of CodeGenGLSL or pass it around.
    // But CodeGenOpCtx takes shared_ptr<SymbolManager>.
    
    // Let's iterate inputs to generate SSBOs
    ss << "// Input Buffers\n";
    for (size_t i = 0; i < func.inCasts_.size(); ++i) {
        ss << "layout(std430, binding = " << binding++ << ") buffer In" << i << " { float d[]; } arg" << i << ";\n";
    }

    ss << "\n// Output Buffers\n";
    for (size_t i = 0; i < func.outCasts_.size(); ++i) {
        ss << "layout(std430, binding = " << binding++ << ") buffer Out" << i << " { float d[]; } out" << i << ";\n";
    }
    
    return ss.str();
}

std::string CodeGenGLSL::GenerateHelperFunctions() {
    std::stringstream ss;
    ss << "\n// Helper Functions\n";
    const auto& templates = CodeGenOpGLSL::GetOpTemplates();
    
    for (auto opcode : usedOps) {
        auto it = templates.find(opcode);
        if (it != templates.end() && !it->second.funcDef.empty()) {
            ss << it->second.funcDef << "\n";
        }
    }
    return ss.str();
}

std::string CodeGenGLSL::GenerateMain(Function &func) {
    std::stringstream ss;
    ss << "\nvoid main() {\n";
    // Tile-based mapping: 
    // - Work Group -> Tile (Grid)
    // - Local Invocation -> Element within Tile
    // Assuming a simple 1D mapping for now, but scalable to 2D/3D
    ss << "    uint tile_size = gl_WorkGroupSize.x;\n";
    ss << "    uint tile_id = gl_WorkGroupID.x;\n";
    ss << "    uint local_id = gl_LocalInvocationID.x;\n";
    ss << "    uint global_idx = tile_id * tile_size + local_id;\n";
    
    // Create SymbolManager
    auto sm = std::make_shared<SymbolManager>();
    
    // Register all tensors to SymbolManager
    for (auto &op : func.Operations()) {
        for (const auto& t : op.iOperand) {
            sm->AddToTensorMap(t->magic, t);
        }
        for (const auto& t : op.oOperand) {
            sm->AddToTensorMap(t->magic, t);
        }
    }
    for (const auto& t : func.inCasts_) {
        sm->AddToTensorMap(t->magic, t);
    }
    for (const auto& t : func.outCasts_) {
        sm->AddToTensorMap(t->magic, t);
    }
    
    // 0. Register Output SSBOs (for Scatter support)
    for (size_t i = 0; i < func.outCasts_.size(); ++i) {
         std::string ssboName = "out" + std::to_string(i);
         if (func.outCasts_[i]->magic != 0) {
             sm->BindSSBOName(func.outCasts_[i]->magic, ssboName);
         }
    }

    // 1. Load Inputs
    for (size_t i = 0; i < func.inCasts_.size(); ++i) {
        std::string ssboName = "arg" + std::to_string(i);
        std::string localName = "v_in_" + std::to_string(i);
        ss << "    float " << localName << " = " << ssboName << ".d[global_idx];\n";
        
        // Register tensor to localName
        auto key = sm->CreateAllocKey(func.inCasts_[i]);
        sm->BindAddrWithVariableName(key, localName, "");
        
        // Register SSBO name for Gather/Scatter support
        if (func.inCasts_[i]->magic != 0) {
            sm->BindSSBOName(func.inCasts_[i]->magic, ssboName);
        }
    }

    // 2. Generate Ops
    for (auto &op : func.Operations()) {
        // We create a dummy subfunc for now as required by CodeGenOpCtx
        // Since we can't easily create a Function, we might need to hack CodeGenOpCtx or use a dummy pointer if possible.
        // But CodeGenOpCtx takes reference.
        // Let's use func as subfunc for now, assuming it won't break things for GLSL.
        CodeGenOpCtx opCtx(sm, func, func, op);
        CodeGenOpGLSL glslOp(opCtx);
        glslOp.Init(op); // Initialize operand info
        
        ss << "    " << glslOp.GenOpCode() << "\n";
    }

    // 3. Store Outputs
    for (size_t i = 0; i < func.outCasts_.size(); ++i) {
         std::string ssboName = "out" + std::to_string(i);
         
         // Find the local variable holding the result
         auto key = sm->CreateAllocKey(func.outCasts_[i]);
         
         // Only store if there is a local variable bound (e.g. not handled by Scatter)
         if (sm->IsVariableBound(key)) {
             std::string localName = sm->QueryVariableName(key);
             ss << "    " << ssboName << ".d[global_idx] = " << localName << ";\n";
         }
    }
    
    ss << "}\n";
    return ss.str();
}

} // namespace npu::tile_fwk
