/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CODEGEN_GLSL_H
#define CODEGEN_GLSL_H

#include "codegen/codegen.h"
#include "codegen_op_glsl.h"

namespace npu::tile_fwk {

class CodeGenGLSL {
public:
    explicit CodeGenGLSL(CodeGenCtx cctx);
    virtual ~CodeGenGLSL() = default;

    virtual void GenCode(Function &topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset);

private:
    CodeGenCtx ctx;
    
    std::string GenerateHeader();
    std::string GenerateInputsOutputs(Function &func);
    std::string GenerateMain(Function &func);
    std::string GenerateHelperFunctions();
    
    std::unordered_set<Opcode> usedOps;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_GLSL_H
