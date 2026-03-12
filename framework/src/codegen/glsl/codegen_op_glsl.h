/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CODEGEN_OP_GLSL_H
#define CODEGEN_OP_GLSL_H

#include "codegen/codegen_op.h"
#include <map>
#include <string>
#include <vector>

namespace npu::tile_fwk {

struct OpTemplate {
    std::string name;
    std::string body; // Template body with placeholders like ${out}, ${in0}, ${in1}
    std::string funcDef; // Helper function definition if needed
};

class CodeGenOpGLSL : public CodeGenOp {
public:
    explicit CodeGenOpGLSL(const CodeGenOpCtx &ctx);
    ~CodeGenOpGLSL() override = default;

    void Init(const Operation &ops) override;
    std::string GenOpCode() const override;

    // Static registry for templates
    static const std::map<Opcode, OpTemplate>& GetOpTemplates();

private:
    std::string GenBinaryOp() const;
    std::string GenUnaryOp() const;
    std::string GenCustomOp() const; // For more complex ops
    std::string GenGatherOp() const;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_OP_GLSL_H
