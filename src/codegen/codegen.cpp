/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file codegen.cpp
 * \brief
 */

#include "codegen.h"
#include "codegen_factory.h"

namespace npu::tile_fwk {
void CodeGen::GenCode(
    Function &topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) {
    auto cg = CodeGenFactory::GetCodeGenCCE(ctx_);
    cg->GenCode(topFunc, invokeParaOffset);
}

void CodeGen::GenCode(
    const std::string &jsonPath, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) {
    auto cg = CodeGenFactory::GetCodeGenCCE(ctx_);
    cg->GenCode(jsonPath, invokeParaOffset);
}

} // namespace npu::tile_fwk
