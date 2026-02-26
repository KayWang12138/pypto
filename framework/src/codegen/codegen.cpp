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
 * \file codegen.cpp
 * \brief
 */
#include <chrono>

#include "codegen.h"
#include "codegen_factory.h"

namespace npu::tile_fwk {
void CodeGen::GenCode(Function &topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) {
    auto start = std::chrono::high_resolution_clock::now();
    ASSERT(topFunc.rootFunc_ != nullptr) << "rootFunc can not be nullptr";

    auto cg = CodeGenFactory::GetCodeGenCCE(ctx_);
    cg->GenCode(topFunc, invokeParaOffset);
    auto end = std::chrono::high_resolution_clock::now();
    auto cost = std::chrono::duration<double, std::milli>(end - start).count();
    CODEGEN_LOGE("hjhj CodeGen TopFunc %s, hash %s, cost: %f ms", topFunc.GetMagicName().c_str(),
        topFunc.GetFunctionHash().c_str(), cost);
}

} // namespace npu::tile_fwk
