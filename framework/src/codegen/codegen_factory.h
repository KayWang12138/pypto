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
 * \file codegen.h
 * \brief
 */

#ifndef CODEGEN_FACTORY_H
#define CODEGEN_FACTORY_H

#include <unordered_set>
#include <utility>

#include "codegen_cce.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {
class CodeGenFactory {
public:
    static std::shared_ptr<CodeGenCCE> GetCodeGenCCE(const CodeGenCtx &ctx) {
        auto platform = config::GetDevicePlatform();
        if (platform >= DPlatform::ASCEND_910B1 && platform <= DPlatform::ASCEND_910B4) {
            return std::make_shared<CodeGenCloudNPU>(ctx);
        }
        ASSERT(false) << "can not support this platform";
        return nullptr;
    }
};

} // namespace npu::tile_fwk

#endif // CODEGEN_FACTORY_H
