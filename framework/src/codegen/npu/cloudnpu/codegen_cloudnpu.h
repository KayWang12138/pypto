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
 * \file codegen.h
 * \brief
 */

#ifndef CODEGEN_CLOUDNPU_H
#define CODEGEN_CLOUDNPU_H

#include <string>
#include <unordered_set>
#include <utility>
#include <mutex>
#include <vector>
#include <chrono>
#include <thread>

#include "tilefwk/platform.h"
#include "interface/operation/operation.h"
#include "codegen/codegen_cce.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/codegen_common.h"
#include "interface/configs/config_manager.h"
#include "codegen/npu/codegen_npu.h"

namespace npu::tile_fwk {

class CodeGenCloudNPU : public CodeGenNPU {
public:
    explicit CodeGenCloudNPU(const CodeGenCtx& cgCtx) : CodeGenNPU(cgCtx) {};
    ~CodeGenCloudNPU() override = default;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_CLOUDNPU_H
