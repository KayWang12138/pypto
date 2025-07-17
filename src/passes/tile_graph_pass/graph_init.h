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
 * \file graph_init.h
 * \brief
 */

#ifndef GRAPI_INIT_PASS_H
#define GRAPI_INIT_PASS_H
#include "interface/operation/opcode.h"
#include "common/data_type.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
class GraphInitPass : public Pass {
public:
    GraphInitPass() : Pass("GraphInitPass") {}
    ~GraphInitPass() override = default;

private:
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
};
} // namespace npu::tile_fwk
#endif  // GRAPI_INIT_PASS_H