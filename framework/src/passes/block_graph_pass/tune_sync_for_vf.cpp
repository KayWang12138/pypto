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
 * \file tune_sync_for_vf.cpp
 * \brief
 */

#include "passes/block_graph_pass/tune_sync_for_vf.h"
#include "passes/pass_log/pass_log.h"

// #define MODULE_NAME "TuneSyncForVF"

namespace npu {
namespace tile_fwk {
Status TuneSyncForVF::RunOnFunction(Function &function) {
    (void) function;
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu