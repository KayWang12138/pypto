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
 * \file add_alloc_checker.cpp
 * \brief
 */

#include "add_alloc_checker.h"

namespace npu {
namespace tile_fwk {
Status AddAllocChecker::DoPreCheck(Function &function) {
    for (auto &[psgID, subFunc] : function.rootFunc_->programs_) {
        (void)psgID;
        if (subFunc->Operations().size() == 0) {
            return SUCCESS;
        }
        auto subgraphID = subFunc->Operations().begin()->GetSubgraphID();
        for (auto &op : subFunc->Operations()) {
            if (op.GetSubgraphID() == NOT_IN_SUBGRAPH) {
                return FAILED;
            }
            if (op.GetSubgraphID() != subgraphID) {
                return FAILED;
            }
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu