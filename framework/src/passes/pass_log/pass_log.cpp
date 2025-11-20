/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pass_log.cpp
 * \brief
 */

#include <iostream>
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {

    std::string GetFormatBacktrace(const Operation& op) {
        auto location = op.GetLocation();
        if (!location) {
            return "";
        }
        
        std::ostringstream oss;
        oss << "[FuncMagic:" << op.BelongTo()->GetFuncMagic() << "]" << "[OpMagic:" << op.opmagic << "]" << "[Backtrace]:" << location->SourceLocation::GetBacktrace() << ".";
        return oss.str();
    }

}