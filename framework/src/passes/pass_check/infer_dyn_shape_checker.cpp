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
 * \file infer_dyn_shape_checker.h
 * \brief
 */

#include "infer_dyn_shape_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "InferDynShape"

namespace npu {
namespace tile_fwk {
Status InferDynShapeChecker::DoPostCheck(Function &function) {
    for (auto& op : function.Operations()) {
        if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode())) {
            const std::shared_ptr<OpAttribute> &attr = op.GetOpAttribute();
            if (attr == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Copy In attr is null.");
                return FAILED;
            }
            std::shared_ptr<CopyOpAttribute> copyAttr = std::static_pointer_cast<CopyOpAttribute>(attr);
            if (copyAttr->GetToDynValidShape().empty()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Op %s[%d] has no dyn to shape attr.",
                    op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
        }
        for (auto opOut : op.GetOOperands()) {
            if (opOut->GetDynValidShape().empty()) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Op %s[%d] output [%d] has no dynamic valid shape.",
                    op.GetOpcodeStr().c_str(), op.GetOpMagic(), opOut->GetMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu