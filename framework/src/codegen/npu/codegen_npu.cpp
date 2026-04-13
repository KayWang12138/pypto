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
#include "codegen_npu.h"

#include <cstring>
#include <error.h>
#include <fstream>

#include "codegen/utils/parallel_execute.h"
#include "codegen_op_npu.h"
#include "codegen/stmt_mgr/codegen_for_block.h"
#include "interface/utils/file_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/op_info_manager.h"
#include "interface/operation/distributed/distributed_common.h"
#include "tilefwk/tilefwk.h"
#include "securec.h"

namespace npu::tile_fwk {

void FloatSpecValMgr::UpdateByOp(const Operation& op)
{
    std::vector<Element> eles;
    if (op.HasAttr(OpAttributeKey::scalar)) {
        eles.emplace_back(op.GetElementAttribute(OpAttributeKey::scalar));
    }
    if (op.HasAttr(OpAttributeKey::vectorScalar)) {
        auto vecScalars = op.GetVectorElementAttribute(OpAttributeKey::vectorScalar);
        eles.insert(eles.end(), vecScalars.begin(), vecScalars.end());
    }

    if (eles.empty()) {
        return;
    }

    for (const auto& e : eles) {
        if (e.GetDataType() == DataType::DT_FP16 || e.GetDataType() == DataType::DT_FP32 ||
            e.GetDataType() == DataType::DT_BF16) {
            double value = e.Cast<float>();
            if (std::isinf(value) || std::isnan(value)) {
                floatSpecVals_.insert({e.GetDataType(), value});
            }
        }
    }
}

void FloatSpecValMgr::PrintFloatSpecVal(std::ostringstream& oss)
{
    // print statement like: union {float f; uint32_t u;} float_inf = {.u = 0x7F800000};
    for (const auto& fs : floatSpecVals_) {
        std::string dtypeCCE = DataType2CCEStr(fs.dtype);
        oss << "union "
            << "{" << dtypeCCE << " f; "
            << "uint32_t u;} " << fs.GetFsVarName() << " = {.u = " << fs.GetFsValueStr() << "};\n";
    }
}


} // namespace npu::tile_fwk
