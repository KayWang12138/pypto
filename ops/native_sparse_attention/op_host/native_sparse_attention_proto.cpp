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
 * \file flash_attention_score_proto.cpp
 * \brief
 */

#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "register/op_ct_impl_registry.h"
#include "register/op_ext_gentask_registry.h"
#include "utils/tile_fwk_op_utils.h"

using namespace ge;

namespace ops {
ge::graphStatus InferShapeNativeSparseAttention(gert::InferShapeContext *context)
{
    (void)context;
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeNativeSparseAttention(gert::InferDataTypeContext *context)
{
    (void)context;
    return GRAPH_SUCCESS;
}

IMPL_OP(NativeSparseAttention)
    .InferShape(InferShapeNativeSparseAttention)
    .InferDataType(InferDataTypeNativeSparseAttention);

ge::graphStatus GetOpspecificInfoNativeSparseAttention(const gert::OpCheckContext *context, ge::AscendString &result)
{
    return TileFwkOpUtils::CommonGenOpSpecificInfo(context, result);
}

ge::graphStatus CalcOpParamNativeSparseAttention(gert::ExeResGenerationContext *context)
{
    ge::AscendString name = "nsa aicpu kfc server";
    ge::AscendString reuse_key = "nsa kfc_stream";
    return TileFwkOpUtils::CommonCalcOpParam(context, name, reuse_key);
}

IMPL_OP_CT(NativeSparseAttention)
    .GetOpSpecificInfo(GetOpspecificInfoNativeSparseAttention)
    .CalcOpParam(CalcOpParamNativeSparseAttention);

ge::graphStatus GenerateTaskNativeSparseAttention(const ge::Node &node, ge::RunContext &run_context,
                                                  std::vector<domi::TaskDef> &task_defs)
{
    return TileFwkOpUtils::CommonGenerateTask(node, run_context, task_defs);
}

REGISTER_NODE_EXT_GENTASK("NativeSparseAttention", GenerateTaskNativeSparseAttention);
} // namespace ops
