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
    (void)context;
    (void)result;
    return GRAPH_SUCCESS;
}

ge::graphStatus CalcOpParamNativeSparseAttention(gert::ExeResGenerationContext *context)
{
    (void)context;
    return GRAPH_SUCCESS;
}

IMPL_OP_CT(NativeSparseAttention)
    .GetOpSpecificInfo(GetOpspecificInfoNativeSparseAttention)
    .CalcOpParam(CalcOpParamNativeSparseAttention);

ge::graphStatus GenerateTaskNativeSparseAttention(const ge::Node &node, ge::RunContext &context,
                                                  std::vector<domi::TaskDef> &tasks)
{
    (void)node;
    (void)context;
    (void)tasks;
    return GRAPH_SUCCESS;
}

REGISTER_NODE_EXT_GENTASK("NativeSparseAttention", GenerateTaskNativeSparseAttention);
} // namespace ops
