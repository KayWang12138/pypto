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
 * \file infer_param_index.h
 * \brief
 */

#ifndef INFER_PARAM_INDEX_PASS_H_
#define INFER_PARAM_INDEX_PASS_H_
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
namespace npu::tile_fwk {
class InferParamIndexPass : public Pass {
public:
    InferParamIndexPass() : Pass("InferParamIndexPass") {}
    ~InferParamIndexPass() override {}
    Status RunOnFunction(Function &function) override;
    std::string DumpParamIndex(const std::map<std::string, DynParamInfo>& dynParamTable);
    std::mutex mtxInfer;
};
}
#endif
