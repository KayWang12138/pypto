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
 * \file tune_tileopseq_for_vf.h
 * \brief
 */

#ifndef TUNE_TILEOPSEQ_FOR_VF_H
#define TUNE_TILEOPSEQ_FOR_VF_H

#include "passes/pass_interface/pass.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/block_graph_pass/insert_sync.h"

namespace npu::tile_fwk {
class TuneTileOpSeqForVF : public Pass {
public:
    TuneTileOpSeqForVF() : Pass("TuneTileOpSeqForVF") {}
    ~TuneTileOpSeqForVF() override = default;

    Status RunOnFunction(Function &function) override;

private:
    void ChangeOpSeq(std::vector<Operation *> &opList, PipeSync &ps, bool isAIV1);
    void PushBackIdx(size_t idx, std::vector<size_t> &vec);
};
}
#endif // TUNE_TILEOPSEQ_FOR_VF_H