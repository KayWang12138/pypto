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
 * \file tune_tileopseq_for_vf.cpp
 * \brief
 */

#include "passes/block_graph_pass/tune_tileopseq_for_vf.h"
#include "passes/block_graph_pass/insert_sync.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "TuneTileOpSeqForVF"

namespace npu {
namespace tile_fwk {
void TuneTileOpSeqForVF::PushBackIdx(size_t idx, std::vector<size_t> &vec) {
    auto it = std::find(vec.begin(), vec.end(). idx);
    if (it == vec.end()) {
        vec.emplace_back(idx);
    }
}

void TuneTileOpSeqForVF::ChangeOpSeq(std::vector<Operation *> &opList, std::vector<size_t> &tunedOpList, std::vector<size_t> &pipeVIdx, PipeSync &ps, bool isAIV1) {
    AIVCore coreType;
    if (!isAIV1) {
        coreType = AIVCore::AIV0;
    } else {
        coreType = AIVCore::AIV1;
    }
    for (size_t i = 0; i < opList.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opList[i]->GetOpcode());
        if (opcfg.pipeIdStart_ == PipeType::PIPE_V && opList[i]->GetAIVCore() == coreType) {
            pipeVIdx.emplace_back(i);
        }
    }
    if (pipeVIdx.size() <= 1) {
        for (size_t i = 0; i < opList.size(); i++) {
            PushBackIdx(i, tunedOpList);
        }
    } else {
        // 将第一个pipeVop前面的op加入到tunedOpList中
        if (pipeVIdx[0] > 0) {
            for (size_t i = 0; i < pipeVIdx[0]; i++) {
                PushBackIdx(i, tunedOpList);
            }
        }
        for (size_t idx = 0; idx + 1 < pipeVIdx.size(); idx++) {
            size_t left = pipeVIdx[idx];
            size_t right = pipeVIdx[idx + 1];
            std::vector<size_t> nonPipeVOp;
            bool hasDep = false;
            for (size_t k = left + 1; k < right; k++) {
                if (ps.HasDataDependency(*opList[left], *opList[k], left, k) || ps.HasDataDependency(*opList[k], *opList[right], k, right)) {
                    hasDep = true;
                }
            }
            // 存在依赖，不能融合
            if (hasDep) {
                for (size_t i = left; i <= right; i++) {
                    PushBackIdx(i, tunedOpList);
                }
            // 不存在依赖，可以融合
            } else {
                PushBackIdx(left, tunedOpList);
                PushBackIdx(right, tunedOpList);
                for (size_t i = left + 1; i < right; i++) {
                    PushBackIdx(i, tunedOpList);
                }
            }
        }
        // 将最后一个pipeVop后面的op加入到tunedOpList中
        if (pipeVIdx[pipeVIdx.size() - 1] < opList.size() - 1) {
            for (size_t i = pipeVIdx[pipeVIdx.size() - 1] + 1; i < opList.size(); i++) {
                PushBackIdx(i, tunedOpList);
            }
        }
    }
}

Status TuneTileOpSeqForVF::RunOnFunction(Function &function) {
    for (auto &program : function.rootFunc_->programs_) {
        std::vector<Operation *> oriOpList(program.second->Operations(false).DuplicatedOpList());
        std::vector<size_t> tunedOpList0;
        std::vector<size_t> tunedOpList1;
        std::vector<size_t> pipeVIdx0;
        std::vector<size_t> pipeVIdx1;
        PipeSync ps;
        for (const auto &op : oriOpList) {
            ps.BuildTensorRangeMap(op);
            auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op->GetOpcode());
            if (opcfg.pipeIdStart_ != PipeType::PIPE_V) {
                continue;
            }
            // 假定：pipe_V的op的AIV类型只能是AIV0或AIV1
            if (op->GetAIVCore() != AIVCore::AIV0 && op->GetAIVCore() != AIVCore::AIV1) {
                APASS_LOG_ERROR_F(Elements::Operation, "Pipe_V op %d %s AIV type is neither AIV0 nor AIV1, RunOnFunction failed.", op->GetOpMagic(), op->GetOpcodeStr().c_str());
                return FAILED;
            }
        }
        ChangeOpSeq(oriOpList, tunedOpList0, pipeVIdx0, ps, false);
        std::vector<Operation *> flagOpList;
        for (size_t i = 0; i < tunedOpList0.size(); i++) {
            flagOpList.emplace_back(oriOpList[tunedOpList0[i]]);
        }
        ChangeOpSeq(flagOpList, tunedOpList1, pipeVIdx1, ps, true);
        // 将调整后的oplist刷新到function中去
        std::vector<Operation *> resOpList;
        for (size_t i = 0; i < tunedOpList1.size(); i++) {
            resOpList.emplace_back(flagOpList[tunedOpList1[i]]);
        }
        program.second->ScheduleBy(resOpList, true);

        // TODO 增加拓扑逻辑校验
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu