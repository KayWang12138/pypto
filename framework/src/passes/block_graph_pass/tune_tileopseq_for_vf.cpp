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
void TuneTileOpSeqForVF::ChangeOpSeq(std::vector<Operation *> &opList, PipeSync &ps, bool isAIV1) {
    AIVCore coreType;
    if (!isAIV1) {
        coreType = AIVCore::AIV0;
    } else {
        coreType = AIVCore::AIV1;
    }

    std::vector<size_t> pipeVIdx;
    for (size_t i = 0; i < opList.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opList[i]->GetOpcode());
        if (opcfg.pipeIdStart_ == PipeType::PIPE_V && opList[i]->GetAIVCore() == coreType) {
            pipeVIdx.emplace_back(i);
        }
    }

    if (pipeVIdx.size() <= 1) {
        return;
    }

    for (size_t idx = 0; idx + 1 < pipeVIdx.size(); idx++) {
        size_t left = pipeVIdx[idx];
        size_t right = pipeVIdx[idx + 1];
        if (right == left + 1) {
            continue;
        }
        bool hasDep = false;
        for (size_t k = left + 1; k < right; k++) {
            if (ps.HasDataDependency(*opList[left], *opList[k], left, k) || ps.HasDataDependency(*opList[k], *opList[right], k, right)) {
                hasDep = true;
                break;
            }
        }
        // 存在依赖，不能融合
        if (hasDep) {
            continue;
        }
        // 不存在依赖，可以融合
        std::vector<Operation *> toMove;
        std::vector<size_t> toMoveIdx;
        for (size_t k = left + 1; k < right; k++) {
            toMove.emplace_back(opList[k]);
            toMoveIdx.emplace_back(k);
        }
        for (auto it = toMoveIdx.rbegin(); it != toMoveIdx.rend(); it++) {
            opList.erase(opList.begin() + *it);
        }
        // 删掉这些op后，right = left + 1, 在right右侧将删掉的op重新插入，即在left + 2的位置开始插入
        auto insertPos = opList.begin() + left + 2;
        opList.insert(insertPos, toMove.begin(), toMove.end());
        // 由于移动，pipeVop的idx会发生变化，需要重新更新pipeVIdx
        pipeVIdx.clear();
        for (size_t i = 0; i < opList.size(); i++) {
            auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opList[i]->GetOpcode());
            if (opcfg.pipeIdStart_ == PipeType::PIPE_V && opList[i]->GetAIVCore() == coreType) {
                pipeVIdx.emplace_back(i);
            }
        }
    }
}

Status TuneTileOpSeqForVF::RunOnFunction(Function &function) {
    size_t funcId = 0;
    for (auto &program : function.rootFunc_->programs_) {
        std::vector<Operation *> opList(program.second->Operations(false).DuplicatedOpList());
        PipeSync ps;
        APASS_LOG_DEBUG_F(Elements::Function, "=======================function %d ======================", funcId);
        for (const auto &op : opList) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Input Operation %d %s", op->GetOpMagic(), op->GetOpcodeStr().c_str());
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
        // AIV0和AIV1各调整一次
        ChangeOpSeq(opList, ps, false);
        ChangeOpSeq(opList, ps, true);
        // 将调整后的oplist刷新到function中去
        program.second->ScheduleBy(opList, true);
        APASS_LOG_DEBUG_F(Elements::Function, "---------------------------------------------------");
        for (const auto &op : opList) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Output Operation %d %s", op->GetOpMagic(), op->GetOpcodeStr().c_str());
        }
        funcId++;

        // TODO 增加拓扑逻辑校验
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu