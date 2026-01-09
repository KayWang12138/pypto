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
 * \file tune_sync_for_vf.cpp
 * \brief
 */

#include "passes/block_graph_pass/tune_sync_for_vf.h"
#include "passes/block_graph_pass/insert_sync.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "TuneSyncForVF"

namespace npu {
namespace tile_fwk {
bool TuneSyncForVF::NeedAdjustSetFlag(Function *subGraphFunc, Operation *vecTileOp0, Operation *vecTileOp1, Operation *setFlag) {
    PipeType pipeX = setFlag->syncQueue_.trigPipeId_;
    float tv = static_cast<float>(subGraphFunc->pipeEndTime[PipeType::PIPE_V]);
    float tx = static_cast<float>(subGraphFunc->pipeEndTime[pipeX]);
    float t0 = static_cast<float>(vecTileOp0->cycleStart);
    float t1 = static_cast<float>(vecTileOp0->cycleEnd);
    float t2 = static_cast<float>(vecTileOp1->cycleEnd);
    float ty = t0 + vfPrarm * vecTileOp0->GetLatency() + vfPrarm * vecTileOp1->GetLatency();
    Operation *tileOpZ = subGraphFunc->setWaitOpMap[vecTileOp0];
    float tb = static_cast<float>(tileOpZ->cycleStart);
    if (std::max(tv - t2 + ty, tx + std::max(0, (ty - std::max(t1, tb)))) < tv) {
        return true;
    }
    return false;
}

bool TuneSyncForVF::NeedAdjustWaitFlag(Function *subGraphFunc, Operation *vecTileOp0, Operation *vecTileOp1, Operation *waitFlag) {
    PipeType pipeX = waitFlag->syncQueue_.pipeId_;
    float tv = static_cast<float>(subGraphFunc->pipeEndTime[PipeType::PIPE_V]);
    float tx = static_cast<float>(subGraphFunc->pipeEndTime[pipeX]);
    float t0 = static_cast<float>(vecTileOp0->cycleStart);
    float t1 = static_cast<float>(vecTileOp0->cycleEnd);
    float t2 = static_cast<float>(vecTileOp1->cycleEnd);
    Operation *tileOpZ = subGraphFunc->waitSetOpMap[vecTileOp1];
    float tb = static_cast<float>(tileOpZ->cycleEnd);
    float ty = std::max(t0, tb) + vfPrarm * vecTileOp0->GetLatency() + vfPrarm * vecTileOp1->GetLatency();
    if (std::max(tv - t2 + ty, tx) < tv) {
        return true;
    }
    return false;
}

void TuneSyncForVF::AdjustSetWaitFlag(Function *subGraphFunc, std::vector<Operation *> &setFlagList, 
        std::vector<Operation *> &waitFlagList, size_t vecTileOp0Idx, size_t vecTileOp1Idx, int groupNum) {
    // 改变opList执行顺序
    // 先将setwaitflag删掉
    std::vector<size_t> setWaitIdx;
    for (size_t k = vecTileOp0Idx + 1; k < vecTileOp1Idx; k++) {
        setWaitIdx.emplace_back(k);
    }
    for (auto it = setWaitIdx.rbegin(); it != setWaitIdx.rend(); it++) {
        opList_.erase(opList_.begin() + *it);
    }
    // 删掉这些op后，vecTileOp1Idx = vecTileOp0Idx + 1, 在vecTileOp1Idx右侧将waitflag插入
    auto insertPos = opList_.begin() + vecTileOp0Idx + 2;
    opList_.insert(insertPos, waitFlagList.begin(), waitFlagList.end());
    // 在vecTileOp0Idx集合的左侧将setflag插入
    size_t mergedSize = mergedOps[groupNum].size();
    auto insertPos2 = opList_.begin() + vecTileOp0Idx - mergedSize;
    opList_.insert(insertPos2. setFlagList.begin(), setFlagList.end());

    // 更新各pipe上op的时间戳
    // TODO: 更新pipe_v的时间戳
    // auto &pipeVIssueQ = subGraphFunc->issueState[static_cast<int>(PipeSeq::AIV_V)];

}

void TuneSyncForVF::ChangeOpSeq(Function *subGraphFunc, bool isAIV1) {
    AIVCore coreType;
    if (!isAIV1) {
        coreType = AIVCore::AIV0;
    } else {
        coreType = AIVCore::AIV1;
    }

    std::vector<size_t> pipeVIdx;
    for (size_t i = 0; i < opList_.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opList_[i]->GetOpcode());
        if (opcfg.pipeIdStart_ == PipeType::PIPE_V && opList_[i]->GetAIVCore() == coreType) {
            pipeVIdx.emplace_back(i);
        }
    }

    if (pipeVIdx.size() <= 1) {
        return;
    }

    mergedOps.clear();
    for (size_t idx = 0; idx + 1 < pipeVIdx.size(); idx++) {
        size_t left = pipeVIdx[idx];
        size_t right = pipeVIdx[idx + 1];
        if (right == left + 1) {
            continue;
        }
        // 判断两个pipeV op间是否有SYNC_SRC或者SYNC_DST或者既不是SYNC_SRC也不是SYNC_DST
        std::vector<Operation *> setFlagList;
        std::vector<Operation *> waitFlagList;
        bool hasNonSetWaitOp = false;
        for (size_t k = left + 1; k < right; k++) {
            if (opList_[k]->GetOpcode() == Opcode::OP_SYNC_SRC) {
                setFlagList.emplace_back(opList_[k]);
            } else if (opList_[k]->GetOpcode() == Opcode::OP_SYNC_DST) {
                waitFlagList.emplace_back(opList_[k]);
            } else {
                hasNonSetWaitOp = true;
                break;
            }
        }
        // 两个pipeV的op间的op如果有一个既不是SYNC_SRC也不是SYNC_DST,则说明这两个pipeV op不能合并
        if (hasNonSetWaitOp || (setFlagList.empty() && waitFlagList.empty())) {
            continue
        }
        // 判断是否需要进行调整 （所有的SYNC_SRC和SYNC_DST中，只要有一个是有收益的，就进行融合）
        bool needAdjustSet = false;
        for (auto &setFlag : setFlagList) {
            if (NeedAdjustSetFlag(subGraphFunc, opList_[left], opList_[right], setFlag)) {
                needAdjustSet = true;
                break;
            }
        }
        if (!needAdjustSet) {
            bool needAdjustWait = false;
            for (auto &waitFlag : waitFlagList) {
                if (NeedAdjustWaitFlag(subGraphFunc, opList_[left], opList_[right], waitFlag)) {
                    needAdjustWait = true;
                    break;
                }
            }
            if (!needAdjustWait) {
                continue;
            }
        }
        // 此时vecTileop1和vecTileop2需要融合，先看vecTileop0是否已经在mergedOps中
        int groupNum = -1;
        for (size_t i = 0; i < mergedOps.size(); i++) {
            for (size_t j = 0; j < mergedOps[i].size(); j++) {
                if (mergedOps[i][j] == opList_[left]) {
                    groupNum = i;
                    break;
                }
            }
        }
        auto vecTileOp0 = opList_[left];
        auto vecTileOp1 = opList_[right];
        if (groupNum == -1) {
            std::vector<Operation> newOp = {vecTileOp0};
            mergedOps.emplace_back(newOp);
            groupNum = mergedOps.size() - 1;
        }
        // 进行调整
        AdjustSetWaitFlag(subGraphFunc, setFlagList, waitFlagList, left, right, groupNum);
        // 将vecTileop1添加到mergedOps中
        mergedOps[groupNum].emplace_back(vecTileOp1);
        // 由于移动，pipeVop的idx会发生变化，需要重新更新pipeVIdx
        pipeVIdx.clear();
        for (size_t i = 0; i < opList_.size(); i++) {
            auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opList_[i]->GetOpcode());
            if (opcfg.pipeIdStart_ == PipeType::PIPE_V && opList_[i]->GetAIVCore() == coreType) {
                pipeVIdx.emplace_back(i);
            }
        }
    }
}

Status TuneSyncForVF::RunOnFunction(Function &function) {
    for (auto &program : function.rootFunc_->programs_) {
        std::vector<Operation *> opList(program.second->Operations(false).DuplicatedOpList());
        opList_ = opList;
        // AIV0和AIV1各调整一次
        ChangeOpSeq(program.second, false);
        ChangeOpSeq(program.second, true);
        // 将调整后的oplist刷新到function中去
        program.second->ScheduleBy(opList_, true);
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu