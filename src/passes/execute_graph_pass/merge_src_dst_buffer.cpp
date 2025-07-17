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
 * \file merge_src_dst_buffer.cpp
 * \brief
 */

#include "merge_src_dst_buffer.h"

namespace npu::tile_fwk {

void SrcDstBufferMergePVC2::Init(const std::vector<Operation *> &opList) {
    subGrpahID_ = opList.front()->GetSubgraphID();
    for (auto& op : opList) {
        for (auto& output : op->GetOOperands()) {
            assert(subGrpahID_ == op->GetSubgraphID());
            if (output->memorymap[subGrpahID_].memId == -1) {
                output->memorymap[subGrpahID_].memId = output->GetMagic();
            }
            for (auto& consumer : output->GetConsumers()) {
                tensorConsumers_[output->memorymap[subGrpahID_].memId].insert(consumer->GetOpMagic());
                if (tensorMaxSize_.find(output->memorymap[subGrpahID_].memId) == tensorMaxSize_.end()) {
                    tensorMaxSize_[output->memorymap[subGrpahID_].memId] = output->GetDataSize();
                } else {
                    tensorMaxSize_[output->memorymap[subGrpahID_].memId] = std::max(tensorMaxSize_[output->memorymap[subGrpahID_].memId], output->GetDataSize());
                }
            }
        }
    }
}

void SrcDstBufferMergePVC2::Run(Function &func) {
    for (auto &subProgram : func.rootFunc_->programs_) {
        ALOG_INFO_F("merge src dst for program id : [%lu]", subProgram.first);
        auto opList = subProgram.second->Operations().DuplicatedOpList();
        Init(opList);
        auto oriOps(opList);
        const std::set<Opcode> ignoreOps = {Opcode::OP_UB_COPY_IN, Opcode::OP_UB_COPY_OUT, Opcode::OP_UB_ALLOC,
            Opcode::OP_L0C_COPY_OUT, Opcode::OP_ROWMAX, Opcode::OP_ROWEXPSUM, Opcode::OP_REMOTE_GATHER,
            Opcode::OP_ROWEXPMAX, Opcode::OP_TRANSPOSE_VNCHWCONV, Opcode::OP_COPY_IN, Opcode::OP_COPY_OUT,
            Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWSUM_SINGLE, Opcode::OP_MAX_POOL, Opcode::OP_COPY_UB_TO_UB};
        std::unordered_map<int, std::shared_ptr<LogicalTensor>> replacedTensors;
        for (size_t i = 0; i < oriOps.size(); i++) {
            ALOG_DEBUG_F("Try reuse op [%d] input by out tensor", oriOps[i]->GetOpMagic());
            if (ignoreOps.count(oriOps[i]->GetOpcode()) != 0) {
                continue;
            }
            if (func.GetRootFunction()->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP_PATH, GraphType::ROOT_GRAPH)) {
                if (oriOps[i]->GetOpcode() == Opcode::OP_PAIRMAX || oriOps[i]->GetOpcode() == Opcode::OP_PAIRSUM) {
                    continue;
                }
            }
            if (oriOps[i]->HasAttr(OpAttributeKey::isCube) &&
                oriOps[i]->GetBoolAttribute(OpAttributeKey::isCube)) {
                continue;
            }
            int inIdx = 0;
            bool hasInplaced = false;
            if (oriOps[i]->HasAttr(OpAttributeKey::inplaceIdx)) {
                inIdx = oriOps[i]->GetIntAttribute(OpAttributeKey::inplaceIdx);
                if (oriOps[i]->GetIOperands().size() > static_cast<size_t>(inIdx)) {
                    auto in = opList[i]->GetIOperands()[inIdx];
                    auto out = opList[i]->GetOOperands()[0];
                    if (in->memorymap[subGrpahID_].memId == out->memorymap[subGrpahID_].memId) {
                        continue;
                    }
                    out->memorymap[subGrpahID_].memId = in->memorymap[subGrpahID_].memId;
                    tensorConsumers_[in->memorymap[subGrpahID_].memId].insert(tensorConsumers_[out->memorymap[subGrpahID_].memId].begin(),
                        tensorConsumers_[out->memorymap[subGrpahID_].memId].end());
                    replacedTensors[out->memorymap[subGrpahID_].memId] = in;
                    hasInplaced = true;
                }
            }
            if (hasInplaced) {
                continue;
            }
            bool findReplaced{false};
            for (auto in : oriOps[i]->GetIOperands()) {
                if (in != nullptr && CanSrcDstReuse(oriOps, i, in, true)) {
                    // 当前输出复用输入
                    auto out = opList[i]->GetOOperands()[0];
                    auto inTensorMagic = in->memorymap[subGrpahID_].memId;
                    auto outTensorMagic = out->memorymap[subGrpahID_].memId;
                    if (inTensorMagic == outTensorMagic) {
                        continue;
                    }
                    ALOG_DEBUG_F("Op [%d] %s reuse src [%d] buffer", oriOps[i]->GetOpMagic(), oriOps[i]->GetOpcodeStr().c_str(), inIdx);
                    ALOG_DEBUG_F("    set out tensor %d reuse src tensor %d", out->GetMagic(), in->GetMagic());
                    out->memorymap[subGrpahID_].memId = in->memorymap[subGrpahID_].memId;
                    if (tensorConsumers_[outTensorMagic].size() > tensorConsumers_[inTensorMagic].size()) {
                        tensorConsumers_[inTensorMagic] = tensorConsumers_[outTensorMagic];
                    }
                    replacedTensors[outTensorMagic] = in;
                    findReplaced = true;
                    break;
                }
                inIdx++;
            }
            if (!findReplaced) {
                for (auto& out : opList[i]->GetOOperands()) {
                    auto outTensorMagic = out->memorymap[subGrpahID_].memId;
                    ALOG_DEBUG_F("Op %d out tensor magic: %d", opList[i]->GetOpMagic(), outTensorMagic);
                    if (replacedTensors.find(outTensorMagic) != replacedTensors.end()) {
                        ALOG_DEBUG_F("Find tensor: %d  replaced by tensor: %d", outTensorMagic,
                                     replacedTensors[outTensorMagic]->memorymap[subGrpahID_].memId);
                        out->memorymap[subGrpahID_].memId = replacedTensors[outTensorMagic]->memorymap[subGrpahID_].memId;
                    }
                }
            }
        }
    }
}

bool SrcDstBufferMergePVC2::CanSrcDstReuse(const std::vector<Operation *> &opList, size_t idx,
                                       std::shared_ptr<LogicalTensor> ioperand, bool strict) {
    if (opList[idx]->GetOOperands().size() == 0) {
        return false;
    }
    if (opList[idx]->GetOpcode() == Opcode::OP_SCATTER_ELEMENT) {
        if (ioperand == opList[idx]->GetIOperands()[0]) {
            return true;
        }
    }
    auto outOperand = opList[idx]->GetOOperands()[0];
    ALOG_DEBUG_F("try reuse src %d dst %d", ioperand->GetMagic(), outOperand->GetMagic());
    if (outOperand->GetMemoryTypeOriginal() != ioperand->GetMemoryTypeOriginal()) {
        ALOG_DEBUG_F("memtype is not same");
        return false;
    }
    // tile shape 必须一样
    if (tensorMaxSize_[outOperand->memorymap[subGrpahID_].memId] != 
        tensorMaxSize_[ioperand->memorymap[subGrpahID_].memId]) {
        ALOG_DEBUG_F("datasize is not same");
        return false;
    }
    if (strict && outOperand->Datatype() != ioperand->Datatype()) {
        ALOG_DEBUG_F("daatype is not same");
        return false;
    }
    for (auto consumer : outOperand->GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
            for (auto assembleOutTensor : consumer->GetOOperands()) {
                if (assembleOutTensor->memorymap[subGrpahID_].memId == outOperand->memorymap[subGrpahID_].memId) {
                    ALOG_DEBUG_F("assemble cannot be reused.");
                    return false;
                }
            }
        }
    }
    // 确保复用UB buffer后不会被覆写
    auto iter = tensorConsumers_.find(ioperand->memorymap[opList[idx]->GetSubgraphID()].memId);
    if (iter != tensorConsumers_.end() && iter->second.size() > 1) {
        ALOG_DEBUG_F("has more 1 output");
        return false;
    }
    return true;
}
} // namespace npu::tile_fwk