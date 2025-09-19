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

void SrcDstBufferMergeImpl::ResetMemoryMap(Function &function) const {
    /* 初始化所有tensor的memorymap */
    for (auto &op : function.Operations()) {
        for (auto &input : op.GetIOperands()) {
            input->memorymap.clear();
        }
        for (auto &output : op.GetOOperands()) {
            output->memorymap.clear();
        }
    }
}

void SrcDstBufferMergeImpl::InitializeTensorMemorymap(Operation &op) const {
    for (auto &input : op.GetIOperands()) {
        TileRange range;
        range.memId = input->tensor->GetRawMagic();
        input->memorymap[BLOCK_GRAPH_DEFAULT_COLOR] = range;
    }
    for (auto &output : op.GetOOperands()) {
        TileRange range;
        range.memId = output->tensor->GetRawMagic();
        output->memorymap[BLOCK_GRAPH_DEFAULT_COLOR] = range;
    }
}

void SrcDstBufferMergeImpl::InitTensorMaxSize(const LogicalTensorPtr &output) {
    for (auto &consumer : output->GetConsumers()) {
        tensorConsumers_[output->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId].insert(consumer->GetOpMagic());
        if (tensorMaxSize_.find(output->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId) == tensorMaxSize_.end()) {
            tensorMaxSize_[output->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId] = output->GetDataSize();
            continue;
        }
        tensorMaxSize_[output->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId] =
            std::max(tensorMaxSize_[output->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId], output->GetDataSize());
    }
}

Status SrcDstBufferMergeImpl::CheckOpValid(const Operation *op, int opId) {
    if (op == nullptr) {
        ALOG_ERROR_F("Op:%d is null", opId);
        return FAILED;
    }
    return SUCCESS;
}

void SrcDstBufferMergeImpl::InitOpOutput(const Operation *op) {
    int outId = 0;
    for (auto &output : op->GetOOperands()) {
        if (output == nullptr) {
            ALOG_DEBUG_F("Op:%s, magic:%d, output:%d is null",
                op->GetOpcodeStr().c_str(), op->GetOpMagic(), outId);
            ++outId;
            continue;
        }
        if (output->memorymap.find(BLOCK_GRAPH_DEFAULT_COLOR) == output->memorymap.end()) {
            ALOG_DEBUG_F("Op:%s, magic:%d, output id:%d does not have memorymap",
                op->GetOpcodeStr().c_str(), op->GetOpMagic(), outId);
            ++outId;
            continue;
        }
        if (output->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId == -1) {
            output->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId = output->GetMagic();
        }
        InitTensorMaxSize(output);
        ++outId;
    }
}

Status SrcDstBufferMergeImpl::Init(const std::vector<Operation *> &opList) {
    if (opList.empty()) {
        ALOG_ERROR_F("OpList empty");
        return FAILED;
    }
    if (opList.front() == nullptr) {
        ALOG_ERROR_F("First op is null");
        return FAILED;
    }
    
    int opId = 0;
    for (auto &op : opList) {
        if (CheckOpValid(op, opId) != SUCCESS) {
            ALOG_ERROR_F("CheckOpValid failed");
            return FAILED;
        }
        InitializeTensorMemorymap(*op);
        InitOpOutput(op);
        ++opId;
    }

    return SUCCESS;
}

bool SrcDstBufferMergeImpl::CheckIgnoreScene(const Operation *oriOps) {
    /* use opcode is unfavorable for reading and modification, maybe use opcalctype */
    const std::set<Opcode> ignoreOps = {Opcode::OP_UB_COPY_IN, Opcode::OP_UB_COPY_OUT, Opcode::OP_UB_ALLOC,
        Opcode::OP_L0C_COPY_OUT, Opcode::OP_ROWMAX, Opcode::OP_ROWEXPSUM, Opcode::OP_REMOTE_GATHER,
        Opcode::OP_ROWEXPMAX, Opcode::OP_TRANSPOSE_VNCHWCONV, Opcode::OP_COPY_IN, Opcode::OP_COPY_OUT,
        Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWSUM_SINGLE, Opcode::OP_MAX_POOL, Opcode::OP_COPY_UB_TO_UB,
        Opcode::OP_PAIRMAX,  Opcode::OP_PAIRMIN, Opcode::OP_PAIRSUM, Opcode::OP_ROWMIN_SINGLE};

    if (ignoreOps.count(oriOps->GetOpcode()) != 0) {
        return true;
    }

    if (oriOps->HasAttr(OpAttributeKey::isCube) &&
        oriOps->GetBoolAttribute(OpAttributeKey::isCube)) {
        return true;
    }
    for (auto &output : oriOps->GetOOperands()) {
        if (output == nullptr) {
            return true;
        }
        if (output->memorymap.find(BLOCK_GRAPH_DEFAULT_COLOR) == output->memorymap.end()) {
            return true;
        }
    }
    return false;
}

std::pair<bool, Status> SrcDstBufferMergeImpl::CheckHasInplaced(const Operation *oriOps, const Operation *ops,
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> &replacedTensors, int &inIdx) {
    if (oriOps->HasAttr(OpAttributeKey::inplaceIdx)) {
        inIdx = oriOps->GetIntAttribute(OpAttributeKey::inplaceIdx);
        if (oriOps->GetIOperands().size() <= static_cast<size_t>(inIdx) ||
            oriOps->GetOOperands().size() <= static_cast<size_t>(0)) {
            ALOG_ERROR_F("Operands size error, in:%d, out:%d, inIdx:%d",
                oriOps->GetIOperands().size(), oriOps->GetOOperands().size(), inIdx);
            return std::make_pair(false, FAILED);
        }

        auto in = ops->GetIOperands()[inIdx];
        auto out = ops->GetOOperands()[0];
        if (in->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId == out->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId) {
            return std::make_pair(true, SUCCESS);
        }
        out->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId = in->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
        tensorConsumers_[in->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId].insert(
            tensorConsumers_[out->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId].begin(),
            tensorConsumers_[out->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId].end());
        replacedTensors[out->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId] = in;
        return std::make_pair(true, SUCCESS);
    }
    return std::make_pair(false, SUCCESS);
}

bool SrcDstBufferMergeImpl::FindReplaced(const Operation *oriOps, const Operation *ops,
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> &replacedTensors, int &inIdx) {
    for (auto in : oriOps->GetIOperands()) {
        if (in != nullptr && CanSrcDstReuse(oriOps, in, true)) {
            // 当前输出复用输入
            auto out = ops->GetOOperands()[0];
            auto inTensorMagic = in->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
            auto outTensorMagic = out->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
            if (inTensorMagic == outTensorMagic) {
                continue;
            }
            ALOG_DEBUG_F("Op [%d] %s reuse src [%d] buffer",
                oriOps->GetOpMagic(), oriOps->GetOpcodeStr().c_str(), inIdx);
            ALOG_DEBUG_F("Set out tensor %d reuse src tensor %d", out->GetMagic(), in->GetMagic());
            out->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId = in->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
            if (tensorConsumers_[outTensorMagic].size() > tensorConsumers_[inTensorMagic].size()) {
                tensorConsumers_[inTensorMagic] = tensorConsumers_[outTensorMagic];
            }
            replacedTensors[outTensorMagic] = in;
            return true;
        }
        inIdx++;
    }

    return false;
}

void SrcDstBufferMergeImpl::NotFindReplacedProcess(const Operation *ops,
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> &replacedTensors) {
    for (auto &out : ops->GetOOperands()) {
        auto outTensorMagic = out->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
        ALOG_DEBUG_F("Op %d out tensor magic: %d", ops->GetOpMagic(), outTensorMagic);
        if (replacedTensors.find(outTensorMagic) != replacedTensors.end()) {
            ALOG_DEBUG_F("Find tensor: %d  replaced by tensor: %d", 
                outTensorMagic, replacedTensors[outTensorMagic]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId);
            out->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId =
                replacedTensors[outTensorMagic]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
        }
    }
}

Status SrcDstBufferMergeImpl::Run(Function &func) {
    if (func.rootFunc_ == nullptr) {
        ALOG_ERROR_F("RootFunc is null");
        return FAILED;
    }
    for (auto &subProgram : func.rootFunc_->programs_) {
        ALOG_INFO_F("Merge src dst for program id : [%lu]", subProgram.first);
        ResetMemoryMap(*subProgram.second);
        auto opList = subProgram.second->Operations().DuplicatedOpList();
        if (Init(opList) != SUCCESS) {
            return FAILED;
        }
        auto oriOps(opList);
        std::unordered_map<int, std::shared_ptr<LogicalTensor>> replacedTensors;
        for (size_t i = 0; i < oriOps.size(); i++) {
            ALOG_DEBUG_F("Try reuse op [%d] input by out tensor", oriOps[i]->GetOpMagic());
            if (CheckIgnoreScene(oriOps[i])) {
                continue;
            }
            int inIdx = 0;
            auto hasInplaced = CheckHasInplaced(oriOps[i], opList[i], replacedTensors, inIdx);
            if (hasInplaced.second == FAILED) {
                return FAILED;
            }
            if (hasInplaced.first) {
                continue;
            }
            bool findReplaced = FindReplaced(oriOps[i], opList[i], replacedTensors, inIdx);
            if (!findReplaced) {
                NotFindReplacedProcess(opList[i], replacedTensors);
            }
        }
    }
    return SUCCESS;
}

bool SrcDstBufferMergeImpl::CheckAssembleReuse(const LogicalTensorPtr &outOperand) {
    for (auto consumer : outOperand->GetConsumers()) {
        if (consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        for (auto assembleOutTensor : consumer->GetOOperands()) {
            if (assembleOutTensor->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId == outOperand->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId) {
                ALOG_DEBUG_F("Assemble cannot be reused.");
                return false;
            }
        }
    }
    return true;
}

bool SrcDstBufferMergeImpl::CanSrcDstReuse(const Operation *ops,
    std::shared_ptr<LogicalTensor> ioperand, bool strict) {
    if (ops->GetOOperands().size() == 0) {
        return false;
    }
    if (ops->GetOpcode() == Opcode::OP_SCATTER_ELEMENT) {
        if (ioperand == ops->GetIOperands()[0]) {
            return true;
        }
    }
    auto outOperand = ops->GetOOperands()[0];
    ALOG_DEBUG_F("Try reuse src %d dst %d", ioperand->GetMagic(), outOperand->GetMagic());
    if (outOperand->GetMemoryTypeOriginal() != ioperand->GetMemoryTypeOriginal()) {
        ALOG_DEBUG_F("Memtype is not same");
        return false;
    }
    // tile shape 必须一样
    if (tensorMaxSize_[outOperand->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId] !=
        tensorMaxSize_[ioperand->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId]) {
        ALOG_DEBUG_F("Datasize is not same");
        return false;
    }
    if (strict && outOperand->Datatype() != ioperand->Datatype()) {
        ALOG_DEBUG_F("Datatype is not same");
        return false;
    }
    if (!CheckAssembleReuse(outOperand)) {
        ALOG_DEBUG_F("Check Assemble op which cannot be reused.");
        return false;
    }
    // 确保复用UB buffer后不会被覆写
    auto iter = tensorConsumers_.find(ioperand->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId);
    if (iter != tensorConsumers_.end() && iter->second.size() > 1) {
        ALOG_DEBUG_F("Has more 1 output");
        return false;
    }
    return true;
}
} // namespace npu::tile_fwk