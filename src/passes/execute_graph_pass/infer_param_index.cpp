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
 * \file infer_param_index.cpp
 * \brief
 */

#include <queue>
#include "infer_param_index.h"
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/operation/opcode.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace npu::tile_fwk {
std::string InferParamIndexPass::DumpParamIndex(const std::map<std::string, DynParamInfo>& dynParamTable)
{
    std::ostringstream ss;
    for (auto paramInfo : dynParamTable) {
        ss << "param: " << paramInfo.first << " ( ";
        ss << "tensorIdx: " << paramInfo.second.tensorIndex << ", ";
        ss << "dimsize: " << paramInfo.second.dimSize << ", ";
        ss << "type: " << static_cast<int>(paramInfo.second.type) << ", ";
        ss << "addrIdx: " << paramInfo.second.tensorBaseAddrIndex << ", ";
        ss << "dimIdx: " << paramInfo.second.dimIndex << " )" << std::endl;
    }
    return ss.str();
}

void ResetDynValidShape(Function& function) {
    for (auto &op : function.Operations()) {
        std::vector<SymbolicScalar> validShape;
        for (auto outOperand : op.GetOOperands()) {
            // 输入输出的tensor shape符号化
            if (OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode())) {
                for (size_t dimIdx = 0U; dimIdx < outOperand->GetShape().size(); ++dimIdx) {
                    validShape.push_back(SymbolicScalar("sym_" + std::to_string(outOperand->GetMagic()) + "_dim_" +
                        std::to_string(dimIdx)));
                }
            }
            outOperand->UpdateDynValidShape(validShape);
        }
        // 清空view和assemble的属性中的dynvalidshape，以便后续重新推导符号化的dynvalidshape
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
            if (viewOpAttribute != nullptr) {
                viewOpAttribute->SetToDynValidShape(std::vector<SymbolicScalar>());
            }
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get());
            if (assembleOpAttribute != nullptr) {
                auto emptyValidShape = std::vector<SymbolicScalar>();
                assembleOpAttribute->SetFromDynValidShape(emptyValidShape);
            }
            continue;
        }
    }
}

bool NeedInferShape(const Operation *op) {
    if (op->GetOOperands().empty()) {
        return false;
    }
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty())) {
        return false;
    }
    return true;
}

void InferShape(Function &function)
{
    size_t i = 0U;
    std::map<int, size_t> opMagic2Idx;
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();
    for (auto op : opList) {
        opMagic2Idx[op->GetOpMagic()] = i;
        i++;
    }
    std::vector<std::vector<size_t>> opInGraph(opList.size());
    std::vector<std::vector<size_t>> opOutGraph(opList.size());
    for (auto op : opList) {
        for (auto producer : op->ProducerOps()) {
            opInGraph[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[producer->GetOpMagic()]);
            opOutGraph[opMagic2Idx[producer->GetOpMagic()]].push_back(opMagic2Idx[op->GetOpMagic()]);
        }
    }
    std::queue<size_t> procOpQueue;
    std::vector<size_t> inDegree(opList.size(), 0);
    for (size_t j = 0; j < opInGraph.size(); ++j) {
        if (opInGraph[j].empty()) {
            procOpQueue.push(j);
        }
        inDegree[j] = opInGraph[j].size();
    }
    while (!procOpQueue.empty()) {
        auto opIdx = procOpQueue.front();
        procOpQueue.pop();
        for (auto outIdx : opOutGraph[opIdx]) {
            inDegree[outIdx]--;
            if (inDegree[outIdx] == 0) {
                procOpQueue.push(outIdx);
            }
        }
        if (NeedInferShape(opList[opIdx])) {
            InferShapeRegistry::GetInstance().CallInferShapeFunc(opList[opIdx]);
        }
    }
}

Status InferParamIndexPass::RunOnFunction(Function &function)
{
    ASLOGI("===> Start InferParamIndexPass.");
    for (auto &subProgram : function.rootFunc_->programs_) {
        auto &subFunc = *subProgram.second;
        ResetDynValidShape(subFunc);
        InferShape(subFunc);
        ALOG_INFO(subFunc.Dump());
        std::map<int, std::vector<SymbolicScalar>> addr2ValidShape;
        for (auto &op : subFunc.Operations()) {
            if (!OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode())) {
                continue;
            }
            int addrPos;
            if (IsCopyIn(op.GetOpcode()))
                addrPos = op.GetIOpAttrOffset(0);
            else
                addrPos = op.GetOOpAttrOffset(0);
            if (addrPos == -1)
                continue;
            if (addr2ValidShape.find(addrPos) == addr2ValidShape.end()) {
                addr2ValidShape[addrPos] = op.GetOOperands()[0]->GetDynValidShape();
            }
        }
        std::map<int, int> addrIdx2GmIdx;
        std::set<std::string> visitedSymbol;
        int gmIdx{0};
        for (auto validShape : addr2ValidShape) {
            addrIdx2GmIdx[validShape.first] = gmIdx;
            int dimIdx{0};
            for (auto dim : validShape.second) {
                if (!dim.IsSymbol()) {
                    continue;
                }
                if (visitedSymbol.count(dim.Dump()) > 0) {
                    continue;
                }
                auto paramInfo = DynParamInfo{static_cast<int>(validShape.second.size()), gmIdx, validShape.first, DynParamInfoType::VALID_SHAPE, dimIdx};
                subFunc.InsertDynParam(dim.Dump(), paramInfo);
                dimIdx++;
            }
            gmIdx++;
        }
        ALOG_DEBUG(DumpParamIndex(subFunc.GetDynParamTable()));
    }
    ASLOGI("===> End InferParamIndexPass By Sequential Execution.");
    return SUCCESS;
}
}  // namespace npu::tile_fwk
