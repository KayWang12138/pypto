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
 * \file supernode_graph_builder.h
 * \brief
 */

#ifndef SUPERNODE_GRAPH_BUILDER_H
#define SUPERNODE_GRAPH_BUILDER_H
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {
class OperationGraphInfo {
public:
    uint64_t GetHash(const Operation *op) const;
    bool CoreTypeMergeable(const std::set<OpCoreType> &coreTypes) const;
    std::vector<int32_t> GetSameLevelOpIdx(int32_t opIdx, Opcode opLabel) const;
    std::vector<Operation*> opList_;
    std::unordered_map<int32_t, int32_t> magic2Idx_;
    std::vector<std::set<int32_t>> inGraph_;
    std::vector<std::set<int32_t>> outGraph_;
    std::vector<uint64_t> opHashList_;
    std::vector<OpCoreType> opCoreType_;
    bool useCVMixPartition_ = false;
};

class NodeGraphInfo {
public:
    Status Build(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                 const std::vector<std::pair<int32_t, int32_t>> &mergePair, bool markIsCube);
    Status AvoidLoop(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                     std::vector<int32_t> &parent, std::vector<std::vector<int32_t>> &node2Op, bool &updated);
    Status BuildInOutGraph(const std::shared_ptr<OperationGraphInfo> operationGraphInfo, bool markIsCube);
    int32_t FindParent(std::vector<int32_t> &parent, int32_t i);
    Status MergeSrcToDstIsland(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                               std::vector<int32_t> &parent, int32_t src, int32_t dst);
    int32_t GetNodeCycle(int32_t nodeIdx) const;
    bool GetNodeMergeable(const std::shared_ptr<OperationGraphInfo> operationGraphInfo, int32_t nodeIdx);
    std::vector<std::vector<int32_t>> node2Op_;
    std::vector<int32_t> op2Node_;
    std::vector<std::set<int32_t>> nodeInGraph_;
    std::vector<std::set<int32_t>> nodeOutGraph_;
    std::vector<std::vector<int32_t>> nodeInGraphList_;
    std::vector<std::vector<int32_t>> nodeOutGraphList_;
    std::vector<OpCoreType> nodeCoreType_;
    std::vector<int32_t> nodeCycles_;
    std::vector<bool> nodeMergeable_;
    std::vector<uint64_t> nodeHashList_;
    std::unordered_map<uint64_t, std::vector<int32_t>> hash2NodeMap_;
};

class SuperNodeGraphBuilder {
public:
    SuperNodeGraphBuilder() = default;
    virtual ~SuperNodeGraphBuilder() = default;

protected:
    Status BuildOpGraph(const std::vector<Operation*> &opList);
    virtual Status BuildSuperNodeGraph();
    Status BuildHashValues();

    // BuildSuperNodeGraph helpers
    inline bool L1CopyInCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair);
    inline bool AssembleCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair);
    inline bool CopyOutCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair, bool assembleScene);
    inline bool CopyInCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair);
    inline bool MulAccCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair);
    inline bool AssembleToCopyoutScene(Operation *op);
    
    // BuildHashValues helpers
    uint64_t CombineHash(const uint64_t h1, const uint64_t h2) const;
    std::vector<std::pair<int32_t, int32_t>> GetReduceNodeMergePair() const;
    Status BuildReduceNodeHash(std::shared_ptr<NodeGraphInfo> reduceNodeInfo);
    Status BuildBalanceOpHash(std::vector<uint64_t> &opHashList);
    
    // Parameters
    bool useReduceBalanceHash_ = true;
    bool useCVMixPartition_ = false;

    // Data
    std::shared_ptr<OperationGraphInfo> operationInfo_;
    std::shared_ptr<NodeGraphInfo> superNodeInfo_;
};

inline bool SuperNodeGraphBuilder::L1CopyInCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    if (opList[i]->GetOOperands().size() > 0 &&
        opList[i]->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        for (auto outNode : operationInfo->outGraph_[i]) {
            mergePair.emplace_back(outNode, i);
            APASS_LOG_DEBUG_F("GraphPartition", "Operation", "Combine %d and %d for L1 CopyIn in building SuperNode.",
                opList[i]->GetOpMagic(), opList[outNode]->GetOpMagic());
        }
        return true;
    }
    return false;
}

inline bool L0CCopyL1Combine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    if (opList[i]->GetIOperands().size() == 1U &&
        opList[i]->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L0C &&
        opList[i]->GetOOperands().size() == 1U &&
        opList[i]->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        for (auto inNode : operationInfo->inGraph_[i]) {
            mergePair.emplace_back(inNode, i);
        }
        for (auto outNode : operationInfo->outGraph_[i]) {
            mergePair.emplace_back(outNode, i);
        }
        return true;
    }
    return false;
}

inline bool SuperNodeGraphBuilder::AssembleCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    // assemble 特殊处理, assemble到local tensor，需要将这些assemble统一island
    if (opList[i]->GetOpcode() == Opcode::OP_ASSEMBLE) {
        if (opList[i]->GetOOperands().size() == 0) {
            return false;
        }
        if (AssembleToCopyoutScene(opList[i])) {
            // 在GenerateMoveOp中需要转换为CopyOut的Assemble, 参考CopyOutCombine处理
            return CopyOutCombine(operationInfo, opList, i, mergePair, true);
        }
        // assmemble和其输入绑定
        if (operationInfo->inGraph_[i].size() > 0) {
            mergePair.emplace_back(i, *(operationInfo->inGraph_[i].begin()));
            APASS_LOG_DEBUG_F("GraphPartition", "Operation", "Combine %d and %d for Assemble in building SuperNode.",
                         opList[i]->GetOpMagic(), opList[*(operationInfo->inGraph_[i].begin())]->GetOpMagic());
        }
        return true;
    }
    return false;
}

inline bool SuperNodeGraphBuilder::CopyOutCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair, bool assembleScene)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    std::vector<int32_t> candidateOpMagic;
    // 所有的copyout操作与其输入绑定
    if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_OUT || assembleScene) {
        for (auto inNode : operationInfo->inGraph_[i]) {
            mergePair.emplace_back(inNode, i);
            APASS_LOG_DEBUG_F("GraphPartition", "Operation", "Combine %d and %d for CopyOut in building SuperNode.",
                opList[operationInfo->magic2Idx_[inNode]]->GetOpMagic(), opList[i]->GetOpMagic());
        }
        return true;
    }
    return false;
}

inline bool SuperNodeGraphBuilder::CopyInCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                          int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    // 所有的copyin操作与其输出绑定
    if ((OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_IN ||
         OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_LOCAL) &&
        operationInfo->outGraph_[i].size() > 0) {
        mergePair.emplace_back(i, *(operationInfo->outGraph_[i].begin()));
        APASS_LOG_DEBUG_F("GraphPartition", "Operation", "Combine %d and %d for CopyIn in building SuperNode.",
                     opList[i]->GetOpMagic(), opList[*(operationInfo->outGraph_[i].begin())]->GetOpMagic());
        return true;
    }
    return false;
}

inline bool SuperNodeGraphBuilder::MulAccCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                          int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    // MulAcc需要与其输入mul绑定
    if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MATMUL) {
        for (auto inOp : operationInfo->inGraph_[i]) {
            if (OpcodeManager::Inst().GetOpCalcType(opList[inOp]->GetOpcode()) == OpCalcType::MATMUL) {
                mergePair.emplace_back(i, inOp);
                APASS_LOG_DEBUG_F("GraphPartition", "Operation", "Combine %d and %d for MulAcc in building SuperNode.",
                             opList[i]->GetOpMagic(), opList[inOp]->GetOpMagic());
            }
        }
        return true;
    }
    return false;
}

inline bool SuperNodeGraphBuilder::AssembleToCopyoutScene(Operation *op)
{
    auto ASSEMBLE_in = op->iOperand.front();
    auto parentOp = *ASSEMBLE_in->GetProducers().begin();
    if (op->iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR ||
        op->oOperand.front()->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR ||
        parentOp->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT || parentOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
        return false;
    }
    return true;
}

}  // namespace npu::tile_fwk
#endif  // SUPERNODE_GRAPH_BUILDER_H