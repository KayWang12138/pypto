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
 * \file partitioner.cpp
 * \brief
 */

#include "passes/tile_graph_pass/partitioner.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
constexpr int32_t COLOR_NOT_SET_FLAG = -2;

void GraphParitioner::Run(Function *funcPtr) {
    opList_.clear();
    UpdateInOutGraph(funcPtr);
    BuildSuperNodes();
    ColorSuperNodes();
    MergeColor();
    funcPtr->SetTotalSubGraphCount(color_);
}

bool GraphParitioner::CoreTypeMergeable(const std::set<OpCoreType>& coreTypes) {
    if (coreTypes.size() == 1) {
        return true;
    }
    const size_t maxSeperateCoreNum = 2;
    if (coreTypes.size() > maxSeperateCoreNum) {
        return false;
    }
    if (coreTypes.size() == maxSeperateCoreNum) {
        auto firstType = *coreTypes.begin();
        auto secondType = *(++coreTypes.begin());
        if (firstType == OpCoreType::AICPU || secondType == OpCoreType::AICPU) {
            return false;
        }
        if (firstType == OpCoreType::ANY || secondType == OpCoreType::ANY) {
            return true;
        }
    }
    return false;
}

int GraphParitioner::FindParent(std::vector<int>& parent, int i) {
  if (parent[i] == i) {
    return i;
  }
  parent[i] = FindParent(parent, parent[i]);
  return parent[i];
}

void GraphParitioner::MergeSrcToDstIsland(std::vector<int>& parent, int src, int dst) {
    int srcParent = FindParent(parent, src);
    int dstParent = FindParent(parent, dst);
    std::set<OpCoreType> coreTypes{opCoreType_[src], opCoreType_[dst],
                                   opCoreType_[srcParent], opCoreType_[dstParent]};
    if (CoreTypeMergeable(coreTypes)) {
        parent[srcParent] = dstParent;
    }
}

void GraphParitioner::UpdateInOutGraph(Function *funcPtr) {
    auto oriOperations = funcPtr->Operations();
    size_t opCount{0};
    for (size_t i = 0; i < oriOperations.size(); i++) {
        if (oriOperations[i].IsNOP()) {
            continue;
        }
        opList_.push_back(&oriOperations[i]);
        opMagic2Idx_[oriOperations[i].GetOpMagic()] = opCount;
        opCount++;
    }
    opInGraph_.resize(opList_.size());
    opOutGraph_.resize(opList_.size());
    std::vector<uint64_t> opHash(opList_.size(), 0);
    for (size_t i = 0; i < opList_.size(); i++) {
        opInGraph_[i].clear();
        opOutGraph_[i].clear();
        for (auto &inputOperand: opList_[i]->GetIOperands()) {
            for (auto &parentOpPtr : inputOperand->GetProducers()) {
                auto opIter = opMagic2Idx_.find(parentOpPtr->GetOpMagic());
                if (opIter == opMagic2Idx_.end()) {
                    continue;
                }
                opInGraph_[i].push_back(opIter->second);
                opOutGraph_[opIter->second].push_back(i);
            }
        }
    }
}

std::vector<int> GraphParitioner::GetSameLevelOpIdx(int opIdx, Opcode opLabel)
{
    std::vector<int> res;
    std::shared_ptr<LogicalTensor> output = opList_[opIdx]->GetOOperands()[0];
    for (auto& parentOpPtr : output->GetProducers()) {
        if (parentOpPtr->GetOpcode() == opLabel) {
            int targetIdx = opMagic2Idx_[parentOpPtr->GetOpMagic()];
            res.push_back(targetIdx);
        }
    }
    return res;
}

void GraphParitioner::BuildSuperNodes() {
    opCoreType_.resize(opList_.size());
    std::vector<int> parent(opList_.size());
    for (size_t i = 0; i < opList_.size(); i++) {
        parent[i] = i;
        opCoreType_[i] = OpcodeManager::Inst().GetCoreType(opList_[i]->GetOpcode());
    }
    for (size_t i = 0; i < opList_.size(); i++) {
        // L1 copy in reuse
        if (opList_[i]->GetOOperands().size() == 1U && opList_[i]->GetOOperands()[0]->GetMemoryTypeToBe() == MemoryType::MEM_L1) {
            for (auto outNode : opOutGraph_[i]) {
                MergeSrcToDstIsland(parent, outNode, i);
            }
            continue;
        }
        // assemble 特殊处理, assemble到local tensor，需要将这些assemble统一island
        if (opList_[i]->GetOpcode() == Opcode::OP_ASSEMBLE) {
            if ((opList_[i]->GetOOperands()[0]->GetMemoryTypeToBe() != MemoryType::MEM_HOST1) &&
                (opList_[i]->GetOOperands()[0]->GetMemoryTypeToBe() != MemoryType::MEM_DEVICE_DDR)) {
                for (int opIdx : GetSameLevelOpIdx(i, Opcode::OP_ASSEMBLE)) {
                    MergeSrcToDstIsland(parent, opIdx, i);
                }
            }
            // assmemble和其输入绑定
            MergeSrcToDstIsland(parent, i, *(opInGraph_[i].begin()));
            continue;
        }
        // 所有的copyout操作与其输入绑定
        if (OpcodeManager::Inst().GetOpCalcType(opList_[i]->GetOpcode()) == OpCalcType::MOVE_OUT &&
            opList_[i]->ProducerOps().size() == 1U) {
            for (auto inputTensor : opList_[i]->GetIOperands()) {
                if (inputTensor->GetMemoryTypeToBe() != MemoryType::MEM_HOST1 && inputTensor->GetMemoryTypeToBe() != MemoryType::MEM_DEVICE_DDR) {
                    for (auto& producer : inputTensor->GetProducers()) {
                        MergeSrcToDstIsland(parent, opMagic2Idx_[producer->GetOpMagic()], i);
                    }
                }
            }
            continue;
        }
        // 所有的copyin操作与其输出绑定
        if ((OpcodeManager::Inst().GetOpCalcType(opList_[i]->GetOpcode()) == OpCalcType::MOVE_IN ||
             OpcodeManager::Inst().GetOpCalcType(opList_[i]->GetOpcode()) == OpCalcType::MOVE_LOCAL)
            && opOutGraph_[i].size() == 1) {
            MergeSrcToDstIsland(parent, i, *(opOutGraph_[i].begin()));
            continue;
        }
        // mulcc需要与其输入mul绑定
        if (opList_[i]->GetOpcode() == Opcode::OP_A_MULACC_B || opList_[i]->GetOpcode() == Opcode::OP_A_MULACC_BT) {
            for (auto inOp : opInGraph_[i]) {
                if (OpcodeManager::Inst().GetOpCalcType(opList_[i]->GetOpcode()) == OpCalcType::MATMUL) {
                    MergeSrcToDstIsland(parent, i, inOp);
                }
            }
            continue;
        }
    }
    UpdateSuperNode(parent);
}

void GraphParitioner::UpdateSuperNode(std::vector<int>& parent)
{
    std::vector<int> parentToSuperNodes(opList_.size(), -1);
    std::vector<std::vector<int>> superNodeList;
    std::vector<OpCoreType> nodeCoreType;
    op2superNodeIdx_.resize(opList_.size());
    for (int i = 0; i < static_cast<int>(opList_.size()); i++) {
        if (FindParent(parent, i) == i) {
            parentToSuperNodes[i] = superNodeList.size();
            // super_nodes.size increse by 1 each time
            superNodeList.push_back(std::vector<int>());
            nodeCoreType.push_back(OpCoreType::AIV);
        }
    }
    for (int i = 0; i < static_cast<int>(opList_.size()); i++) {
        op2superNodeIdx_[i] = parentToSuperNodes[FindParent(parent, i)];
        superNodeList[op2superNodeIdx_[i]].push_back(i);
        if (opCoreType_[i] != OpCoreType::ANY && opCoreType_[i] != OpCoreType::AIV) {
            nodeCoreType[op2superNodeIdx_[i]] = opCoreType_[i];
        }
    }
    std::vector<std::vector<int>> superInGraph(superNodeList.size(), std::vector<int>());
    std::vector<std::vector<int>> superOutGraph(superNodeList.size(), std::vector<int>());
    for (size_t i = 0; i < opList_.size(); i++) {
        for (int j : opOutGraph_[i]) {
            if (op2superNodeIdx_[i] != op2superNodeIdx_[j]) {
                if (std::find(superOutGraph[op2superNodeIdx_[i]].begin(), superOutGraph[op2superNodeIdx_[i]].end(), op2superNodeIdx_[j]) ==
                              superOutGraph[op2superNodeIdx_[i]].end()) {
                    superOutGraph[op2superNodeIdx_[i]].push_back(op2superNodeIdx_[j]);
                    superInGraph[op2superNodeIdx_[j]].push_back(op2superNodeIdx_[i]);
                }
            }
        }
    }
    superNodeHash_.resize(superNodeList.size());
    for (size_t i = 0; i < superNodeList.size(); i++) {
        std::set<int> superNodeSet(superNodeList[i].begin(), superNodeList[i].end());
        std::stringstream ss;
        std::stringstream ssDebug;
        for (auto opIdx : superNodeList[i]) {
            ss << opList_[opIdx]->GetOpcodeStr() <<", ";
            ssDebug << opList_[opIdx]->GetOpcodeStr() << "[" << opList_[opIdx]->GetOpMagic() <<"], ";
        }
        superNodeHash_[i] = ss.str();
        ASLOGI("super %zu node: %s", i, ssDebug.str().c_str());
    }
    superNodes_.swap(superNodeList);
    nodeCoreType_.swap(nodeCoreType);
    superInGraph_.swap(superInGraph);
    superOutGraph_.swap(superOutGraph);
}

bool GraphParitioner::IsAllOutSame(const std::vector<int>& outSuperNodes) {
    if (outSuperNodes.size() <= 1U) {
        return false;
    }
    std::set<std::string> opcodes;
    for (auto outSuperNode : outSuperNodes) {
        opcodes.insert(superNodeHash_[outSuperNode]);
    }
    return (opcodes.size() == 1U);
}

void GraphParitioner::ProcessSuperNode(const int superNodeIdx, std::vector<int>& superDegree,
                                       std::queue<int>& superQueue, int color) {
    superNodesColor_[superNodeIdx] = color;
    ALOG_INFO_F("Set super node idx [%d] color [%d].", superNodeIdx, color);
    bool isAllOutSame = IsAllOutSame(superOutGraph_[superNodeIdx]);
    for (int i : superOutGraph_[superNodeIdx]) {
        superDegree[i]--;
        ALOG_INFO_F("Super node out [%d] is all same [%d].", i, isAllOutSame);
        // 如果是串行结构，直接将其color设置成与superNodeIdx相同
        std::set<OpCoreType> coreTypes{nodeCoreType_[superNodeIdx], nodeCoreType_[i]};
        if (superDegree[i] == 0) {
            // 需要避免所有输出节点是并行分支的情况
            if (isAllOutSame) {
                superQueue.push(i);
                continue;
            }
            if (superInGraph_[i].size() == 1U && CoreTypeMergeable(coreTypes)) {
                ProcessSuperNode(i, superDegree, superQueue, color);
            } else if (superNodesColor_[i] == -1) {
                superQueue.push(i);
            }
        }
    }
}

void GraphParitioner::ColorSuperNodes() {
    superNodesColor_.clear();
    for (size_t i = 0; i < superNodes_.size(); i++) {
      superNodesColor_.push_back(-1);
    }
    std::vector<int> superDegree(superNodes_.size());
    std::queue<int> superQueue;
    // find all the input & vector super nodes
    for (size_t i = 0; i < superNodes_.size(); i++) {
        superDegree[i] = superInGraph_[i].size();
        if (superDegree[i] == 0) {
            superQueue.push(i);
        }
    }
    while (!superQueue.empty()) {
        auto superNodeIdx = superQueue.front();
        superQueue.pop();
        ProcessSuperNode(superNodeIdx, superDegree, superQueue, color_++);
    }
    // update the color of each tile op
    for (size_t i = 0; i < opList_.size(); i++) {
        opList_[i]->UpdateSubgraphID(superNodesColor_[op2superNodeIdx_[i]]);
    }
}

void GraphParitioner::MergeColor() {
    std::vector<std::set<int>> inColorGraph(color_);
    std::vector<std::set<int>> outColorGraph(color_);
    std::map<int, std::vector<size_t>> color2superNode;
    std::vector<OpCoreType> colorCoreType(color_, OpCoreType::AIV);
    for (size_t i = 0; i < superNodes_.size(); i++) {
        color2superNode[superNodesColor_[i]].push_back(i);
        for (auto inSuperNodeIdx : superInGraph_[i]) {
            if (superNodesColor_[inSuperNodeIdx] != superNodesColor_[i]) {
                inColorGraph[superNodesColor_[i]].insert(superNodesColor_[inSuperNodeIdx]);
                outColorGraph[superNodesColor_[inSuperNodeIdx]].insert(superNodesColor_[i]);
            }
        }
        if (nodeCoreType_[i] != OpCoreType::ANY && nodeCoreType_[i] != OpCoreType::AIV) {
            colorCoreType[superNodesColor_[i]] = nodeCoreType_[i];
        }
    }
    for (size_t i = 0; i < colorCoreType.size(); i++) {
        bool isCube{false};
        if (colorCoreType[i] == OpCoreType::AIC) {
            isCube = true;
        }
        auto superNodeVec = color2superNode[i];
        for (auto superNodeIdx : superNodeVec) {
            for (auto opIdx : superNodes_[superNodeIdx]) {
                opList_[opIdx]->SetAttribute(OpAttributeKey::isCube, isCube);
            }
        }
    }
    std::vector<int> oldColor2NewColor(color_);
    for (int colorId = 0; colorId < color_; colorId++) {
        oldColor2NewColor[colorId] = colorId;
    }
    for (int colorId = 0; colorId < color_; colorId++) {
        if (outColorGraph[colorId].size() != 1U) {
            continue;
        }
        std::set<OpCoreType> coreTypes{colorCoreType[colorId], colorCoreType[*outColorGraph[colorId].begin()]};
        if (!CoreTypeMergeable(coreTypes)) {
            continue;
        }
        if (inColorGraph[*outColorGraph[colorId].begin()].size() == 1U) {
            oldColor2NewColor[*outColorGraph[colorId].begin()] = oldColor2NewColor[colorId];
            continue;
        }
        if (color2superNode[colorId].size() == 1U) {
            oldColor2NewColor[colorId] = oldColor2NewColor[*outColorGraph[colorId].begin()];
            continue;
        }
    }
    std::map<int, int> sortedColors;
    int colorCount{0};
    for (auto &color : oldColor2NewColor) {
        if (sortedColors.find(color) == sortedColors.end()) {
            sortedColors[color] = colorCount++;
        }
    }
    std::set<int> colorSet;
    for (auto op : opList_) {
        ALOG_INFO_F("op %s[%d] old color[%d] change to [%d].", op->GetOpcodeStr().c_str(), op->GetOpMagic(), op->GetSubgraphID(),
                    sortedColors.at(oldColor2NewColor[op->GetSubgraphID()]));
        op->UpdateSubgraphID(sortedColors.at(oldColor2NewColor[op->GetSubgraphID()]));
        colorSet.insert(op->GetSubgraphID());
    }
    ALOG_INFO_F("Subgraph size [%d] is changed to [%zu].", color_, colorSet.size());
    color_ = static_cast<int>(colorSet.size());
}

Status PartitionVCPass::RunOnFunction(Function& function)
{
    ASLOGI("===> Start PartitionVCPass.");
    GraphParitioner graphPartitioner;
    Function* funcPtr = &function;
    graphPartitioner.Run(funcPtr);
    ASLOGI("===> End PartitionVCPass.");
    return SUCCESS;
}

Status PartitionVCPass::PreCheck(Function &function) {
    ALOG_INFO("PreCheck for pass: PartitionVCPass");
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR("Loopcheck failed before pass: PartitionVCPass");
    }
    return SUCCESS;
}

Status PartitionVCPass::PostCheck(Function &function) {
    ALOG_INFO("PostCheck for pass: PartitionVCPass");
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR("Loopcheck failed after pass: PartitionVCPass");
    }

    for (auto &op : function.Operations()) {
        if (op.GetIOperands().size() == 1 && op.GetIOperands()[0]->GetProducers().size() == 0 &&
            (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_COPY_OUT)) {
            ALOG_WARN_F("Operation magic: %d, when indegree is 0, opcode should not be %s", op.GetOpMagic(), op.GetOpcodeStr().c_str());
        }
        if (op.GetOOperands().size() == 1 && op.GetOOperands()[0]->GetConsumers().size() == 0 &&
            (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_COPY_IN)) {
            ALOG_WARN_F("Operation magic: %d, when outdegree is 0, opcode should not be %s", op.GetOpMagic(), op.GetOpcodeStr().c_str());
        }
    }
    return SUCCESS;
}
}  // namespace npu::tile_fwk