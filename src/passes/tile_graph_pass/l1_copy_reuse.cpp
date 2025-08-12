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
 * \file l1_copy_reuse.cpp
 * \brief
 */

#include "l1_copy_reuse.h"
#include "passes/pass_config/pass_config_manager.h"

namespace npu::tile_fwk {
inline std::vector<int> GetGMInputFeature(const Operation &op) { // 提取GM tensor的特征
    auto ioperand = op.GetIOperands()[0];
    if (ioperand == nullptr) {
        ALOG_ERROR_F("[L1CopyReuse] op %d ioperand is nullptr", op.GetOpMagic());
        return {};
    }
    std::vector<int> vec = {ioperand->GetRawTensor()->GetRawMagic()};
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    std::vector<OpImmediate> opImmList = attr->GetCopyInAttr().first;
    for (auto &opImm : opImmList){
        auto offset = opImm.GetSpecifiedValue();
        if (offset.ConcreteValid()) {
            vec.push_back(offset);
        } else {
            std::hash<std::string> hasher;
            auto offsetHash = hasher(opImm.Dump());
            vec.push_back(static_cast<int>(offsetHash));
        }
    }
    auto shape = attr->GetSpecifiedShape(1);
    vec.insert(vec.end(), shape.begin(), shape.end());
    return vec;
}

// key : 需要被删除的copyin op, value: 保留的copyin op
void L1CopyInReuseRunner::GetDuplicateOps(std::vector<Operation *> &opOriList,
                                          const std::vector<int> &opIdx) {
    std::map<std::vector<int>, int> tensor2Op;
    replacedCopyMap_.clear();
    tensormagic2Op_.clear();
    for (auto i : opIdx) {
        if (opOriList[i]->GetOpcode() == Opcode::OP_COPY_IN && opOriList[i]->GetOOperands()[0]->GetMemoryTypeOriginal() == MEM_L1) {
            auto outputMagic = opOriList[i]->GetOOperands()[0]->GetRawTensor()->GetRawMagic();
            auto feature = GetGMInputFeature(*opOriList[i]);
            if (tensor2Op.find(feature) != tensor2Op.end() && tensor2Op[feature] != i) {
                replacedCopyMap_[i] = tensor2Op[feature];
                tensormagic2Op_[outputMagic] = tensor2Op[feature];
            } else {
                tensor2Op[feature] = i;
            }
        }
    }
}

void L1CopyInReuseRunner::TackleOp(int i, Operation *op, std::vector<std::vector<int>> &replacedInputs, std::vector<std::vector<int>> &replacedOutputs) {
    if (op->GetOpcode() == Opcode::OP_COPY_IN && op->GetOOperands()[0]->GetMemoryTypeOriginal() == MEM_L1) {
        auto allocedL1BufId = op->GetOOperands()[0]->GetRawTensor()->GetRawMagic();
        if (tensormagic2Op_.find(allocedL1BufId) != tensormagic2Op_.end()) {
            ALOG_INFO_F("[L1CopyReuse] Remove useless op [%d, %s]", op->GetOpMagic(), op->GetOpcodeStr().c_str());
            op->SetAsDeleted();
        }
    } else {
        for (size_t k = 0; k < op->GetIOperands().size(); k++) {
            auto ioperandID = op->GetIOperands()[k]->GetRawTensor()->GetRawMagic();
            if (tensormagic2Op_.find(ioperandID) != tensormagic2Op_.end()) {
                replacedInputs.push_back({i, static_cast<int>(k), tensormagic2Op_[ioperandID], 0});
            }
        }
        // 这里需要处理控制依赖。
        for (size_t k = 0; k < op->GetOOperands().size(); k++) {
            auto ioperandID = op->GetOOperands()[k]->GetRawTensor()->GetRawMagic();
            if (tensormagic2Op_.find(ioperandID) != tensormagic2Op_.end()) {
                replacedOutputs.push_back({i, static_cast<int>(k), tensormagic2Op_[ioperandID], 0});
            }
        }
    }
}

// 合并重复的L1_COPY_IN和L1_ALLOC节点
void L1CopyInReuseRunner::MergeDupL1CopyIn(
    Function &func, std::vector<std::vector<int>> &colorNode, int color) {
    std::vector<Operation *> oriList;
    for (auto &op : func.Operations()) {
        oriList.emplace_back(&op);
    }
    int colorCount = 0;
    for (int j = 0; j < color; j++) {
        if (colorNode[j].empty()) {
            continue;
        }
        colorCount++;
        std::sort(colorNode[j].begin(), colorNode[j].end());
        GetDuplicateOps(oriList, colorNode[j]);
        std::vector<std::vector<int>> replacedInputs, replacedOutputs;
        for (int i : colorNode[j]) {
            oriList[i]->UpdateSubgraphID(colorCount-1);
            L1CopyInReuseRunner::TackleOp(i, oriList[i], replacedInputs, replacedOutputs);
        }
        // 重新连边
        for (auto &replacedInput : replacedInputs) {
            ALOG_INFO_F("[L1CopyReuse] Relink op [%d] input [%d] to op [%d] output [%d]", oriList[replacedInput[0]]->GetOpMagic(), replacedInput[1],
                        oriList[replacedInput[2]]->GetOpMagic(), replacedInput[3]);
            FunctionUtils::RelinkOperationInput(oriList[replacedInput[0]], replacedInput[1],
                                                oriList[replacedInput[2]], replacedInput[3]);
        }
        for (auto &replacedOutput : replacedOutputs) {
            auto rewriteOp = oriList[replacedOutput[0]];
            auto copyinOp = oriList[replacedOutput[2]];
            assert(func.TensorReuse(rewriteOp->GetOOperands()[replacedOutput[1]], copyinOp->GetOOperands()[0]));
        }
    }
    func.SetTotalSubGraphCount(colorCount);
}

int L1CopyInReuseRunner::GetMaxInColor(const std::vector<int> &nodes, const OperationsViewer &opOriList, int curColor) {
    int maxInColor = -1;
    for (int j : nodes) {
        for (int k : inGraph[j]) {
            auto opColor = opOriList[k].GetSubgraphID();
            if (opColor != curColor) {
                maxInColor = std::max(maxInColor, opColor);
            }
        }
    }
    return maxInColor;
}

inline std::vector<int> GetCopyIn(const OperationsViewer &opOriList, int color, std::vector<std::vector<int>> &colorNode) {
    // 获取子图L1CopyIn数据量
    std::vector<int> colorCopyIn(color, 0);
    for (int i = 0; i < color; i++) {
        for (int j : colorNode[i]) {
            if (opOriList[j].GetOpcode() == Opcode::OP_COPY_IN && opOriList[j].GetOOperands()[0]->GetMemoryTypeOriginal() == MEM_L1) {
                int volume = BytesOf(opOriList[j].GetOOperands()[0]->Datatype());
                std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(opOriList[j].GetOpAttribute());
                auto shape = attr->GetSpecifiedShape(1);
                for (int k : shape) {
                    volume *= k;
                }
                colorCopyIn[i] = colorCopyIn[i] + volume;
            }
        }
    }
    return colorCopyIn;
}

void L1CopyInReuseRunner::GetOpHash(std::vector<uint64_t> &hashList, const std::string op, int idx) {
    uint64_t a = 0x12345678;
    uint64_t p = 37;
    uint64_t hash = 0;
    for (char c : op) {
        hash = hash * p + static_cast<uint64_t>(c);
    }
    for (int j : inGraph[idx]) {
        hash = hash * p + (hashList[j] ^ a);
    }
    hashList[idx] = hash;
}

void L1CopyInReuseRunner::GetColorHash(const OperationsViewer &opOriList, std::vector<uint64_t> &hashColor) {
    std::vector<uint64_t> hashTileOp(opOriList.size(), 0);
    for (size_t i = 0; i < opOriList.size(); i++) {
            GetOpHash(hashTileOp, opOriList[i].GetOpcodeStr(), i);
    }
    uint64_t a = 0x12345678;
    uint64_t p = 23;
    std::set<int32_t> mulaccGraph;
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        if (opOriList[i].GetOpcode() == Opcode::OP_COPY_IN && opOriList[i].GetOOperands()[0]->GetMemoryTypeOriginal() == MEM_L1) {
            mulaccGraph.insert(opOriList[i].GetSubgraphID());
        }
        hashColor[opOriList[i].GetSubgraphID()] = hashColor[opOriList[i].GetSubgraphID()] * p + (hashTileOp[i] ^ a);
    }
    int order = 0;
    for (int i : mulaccGraph) {
        hashMap[hashColor[i]].push_back(i);
        if (hashMap[hashColor[i]].size() == 1) {
            hashOrder[hashColor[i]] = order;
            order++;
        }
    }
    for (auto& entry : hashMap) {
        ALOG_INFO_F("[L1CopyReuse] Subgraph hash: %lu, Subgraph ID: %s.", entry.first, IntVecToStr(entry.second).c_str());
    }
    for (auto& entry : hashOrder) {
        ALOG_INFO_F("[L1CopyReuse] Subgraph hash: %lu, Hash order: %d.", entry.first, entry.second);
    }
}

inline void HashUpdate(std::unordered_map<uint64_t, std::vector<int>> &hashMap, std::unordered_map<uint64_t, int> &hashOrder, int color, std::vector<uint64_t> hashColor) {
    // 更新子图哈希
    for (auto entry = hashMap.begin(); entry != hashMap.end();) {
        if (entry->second.empty()) {
            entry = hashMap.erase(entry);
        } else {
            entry++;
        }
    }
    hashOrder.clear();
    int order = 0;
    for (int i = 0; i < color; i++) {
        if (hashMap.find(hashColor[i]) != hashMap.end() && hashOrder.find(hashColor[i]) == hashOrder.end()) {
            hashOrder[hashColor[i]] = order;
            order++;
        }
    }
    for (auto& entry : hashMap) {
        ALOG_INFO_F("[L1CopyReuse] Subgraph hash: %lu, Subgraph ID: %s.", entry.first, IntVecToStr(entry.second).c_str());
    }
    for (auto& entry : hashOrder) {
        ALOG_INFO_F("[L1CopyReuse] Subgraph hash: %lu, Hash order: %d.", entry.first, entry.second);
    }
}

std::vector<int> L1CopyInReuseRunner::SetNumLR() {
    std::vector<int> numLRList(hashMap.size(), numLR);
    for (auto &entry : numLRMap) {
        int i = entry.first;
        if ( i >= 0 && i < static_cast<int>(hashMap.size())) {
            numLRList[i] = entry.second;
        }
    }
    return numLRList;
}

void L1CopyInReuseRunner::Phase1(Function &func, int color, std::vector<std::vector<int>> &colorNode,
                                            std::vector<int> &colorCopyIn, std::vector<uint64_t> &hashColor) {
    // 针对matmul的L1 copy reuse进行子图合并
    auto opOriList = func.Operations();
    std::map<std::vector<int>, int> l1InputList;
    std::vector<int> numLRList = SetNumLR();  //L1Reuse参数设置
    std::vector<int> mergedNum(color, 1);
    for (int i = 0; i < color; i++) {
        int tmpColor = -1;
        auto maxInColor = GetMaxInColor(colorNode[i], opOriList, i);
        size_t j = 0;
        while (colorCopyIn[i] <= copyInThreshold && j < colorNode[i].size()) {
            auto opIdx = colorNode[i][j];
            if (opOriList[opIdx].GetOpcode() == Opcode::OP_COPY_IN && opOriList[opIdx].GetOOperands()[0]->GetMemoryTypeOriginal() == MEM_L1) {
                auto vec = GetGMInputFeature(opOriList[opIdx]);
                auto copyId = l1InputList.find(vec);
                if (copyId != l1InputList.end() && copyId->second >= maxInColor && colorCopyIn[copyId->second] + colorCopyIn[i] <= copyInThreshold &&
                    0 < mergedNum[copyId->second] && mergedNum[copyId->second] < numLRList[hashOrder[hashColor[i]]]) {
                    tmpColor = copyId->second;
                    break;
                }
            }
            j++;
        }
        if (tmpColor == -1) {
            tmpColor = i;
        }
        for (auto opIdx : colorNode[i]) {
            if (opOriList[opIdx].GetOpcode() == Opcode::OP_COPY_IN && opOriList[opIdx].GetOOperands()[0]->GetMemoryTypeOriginal() == MEM_L1) {
                auto vec = GetGMInputFeature(opOriList[opIdx]);
                l1InputList[vec] = tmpColor;
            }
        }  // 记录当前子图所有的L1_COPY_IN搬入的tensor特征
        if (tmpColor != i) {
            for (auto t : colorNode[i]) {
                opOriList[t].UpdateSubgraphID(tmpColor);
                colorNode[tmpColor].push_back(t);
            }
            ALOG_INFO_F("[L1CopyReuse] Subgraph merge: %lu, %lu.", i, tmpColor);
            colorNode[i].clear();
            colorCopyIn[tmpColor] = colorCopyIn[tmpColor] + colorCopyIn[i];
            mergedNum[i] = 0;
            mergedNum[tmpColor] += 1;
            hashMap[hashColor[i]].erase(std::find(hashMap[hashColor[i]].begin(), hashMap[hashColor[i]].end(), i));
            hashMap[hashColor[tmpColor]].erase(std::find(hashMap[hashColor[tmpColor]].begin(), hashMap[hashColor[tmpColor]].end(), tmpColor));
            hashColor[tmpColor] += hashColor[i];
            hashColor[i] = 0;
            hashMap[hashColor[tmpColor]].push_back(tmpColor);
        }  // 合入子图
    }
}

std::vector<int> L1CopyInReuseRunner::SetNumDB() {
    std::vector<int> numDBList(hashMap.size(), numDB_);
    for (auto &entry : numDBMap) {
        int i = entry.first;
        if ( i >= 0 && i < static_cast<int>(hashMap.size())) {
            numDBList[i] = entry.second;
        }
    }
    return numDBList;
}

inline std::vector<int> AdjustNumDBCore(bool isLoadBalance, int color, int numDB) {
    std::vector<int> pingColorList(color, 1);
    int numMerged = (color + numDB - 1) / numDB;
    if (isLoadBalance) {
        int coreNum = PassConfigManager::Instance().GetPlatformConfig().GetCoreNum(NpuCoreType::AICORE);
        int columns = numMerged / coreNum;
        int packed = columns * coreNum * numDB;
        for (int i = 0; i < columns * coreNum; i++) {
            pingColorList[numDB * i] = 0;
        }
        // 计算各个核铺满后还剩多少任务
        int remain = color - packed;
        int numDB2 = (remain + coreNum - 1) / coreNum; 
        for (int i = 0; i < (remain + numDB2 - 1) / numDB2; i++) {
            pingColorList[packed + numDB2 * i] = 0;
        }
        return pingColorList;
    }
    for (int i = 0; i < numMerged; i++) {
        pingColorList[numDB * i] = 0;
    }
    return pingColorList;
}

void L1CopyInReuseRunner::Run(Function &func, int color, std::vector<std::vector<int>> &colorNode) {
    auto opOriList = func.Operations();
    // 计算子图哈希，识别同构子图
    std::vector<uint64_t> hashColor(color, 0);
    GetColorHash(opOriList, hashColor);
    // 记录各子图的大小
    auto colorCopyIn = GetCopyIn(opOriList, color, colorNode);
    // 合并阈值参数设置
    copyInThreshold = func.paramConfigs_.sgCopyInThreshold;
    numLR = func.paramConfigs_.l1ReuseNum;
    numDB_ = func.paramConfigs_.cubeNBufferNum;
    numLRMap = func.paramConfigs_.l1ReuseMap;
    numDBMap = func.paramConfigs_.cubeNBufferMap;
    isLoadBalance = func.paramConfigs_.loadBalance;
    ALOG_INFO_F("[L1CopyReuse] Param Setting numLR %d, numDB %d, isLoadBalance %d, copyInThreshold %d.", 
                                                            numLR, numDB_, isLoadBalance, copyInThreshold);
    if (numLR != 0 || numLRMap.size() != 0) {
        Phase1(func, color, colorNode, colorCopyIn, hashColor);
        HashUpdate(hashMap, hashOrder, color, hashColor);
    }
    std::vector<int> hashMergeNum = SetNumDB();  //NBuffer参数设置
    for (auto &entry : hashMap) {
        uint64_t colorHashValue = entry.first;
        std::vector<int> &colorValues = entry.second;
        if (colorCopyIn[colorValues[0]] > copyInThreshold) {
            continue;
        }
        int pingColor = -1;
        std::vector<int> pingColorList = AdjustNumDBCore(isLoadBalance, colorValues.size(), hashMergeNum[hashOrder[colorHashValue]]);
        for (size_t i = 0; i < colorValues.size(); i++) {
            if (pingColorList[i] == 0) {
                pingColor = colorValues[i];
            } else {
                int pongColor = colorValues[i];
                for (auto opIdxMergedDB : colorNode[pongColor]) {
                    opOriList[opIdxMergedDB].UpdateSubgraphID(pingColor);
                    colorNode[pingColor].push_back(opIdxMergedDB);
                }
                ALOG_INFO_F("[L1CopyReuse] Subgraph merge: %lu, %lu.", pingColor, pongColor);
                colorNode[pongColor].clear();
            }
        }
    }
    MergeDupL1CopyIn(func, colorNode, color);
    for (auto &op : func.Operations()) {
        assert(static_cast<size_t>(op.GetSubgraphID()) <= func.GetTotalSubGraphCount());
    }
    RemoveUselessViews(func); //删除节点
    func.EraseOperations(true);
    RescheduleUtils::PrintColorNode(func);
}

void L1CopyInReuseRunner::RemoveUselessViews(Function &func) const {
    for (auto& op : func.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW && op.GetIOperands().size() == 1 && op.GetOOperands().size() == 1) {
            auto input = op.GetIOperands()[0];
            auto output = op.GetOOperands()[0];
            auto iOperandMem = input->GetMemoryTypeOriginal();
            auto oOperandMem = output->GetMemoryTypeOriginal();
            if (iOperandMem == MemoryType::MEM_DEVICE_DDR && oOperandMem == MemoryType::MEM_DEVICE_DDR) {
                bool hasNoConsumer{true};
                for (auto consumer : output->GetConsumers()) {
                    if (!(consumer->IsDeleted()) && consumer->BelongTo() == &func) {
                        hasNoConsumer = false;
                    }
                }
                if (hasNoConsumer) {
                    op.SetAsDeleted();
                }
            }
        }
    }
}

Status L1CopyInReuseMerge::L1CopyInReuse(Function &func) const {
    auto numLR = func.paramConfigs_.l1ReuseNum;
    auto numLRMap = func.paramConfigs_.l1ReuseMap;
    auto numDB = func.paramConfigs_.cubeNBufferNum;
    auto numDBMap = func.paramConfigs_.cubeNBufferMap;
    ALOG_INFO_F("[L1CopyReuse] L1 Reuse Setting: %d", numLR);
    if (numLR == 0 && numDB == 1 && numLRMap.size() == 0 && numDBMap.size() == 0) {
        ALOG_INFO_F("[L1CopyReuse] Init Param default.");
        return SUCCESS;
    }
    int colorMax{0};
    std::set<int> colorSet;
    auto opOriList = func.Operations();
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetOpcode() == Opcode::OP_COPY_IN && opOriList[i].GetOOperands()[0]->GetMemoryTypeOriginal() == MEM_L1) {
            auto feature = GetGMInputFeature(opOriList[i]);
            if (feature.size() == 0) {
                ALOG_ERROR_F("[L1CopyReuse] Get Feature FAILED.");
                return FAILED;
            }
            ALOG_INFO_F("[L1CopyReuse] Op %d feature: %s", i, IntVecToStr(feature).c_str());
        }
        auto opColor = opOriList[i].GetSubgraphID();
        colorSet.insert(opColor);
        if (opColor > colorMax) {
            colorMax = opColor;
        }
    }
    int color = colorMax + 1;
    std::vector<std::vector<int>> colorNode(color);
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].IsNOP()) {
            continue;
        }
        auto opColor = opOriList[i].GetSubgraphID();
        colorNode[opColor].push_back(i);
    }
    std::vector<Operation *> opList;
    for (auto &op : func.Operations()) {
        opList.emplace_back(&op);
    }
    auto inOutGraph = RescheduleUtils::GetInOutGraphs(opList, func.GetFuncMagic());
    auto &inGraph = inOutGraph[0];
    L1CopyInReuseRunner runner(inGraph);
    runner.Run(func, color, colorNode);
    return SUCCESS;
}
}  // namespace npu::tile_fwk
