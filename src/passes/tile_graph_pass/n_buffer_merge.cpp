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
 * \file n_buffer_merge.cpp
 * \brief
 */

#include "passes/tile_graph_pass/n_buffer_merge.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "interface/utils/log.h"
#include "passes/pass_utils/parallel_tool.h"
#include "passes/pass_config/pass_config_manager.h"

namespace npu::tile_fwk {

void NBufferMerge::GetOpHash(std::vector<uint64_t> &hashList, const std::string op, int idx) {
    uint64_t a = 0x12345678;
    uint64_t p = 37;
    uint64_t hash = 0;
    for (char c : op) {
        hash = hash * p + static_cast<uint64_t>(c);
    }
    for (int j : inGraph_[idx]) {
        hash = hash * p + (hashList[j] ^ a);
    }
    hashList[idx] = hash;
}

void NBufferMerge::GetOpHashReverse(std::vector<uint64_t> &hashList, const std::string op, int idx) {
    uint64_t a = 0x12345678;
    uint64_t p = 37;
    uint64_t hash = 0;
    for (char c : op) {
        hash = hash * p + static_cast<uint64_t>(c);
    }
    for (int j : outGraph_[idx]) {
        hash = hash * p + (hashList[j] ^ a);
    }
    hashList[idx] = hash;
}

void UpdateOpColor(OperationsViewer &opOriList, 
                   int &color, 
                   std::vector<int> &colorCycles,
                   std::vector<std::vector<int>> &colorNode) {
    std::vector<int> oriColor2NewColor(color);
    int colorCount = 0;
    for (int i = 0; i < color; i++) {
        if (colorCycles[i] != 0) {
            oriColor2NewColor[i] = colorCount;
            colorCount++;
        }
        colorCycles[i] = 0;
        colorNode[i].clear();
    }
    color = colorCount;
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        opOriList[i].UpdateSubgraphID(oriColor2NewColor[opOriList[i].GetSubgraphID()]);
    }
}

Status NBufferMerge::ColorTopo(int &color1, 
                                   std::vector<std::vector<int>> &inputColor, 
                                   std::vector<std::vector<int>> &outputColor, 
                                   OperationsViewer &opOriList) {
    std::vector<int> colorQueue(color1);
    std::vector<int> colorInDegree(color1);
    int colorQueueHead = 0;
    int colorQueueTail = 0;
    for (int i = 0; i < color1; i++) {
        colorInDegree[i] = inputColor[i].size();
        if (colorInDegree[i] == 0) {
            // 找到入度为0的color作为queue的起始点
            colorQueue[colorQueueTail++] = i;
        }
    }
    // 从入度为0的点开始，不断解依赖，如果没有成环，那么遍历完所有color，不应该存在入度不为0的color
    while (colorQueueHead < colorQueueTail) {
        int i = colorQueue[colorQueueHead++];
        for (int j : outputColor[i]) {
            colorInDegree[j]--;
            if (colorInDegree[j] == 0) {
                 colorQueue[colorQueueTail++] = j;
            }
        }
    }
    // 这里1.0中遍历了前color个节点，这里修改成遍历所有的color
    for (int i = 0; i < color1; i++) {
        if (colorInDegree[i] != 0) {
            ALOG_ERROR_F("[NBUFFER_MERGE] Color [%d] has cycle in graph", i);
            return FAILED;
        }
    }
    std::vector<int> colorQueueReverse(color1);
    for (int i = 0; i < color1; i++) {
        colorQueueReverse[colorQueue[i]] = i;
    }
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        opOriList[i].UpdateSubgraphID(colorQueueReverse[opOriList[i].GetSubgraphID()]);
    }
    return SUCCESS;
}

Status NBufferMerge::CheckAndFixColorOrder(OperationsViewer &opOriList, 
                                            int &color1, 
                                            std::vector<int> &colorCycles1,
                                            std::vector<std::vector<int>> &colorNode1) {
    UpdateOpColor(opOriList, color1, colorCycles1, colorNode1);
    // 颜色拓扑排序
    std::vector<std::vector<int>> inputColor(color1);
    std::vector<std::vector<int>> outputColor(color1);
    for (size_t i = 0; i < opOriList.size(); i++) {
        for (int j : outGraph_[i]) {
            if (opOriList[i].GetSubgraphID() < 0 || opOriList[j].GetSubgraphID() < 0) {
                continue;
            }
            if (opOriList[i].GetSubgraphID() != opOriList[j].GetSubgraphID()) {
                outputColor[opOriList[i].GetSubgraphID()].push_back(opOriList[j].GetSubgraphID());
                inputColor[opOriList[j].GetSubgraphID()].push_back(opOriList[i].GetSubgraphID());
            }
        }
    }
    if (ColorTopo(color1, inputColor, outputColor, opOriList) == FAILED) {
        return FAILED;
    }
    // 重新统计colorNode等
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        colorCycles1[opOriList[i].GetSubgraphID()] += opOriList[i].GetLatency();
        colorNode1[opOriList[i].GetSubgraphID()].push_back(i);
    }
    return SUCCESS;
}

void NBufferMerge::InitParam(OperationsViewer &opOriList) {
    std::vector<std::mutex> subgraphMtx(color_);
    std::vector<std::mutex> inColorMtx(color_);
    std::vector<std::mutex> outColorMtx(color_);
    ParallelTool::Instance().Parallel_for(0, opOriList.size(),1,[&](int st,int et,int tid) {
        (void) tid;
        for (int i = st; i < et; i++) {
            // 过滤FromInCast节点和NOP节点
            if (opOriList[i].GetSubgraphID() < 0) {
                continue;
            }
            int subgraphId = opOriList[i].GetSubgraphID();
            {
                std::unique_lock lock(subgraphMtx.at(subgraphId));
                colorCycles_[subgraphId] += opOriList[i].GetLatency();
                colorNode_[subgraphId].push_back(i);
            }
            for (auto inputNode : opOriList[i].ProducerOps()) {
                auto parentColor = inputNode->GetSubgraphID();
                auto currentColor = opOriList[i].GetSubgraphID();
                if (parentColor != -1 && parentColor != currentColor) {
                    {
                        std::unique_lock lock(inColorMtx[currentColor]);
                        inColor_[currentColor].push_back(parentColor);
                    }
                    {
                        std::unique_lock lock(outColorMtx[parentColor]);
                        outColor_[parentColor].push_back(currentColor);
                    }
                }
            }
        }
    });
}

Status NBufferMerge::Init(Function &func) {
    size_t colorMax{0U};
    std::set<int> colorSet;
    auto opOriList = func.Operations();
    for (size_t i = 0; i < opOriList.size(); i++) {
        if (opOriList[i].GetSubgraphID() < 0) {
            continue;
        }
        colorSet.insert(opOriList[i].GetSubgraphID());
        if (opOriList[i].GetSubgraphID() > static_cast<int>(colorMax)) {
            colorMax = opOriList[i].GetSubgraphID();
        }
    }
    if (colorSet.size() == 0) {
        ALOG_INFO_F("[NBUFFER_MERGE] color size is 0, skip nbuffer merge.");
        return SUCCESS;
    }
    if (colorSet.size() != colorMax + 1) {
        ALOG_ERROR_F("[NBUFFER_MERGE] colors are not continously numbered from 0");
        return FAILED;
    }
    color_ = colorMax + 1;
    colorNode_.resize(color_);
    colorCycles_.resize(color_, 0);
    inColor_.resize(color_);
    outColor_.resize(color_);
    InitParam(opOriList);
    std::vector<Operation *> opList;
    for (auto &op : func.Operations()) {
        opList.emplace_back(&op);
    }
    auto inOutGraph = RescheduleUtils::GetInOutGraphs(opList, func.GetFuncMagic());
    inGraph_ = inOutGraph[0];
    outGraph_ = inOutGraph[1];
    ALOG_INFO_F("[NBUFFER_MERGE] Before Nbuffer merge");
    RescheduleUtils::PrintColorNode(func);
    return SUCCESS;
}

std::map<uint64_t, size_t> NBufferMerge::GetIsoColorMergeNum(const OperationsViewer &opOriList,
    const std::map<uint64_t, std::vector<int>> &hashMap) const {
    std::map<uint64_t, size_t> hashCoreNum;
    for (auto& entry : hashMap) {
        if (entry.first == 0 || entry.second.empty()) {
            continue;
        }
        auto subGraphIdx = entry.second.front();
        for (auto& opIdx : colorNode_[subGraphIdx]) {
            if (opOriList[opIdx].HasAttr(OpAttributeKey::isCube) &&
                opOriList[opIdx].GetBoolAttribute(OpAttributeKey::isCube)) {
                hashCoreNum[entry.first] = PassConfigManager::Instance().GetPlatformConfig().GetCoreNum(NpuCoreType::AICORE);
                break;
            }
        }
        if (hashCoreNum.find(entry.first) == hashCoreNum.end()) {
            hashCoreNum[entry.first] = PassConfigManager::Instance().GetPlatformConfig().GetCoreNum(NpuCoreType::VECTORCORE);
        }
        ALOG_INFO_F("[NBUFFER_MERGE] Subgraph hash: %lu, size %zu, core num: %zu.", 
                    entry.first, entry.second.size(), hashCoreNum[entry.first]);
        if (entry.second.size() <= hashCoreNum[entry.first]) {
            hashCoreNum[entry.first] = 1U;
            continue;
        }
        auto initNum = (entry.second.size() + hashCoreNum[entry.first] - 1) / hashCoreNum[entry.first];
        auto usedCore = (entry.second.size() + initNum - 1) / initNum;
        while ((usedCore < hashCoreNum[entry.first]) && (initNum > 1)) {
            initNum--;
            usedCore = (entry.second.size() + initNum - 1) / initNum;
        }
        hashCoreNum[entry.first] = initNum;
        ALOG_INFO_F("[NBUFFER_MERGE] Subgraph hash: %lu, merge num: %zu.", entry.first, hashCoreNum[entry.first]);
    }
    return hashCoreNum;
}

void NBufferMerge::GetColorHash(const OperationsViewer &opOriList, 
                                 std::vector<uint64_t> &hashColor,
                                 std::map<uint64_t, std::vector<int>> &hashMap) {
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
        // 单独的reshape不用合并
        if (opOriList[i].GetOpcode() == Opcode::OP_RESHAPE) {
            hashColor[opOriList[i].GetSubgraphID()] = 0;
            continue;
        }
        if (opOriList[i].HasAttr(OpAttributeKey::isCube) && (opOriList[i].GetBoolAttribute(OpAttributeKey::isCube))) {
            mulaccGraph.insert(opOriList[i].GetSubgraphID());
            continue;
        }
        hashColor[opOriList[i].GetSubgraphID()] = hashColor[opOriList[i].GetSubgraphID()] * p + (hashTileOp[i] ^ a);
    }
    for (auto subgraphId : mulaccGraph) {
        hashColor[subgraphId] = 0;
    }
    for (int i = 0; i < color_; i++) {
        hashMap[hashColor[i]].push_back(i);
    }
}

inline int GetCopyIn(const OperationsViewer &opOriList, std::vector<int> &colorNode) {
    // 获取子图CopyIn数据量
    int colorCopyIn = 0;
    for (int j : colorNode) {
        if (opOriList[j].GetOpcode() == Opcode::OP_COPY_IN) { //getopcode == opcode::OP_COPY_IN
            int volume = BytesOf(opOriList[j].GetOOperands()[0]->Datatype());
            std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(opOriList[j].GetOpAttribute());
            if (attr == nullptr) {
                ALOG_ERROR_F("[NBUFFER_MERGE] CopyOpAttribute is nullptr.");
                return -1;
            }
            auto shape = attr->GetSpecifiedShape(1);
            for (int k : shape) {
                volume *= k;
            }
            colorCopyIn = colorCopyIn + volume;
        }
    }
    return colorCopyIn;
}

std::vector<std::vector<int>> NBufferMerge::SortColorWithInput(std::vector<int> &colorValues) const {
    std::map<int, std::vector<int>> inColorToOutColor;
    int inCount = -1;
    for (auto color : colorValues) {
        if (inColor_[color].empty()) {
            inColorToOutColor[inCount--].push_back(color);
            continue;
        }
        for (auto inColor : inColor_[color]) {
            inColorToOutColor[inColor].push_back(color);
        }
    }
    std::map<int, std::vector<int>> outColorToInColor;
    int outCount = -1;
    for (auto color : colorValues) {
        if (outColor_[color].empty()) {
            outColorToInColor[outCount--].push_back(color);
            continue;
        }
        for (auto outColor : outColor_[color]) {
            outColorToInColor[outColor].push_back(color);
        }
    }
    std::vector<std::vector<int>> res;
    std::map<int, std::vector<int>> colorWithSameInOut = 
        (inColorToOutColor.size() <= outColorToInColor.size()) ? inColorToOutColor : outColorToInColor;
    std::set<int> visitedColorSet;
    for (auto &entry : colorWithSameInOut) {
        std::vector<int> sortedColor;
        for (auto subgraphColor : entry.second) {
            if (visitedColorSet.count(subgraphColor) == 0) {
                visitedColorSet.insert(subgraphColor);
                sortedColor.push_back(subgraphColor);
            }
        }
        if (!sortedColor.empty()) {
            res.push_back(sortedColor);
        }
    }
    return res;
}

void NBufferMerge::MergePingPong(std::vector<std::vector<int>> &sortedColors, 
                                     const OperationsViewer &opOriList, 
                                     std::vector<uint64_t> &hashColor, 
                                     std::map<uint64_t, size_t> &hashMergeNum, 
                                     uint64_t &colorHashValue) {
    int pingColor = -1;
    for (auto &input2Color : sortedColors) {
        for (size_t i = 0; i < input2Color.size(); i++) {
            if (i % hashMergeNum[colorHashValue] == 0) {
                pingColor = input2Color[i];
            } else {
                int pongColor = input2Color[i];
                for (auto opIdxMergedDB : colorNode_[pongColor]) {
                    opOriList[opIdxMergedDB].UpdateSubgraphID(pingColor);
                    colorNode_[pingColor].push_back(opIdxMergedDB);
                }
                colorCycles_[pingColor] += colorCycles_[pongColor];
                hashColor[pingColor] += hashColor[pongColor];
                colorCycles_[pongColor] = 0;
                colorNode_[pongColor].clear();
                hashColor[pongColor] = 0;
            }
        }
    }
}

Status NBufferMerge::MergeProcess(const OperationsViewer &opOriList, 
                                      std::map<uint64_t, std::vector<int>> &hashMap, 
                                      std::map<uint64_t, size_t> &hashMergeNum, 
                                      std::vector<uint64_t> &hashColor) {
    std::vector<uint64_t> hashMapKeys;
    for (auto &entry : hashMap) {
        hashMapKeys.push_back(entry.first);
    }
    ParallelTool::Instance().Parallel_for(0, hashMapKeys.size(),1,[&](int st,int et,int tid) {
        (void) tid;
        for(int hashMapKeyIdx = st; hashMapKeyIdx < et; hashMapKeyIdx++) {
            uint64_t colorHashValue = hashMapKeys[hashMapKeyIdx];
            if (colorHashValue == 0) continue;
            std::vector<int> &colorValues = hashMap[colorHashValue];
            auto sortedColors = SortColorWithInput(colorValues);
            if (sortedColors.empty()) continue;
            MergePingPong(sortedColors, opOriList, hashColor, hashMergeNum, colorHashValue);
        }
    });
    return SUCCESS;
}

Status NBufferMerge::NBufferMergeProcess(Function &func, int numDB) {
    if (Init(func) == FAILED) {
        ALOG_ERROR_F("[NBUFFER_MERGE] Init Failed.");
        return FAILED;
    }
    if (color_ == 0) {
        return SUCCESS;
    }
    // 如果子图个数已经少于核数； 后续按照core的类型来判断
    int coreNum = PassConfigManager::Instance().GetPlatformConfig().GetCoreNum(NpuCoreType::AICORE);
    if (color_ <= coreNum) {
        ALOG_INFO_F("[NBUFFER_MERGE] NBufferMerge is skipped. color: %d, aiCoreNum: %d", color_, coreNum);
        return SUCCESS;
    }
    ALOG_INFO_F("[NBUFFER_MERGE] User set nbuffer num: %d", numDB);
    // 获取节点和子图的hash
    auto opOriList = func.Operations();
    std::vector<uint64_t> hashColor(color_, 0);
    std::map<uint64_t, std::vector<int>> hashMap;
    GetColorHash(opOriList, hashColor, hashMap);
    std::map<uint64_t, size_t> hashMergeNum;
    if (numDB == 1) {
        ALOG_INFO_F("[NBUFFER_MERGE] Manually Set NumDB 1, Automatically Calculate MergeNum.");
        hashMergeNum = GetIsoColorMergeNum(opOriList, hashMap);
    } else {
        ALOG_INFO_F("[NBUFFER_MERGE] Manually Set NumDB %d.", numDB);
        for (auto& entry : hashMap) {
            hashMergeNum[entry.first] = numDB;
        }
    }
    if (MergeProcess(opOriList, hashMap, hashMergeNum, hashColor) == FAILED) {
        return FAILED;
    }
    if (CheckAndFixColorOrder(opOriList, color_, colorCycles_, colorNode_) == FAILED) {
        return FAILED;
    }
    func.SetTotalSubGraphCount(color_);
    ALOG_DEBUG_F("[NBUFFER_MERGE] After Nbuffer merge");
    RescheduleUtils::PrintColorNode(func);
    return SUCCESS;
}

Status NBufferMerge::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start NBufferMerge.");
    auto numDB = function.paramConfigs_.NbufferNum;
    ALOG_INFO_F("[NBUFFER_MERGE] nbuffer num: %d", numDB);
    if (numDB == 0) {
        ALOG_INFO_F("[NBUFFER_MERGE] Manually Set NumDB 0, Skip NBufferMerge.");
        return SUCCESS;
    }
    if (NBufferMergeProcess(function, numDB) == FAILED) {
        return FAILED;
    }
    ALOG_INFO_F("===> Finish NBufferMerge.");
    return SUCCESS;
}
}  // namespae npu::tile_fwk
