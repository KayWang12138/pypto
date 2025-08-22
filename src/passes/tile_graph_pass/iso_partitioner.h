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
 * \file iso_partitioner.h
 * \brief
 */

#ifndef PASS_ISO_PARTITIONER_H
#define PASS_ISO_PARTITIONER_H
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {

enum class GraphExtendResult { EXTEND_SUCCESS, EXTEND_LINK_EXHAUST, EXTEND_NODE_EXHAUST };

class OperationGraphInfo {
public:
    uint64_t GetHash(const Operation *op) const;
    Status Build(std::vector<Operation*> &opList);
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

class SubGraph {
public:
    SubGraph(std::shared_ptr<OperationGraphInfo> operationInfo, std::shared_ptr<NodeGraphInfo> superNodeInfo)
        : operationInfo_(operationInfo), superNodeInfo_(superNodeInfo)
    {}
    int32_t GetExpandCandidate(size_t expandNodeIdx, size_t expandLinkIdx, GraphExtendResult &res);
    void AddNode(int32_t nodeIdx);
    void Merge(SubGraph *sg);
    bool HasNode(int32_t nodeIdx) const;
    void BuildInOutSet();
    int32_t GetLatency() const;
    void Clear();
    std::string DumpStr();
    const std::vector<int32_t> &GetNodeList();
    std::vector<Operation*> GetOpList();
    std::shared_ptr<OperationGraphInfo> operationInfo_;
    std::shared_ptr<NodeGraphInfo> superNodeInfo_;
    std::vector<int32_t> nodeList_;
    std::unordered_set<int32_t> nodeSet_;
    std::unordered_set<int32_t> inNodes_;
    std::unordered_set<int32_t> outNodes_;
    std::set<std::pair<int32_t, int32_t>> mergeHistoryIsoSub_;
    int32_t cycle_{0};
    OpCoreType coreType_{OpCoreType::ANY};
    bool mergeable_{true};
};

class IsomorphismGraphGroup {
public:
    Status BuildGraphGroup(std::shared_ptr<OperationGraphInfo> operationInfo,
                         std::shared_ptr<NodeGraphInfo> superNodeInfo, std::vector<int32_t> &expandCandidate,
                         std::unordered_set<int32_t> &currentNodeSet, std::vector<int32_t> &idxInLinkNum,
                         std::deque<int32_t> &zeroInQueue);
    Status ExpandIsoGraphs(std::unordered_set<int32_t> &currentNodeSet, std::vector<int32_t> &idxInLinkNum,
                         std::deque<int32_t> &zeroInQueue, int32_t cycleUpperBound);
    static bool IsoGraphMerge(std::shared_ptr<IsomorphismGraphGroup> &currGraph,
                              std::shared_ptr<IsomorphismGraphGroup> &mergeGraph,
                              std::vector<std::pair<int32_t, int32_t>> &isoSubIdxs);
    size_t Size() const;
    void Clear();
    bool GetMergeable();
    int32_t GetLatency() const;
    Status InLinkCountDelete(int32_t nodeIdx, std::vector<int32_t> &idxInLinkNum, std::deque<int32_t> &zeroInQueue);
    bool IsLegalIsoGraphExtender(std::vector<int32_t> &expandCandidate, std::unordered_set<int32_t> &currentNodeSet,
                                 std::vector<int32_t> &idxInLinkNum, int32_t cycleUpperBound);
    bool IsLegalSubGraphMerge(SubGraph *sg1, SubGraph *sg2);
    std::shared_ptr<SubGraph> GetSubGraph(int32_t idx);
    std::vector<std::shared_ptr<SubGraph>> isoGraphs_;
    std::unordered_set<int32_t> subVisitedNodeSet_;
    bool mergeable_;
    std::shared_ptr<OperationGraphInfo> operationInfo_;
    std::shared_ptr<NodeGraphInfo> superNodeInfo_;
};

class IsoPartitioner {
public:
    Status PartitionGraph(Function &function);
    Status SetParameter(int32_t cycleUpperBound, int32_t parallelNum, int32_t cycleLowerBound, 
                        bool useReduceBalanceHash);

private:
    Status BuildOpGraph(const std::vector<Operation*> &opList);
    Status BuildSuperNodeGraph();
    Status BuildHashValues();
    Status BuildIsomorphismGroups();
    std::vector<std::pair<int32_t, int32_t>> GetReduceNodeMergePair() const;
    Status BuildReduceNodeHash(std::shared_ptr<NodeGraphInfo> reduceNodeInfo);
    Status BuildBalanceOpHash(std::vector<uint64_t> &opHashList);
    Status IsomorphismGroupMergeStep(bool nonIsoGraphsMerge);
    Status IsomorphismGroupMergeProcess(bool nonIsoGraphsMerge);
    Status UpdatePartitionResult(Function &function);
    Status IsomorphismGroupMergePrepare(std::vector<std::pair<int32_t, int32_t>> &isoSubIdxs,
                                      std::vector<std::set<int32_t>> &isoInGraph,
                                      std::vector<std::set<int32_t>> &isoOutGraph,
                                      std::vector<std::vector<int32_t>> &isoNodeList,
                                      std::vector<int32_t> &isoIdx2color);
    std::vector<int32_t> GetCandidateMergeColors(int32_t currColor, std::vector<std::set<int32_t>> &isoInGraph,
                                                 std::vector<std::set<int32_t>> &isoOutGraph,
                                                 std::vector<std::vector<int32_t>> &isoNodeList,
                                                 std::vector<int32_t> &isoIdx2color, bool nonIsoGraphsMerge);
    bool SuitableForMergeCheck(int32_t currColor, int32_t mergeColor, bool nonIsoGraphsMerge) const;
    uint64_t CombineHash(const uint64_t h1, const uint64_t h2) const;
    std::shared_ptr<OperationGraphInfo> operationInfo_;
    std::shared_ptr<NodeGraphInfo> superNodeInfo_;
    std::vector<std::shared_ptr<IsomorphismGraphGroup>> isoSubGroups_;
    int32_t tryMergeLoopNum_ = 100;
    bool useReduceBalanceHash_ = true;
    bool useCVMixPartition_ = false;
    int32_t cycleUB_ = -1;
    int32_t parallelNum_ = -1;
    int32_t cycleLB_ = -1;
};

class GraphPartition : public Pass {
public:
    GraphPartition() : Pass("GraphPartition")
    {}
    ~GraphPartition() override = default;
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status PostOperationCheck(Function &function);
    Status PostSubgraphCheck(const std::vector<std::vector<Operation*>> &subgraphs);
    Status RunOnFunction(Function &function) override;
};
}  // namespace npu::tile_fwk
#endif  // PASS_ISO_PARTITIONER_H