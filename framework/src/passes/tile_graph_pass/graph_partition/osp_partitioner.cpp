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
 * \file osp_partitioner.cpp
 * \brief
 */

#include "osp_partitioner.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_check/iso_partitioner_checker.h"

#include "passes/algorithms/osp/bsp/model/BspSchedule.hpp"
#include "passes/algorithms/osp/bsp/model/util/SetSchedule.hpp"
#include "passes/algorithms/osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "passes/algorithms/osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "passes/algorithms/osp/bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "passes/algorithms/osp/bsp/scheduler/GreedySchedulers/GreedyMetaScheduler.hpp"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/IsomorphicSubgraphScheduler.hpp"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/PrecomputedHashComputer.hpp"
#include "passes/algorithms/osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include.hpp"
#include "passes/algorithms/osp/coarser/sarkar/sarkar_mul.hpp"

#define MODULE_NAME "GraphPartition"

using namespace npu::tile_fwk;

Status OspPartitioner::BuildSuperNodeGraph()
{
    std::vector<Operation*> &opList = operationInfo_->opList_;
    if (opList.size() != operationInfo_->inGraph_.size() || opList.size() != operationInfo_->outGraph_.size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Osp: Operation inGraph and outGraph have not been initialized.");
        return FAILED;
    }
    std::vector<std::pair<int32_t, int32_t>> mergePair;
    for (size_t i = 0; i < opList.size(); i++) {
        if (ConvertCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (L1CopyInCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (AssembleCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (CopyOutCombine(operationInfo_, opList, i, mergePair, false)) {
            continue;
        }
        if (CopyInCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (MulAccCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (ExpandCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
    }
    superNodeInfo_ = std::make_shared<NodeGraphInfo>();
    if (superNodeInfo_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Osp: Create SuperNodeInfo failed.");
        return FAILED;
    }
    if (superNodeInfo_->Build(operationInfo_, mergePair, !useCVMixPartition_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Osp: Build SuperNodeInfo Failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OspPartitioner::PartitionGraph(Function &function)
{
    APASS_LOG_INFO_F(Elements::Function, "Running OspPartitioner, useCVMixPartition: %d.", useCVMixPartition_);
    if (BuildOpGraph(function.Operations().DuplicatedOpList()) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Partition the computational graph failed in building operation graph.");
        return FAILED;
    }
    if (BuildSuperNodeGraph() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Partition the computational graph failed in building SuperNode graph.");
        return FAILED;
    }    
    if (ospMode_ == OspMode::MERKLEBSP) {
        if (BuildHashValues() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Partition the computational graph failed in building SuperNode hash values.");
            return FAILED;
        }
    }
    if (RunOspPartition(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate a partition.");
        return FAILED;
    }
    return SUCCESS;
}

Status OspPartitioner::RunOspPartition(Function &function)
{
    Status status = FAILED;
    std::vector<osp::VertexIdxT<GraphType>> vertexContractionMap;
    CoarseGraphType coarseGraph;

    switch (ospMode_)
    {
        case OspMode::SARKAR:
        {
            GraphType graph;
            if (ConstructDag(graph) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate graph.");
                return FAILED;
            }
            status = RunSarkar(graph, coarseGraph, vertexContractionMap);
        }
        break;

        case OspMode::MERKLEBSP:
        {
            osp::BspInstance<GraphType> bspInstance;
            if (ConstructBspInstance(bspInstance) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate a bsp instance.");
                return FAILED;
            }
            status = RunMerkleBsp(bspInstance, vertexContractionMap);
        }
        break;

        default:
        {
            APASS_LOG_ERROR_F(Elements::Config, "OSP Coarsen mode not implemented.");
            status = FAILED;
        }
    }

    if (status == FAILED) {
        return FAILED;
    }

    if (UpdatePartitionResult(function, vertexContractionMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "OSP failed to update the partition result.");
        return FAILED;
    }

    return status;
}

Status OspPartitioner::RunSarkar(const GraphType &graph, CoarseGraphType &coarseGraph, std::vector<osp::VertexIdxT<GraphType>> &vertexContractionMap)
{
    osp::sarkar_params::MulParameters< osp::VWorkwT<GraphType> > params;
    params.seed_ = 1729U;
    params.geomDecay_ = 0.875;
    params.leniency_ = 0.005;
    params.commCostVec_ = std::vector<osp::VWorkwT<GraphType>>({1, 2, 5, 10, 20, 50, 100, 200, 500, 1000});
    params.maxWeight_ = archParameters_.partitionWorkUpperBound_;
    params.smallWeightThreshold_ = archParameters_.partitionWorkLowerBound_; 
    params.maxNumIterationWithoutChanges_ = 3U;
    params.bufferMergeMode_ = osp::sarkar_params::BufferMergeMode::FULL;

    osp::SarkarMul<GraphType, CoarseGraphType> coarser;
    coarser.SetParameters(params);

    bool coarsenStatus = coarser.CoarsenDag(graph, coarseGraph, vertexContractionMap);
    if (not coarsenStatus) {
        APASS_LOG_ERROR_F(Elements::Function, "OSP Sarkar failed to generate a coarse graph.");
        return FAILED;
    }

    return SUCCESS;
}

Status OspPartitioner::RunMerkleBsp(const osp::BspInstance<GraphType> &bspInst, std::vector<osp::VertexIdxT<GraphType>> &vertexContractionMap) {
    osp::GrowLocalAutoCores<ConstrGraphType> growlocal;
    osp::BspLocking<ConstrGraphType> locking;
    osp::GreedyChildren<ConstrGraphType> children;
    
    osp::KlTotalLambdaCommImprover<ConstrGraphType> kl(42);
    kl.SetSuperstepRemoveStrengthParameter(1.0);
    kl.SetTimeQualityParameter(1.0);
    
    osp::ComboScheduler<ConstrGraphType> growlocalKl(growlocal, kl);
    osp::ComboScheduler<ConstrGraphType> lockingKl(locking, kl);
    osp::ComboScheduler<ConstrGraphType> childrenKl(children, kl);

    osp::GreedyMetaScheduler<ConstrGraphType> scheduler;
    scheduler.AddScheduler(growlocalKl);
    scheduler.AddScheduler(lockingKl);
    scheduler.AddScheduler(childrenKl);
    scheduler.AddSerialScheduler();

    osp::MerkleHashComputer<GraphType, osp::PrecomBwdMerkleNodeHashFunc<GraphType>> hashComputer(bspInst.GetComputationalDag(), bspInst.GetComputationalDag(), this->superNodeInfo_->nodeHashList_);
    osp::IsomorphicSubgraphScheduler<GraphType, ConstrGraphType> isoScheduler(scheduler, hashComputer);
    isoScheduler.SetWorkThreshold(200);
    isoScheduler.SetCriticalPathThreshold(500);
    isoScheduler.SetOrbitLockRatio(0.5);
    isoScheduler.SetMergeDifferentTypes(false);
    isoScheduler.SetAllowTrimmedScheduler(false);
    isoScheduler.SetUseMaxBsp(false);
    vertexContractionMap = isoScheduler.ComputePartition(bspInst);  
    return SUCCESS;
}

Status OspPartitioner::UpdatePartitionResult(Function &function, std::vector<osp::VertexIdxT<GraphType>> &vertexContractionMap) 
{
    int32_t numColors = 0;
    for (size_t i = 0; i < vertexContractionMap.size(); ++i) {
        const int32_t superNode = static_cast<int32_t>(i);
        const int32_t nodeColor = vertexContractionMap[i];
        for (const int32_t idx : superNodeInfo_->node2Op_[superNode]) {
            operationInfo_->opList_[idx]->UpdateSubgraphID(nodeColor);
        }
        numColors = std::max(numColors, nodeColor + 1);
    }
    function.SetTotalSubGraphCount(numColors);
    return SUCCESS;
}

void OspPartitioner::SetVertexCommMemWeight(GraphType &graph, int32_t vertex)
{
    std::set<int32_t> operandsInChildSupernodes;
    const auto &children = superNodeInfo_->nodeOutGraphList_[vertex];
    for (const auto &child : children) {       
        const auto &childOperators = superNodeInfo_->node2Op_[child];

        for (const auto &op : childOperators) {
            operandsInChildSupernodes.insert(op);
        }
    }

    const auto &operators = superNodeInfo_->node2Op_[vertex]; 
    osp::VCommwT<GraphType> commWeight = 10;
    osp::VMemwT<GraphType> memWeight = 10;

    for (const auto &op : operators) {
        const auto operInfo = operationInfo_->opList_[op];
        const std::string opCode = operInfo->GetOpcodeStr();
        const bool isView = opCode.find("View") != std::string::npos;
        if (isView) {
            for (auto &inputLogicalTensor : operationInfo_->opList_[op]->GetIOperands()) {
                const size_t memorySize = inputLogicalTensor->MemorySize();
                memWeight += static_cast<osp::VMemwT<GraphType>>(memorySize);
            }
        }

        for (auto &outputLogicalTensor : operationInfo_->opList_[op]->GetOOperands()) {
            for (auto &consumer : outputLogicalTensor->GetConsumers()) {
                if (operationInfo_->magic2Idx_.count(consumer->GetOpMagic()) == 0) {
                    continue;
                }
                
                int32_t operationIdx = operationInfo_->magic2Idx_[consumer->GetOpMagic()];
                if (operandsInChildSupernodes.find(operationIdx) == operandsInChildSupernodes.end()) {
                    continue;
                }

                const size_t memorySize = outputLogicalTensor->MemorySize();
                commWeight += static_cast<osp::VCommwT<GraphType>>(memorySize);
                break;
            }
        }
    }
    graph.SetVertexMemWeight(vertex, memWeight);
    graph.SetVertexCommWeight(vertex, static_cast<osp::VCommwT<GraphType>>(commWeight * archParameters_.commCorrectionFactor_));
}

Status OspPartitioner::ConstructDagCVSplit(GraphType &graph)
{    
    if (ConstructDagCVMix(graph) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Construct CV Mixed Dag has failed.");
        return FAILED;
    }
    
    for (const auto &superNode : graph.Vertices()) {
        const OpCoreType vertexType = superNodeInfo_->nodeCoreType_[superNode];
        if (vertexType != OpCoreType::AIC && vertexType != OpCoreType::AIV && vertexType != OpCoreType::AICPU) {
            APASS_LOG_ERROR_F(Elements::Operation, "SuperNode (%d) has core type (%d) which is neither cube nor vector nor ai-scalar.", superNode, vertexType);
            return FAILED;
        }
        graph.SetVertexType(superNode, GetOspCoreTypeSplit(vertexType));
    }

    return SUCCESS;
}

Status OspPartitioner::ConstructDagCVMix(GraphType &graph)
{    
    graph = GraphType(superNodeInfo_->nodeOutGraphList_, superNodeInfo_->nodeInGraphList_);
    
    for (const auto &superNode : graph.Vertices()) {
        graph.SetVertexWorkWeight(superNode, superNodeInfo_->nodeCycles_[superNode]); 
        SetVertexCommMemWeight(graph, superNode);

        OpCoreType vertexType = superNodeInfo_->nodeCoreType_[superNode];
        if (vertexType != OpCoreType::AIC && vertexType != OpCoreType::AIV && vertexType != OpCoreType::AICPU) {
            APASS_LOG_ERROR_F(Elements::Operation, "SuperNode (%d) has core type (%d) which is neither cube nor vector nor ai-scalar.", superNode, vertexType);
            return FAILED;
        }
        graph.SetVertexType(superNode, GetOspCoreTypeMix(vertexType));
    }

    return SUCCESS;
}

void OspPartitioner::ConstructBspArchCVSplit(osp::BspArchitecture<GraphType> &bspArch)
{
    const size_t numCubeCores = Platform::Instance().GetSoc().GetAICCoreNum();
    const size_t numVectorCores = Platform::Instance().GetSoc().GetAIVCoreNum();
    const size_t numAiScalarCores = Platform::Instance().GetSoc().GetAICPUNum();

    const size_t numCores = numCubeCores + numVectorCores + numAiScalarCores;
    std::vector<osp::VTypeT<GraphType>> procTypes(numCores);
    std::vector<osp::VWorkwT<GraphType>> procMemoryBound(numCores);
    
    for (size_t i = 0; i < numCores; i++) {
        if (i < numCubeCores) { // Cube Cores
            procTypes[i] = GetOspCoreTypeSplit(OpCoreType::AIC);
            procMemoryBound[i] = static_cast<osp::VWorkwT<GraphType>>( Platform::Instance().GetAICCore().GetMemorySize(MemoryType::MEM_L1) );
        } else if (i < numCubeCores + numVectorCores) { // Vector Cores
            procTypes[i] = GetOspCoreTypeSplit(OpCoreType::AIV);
            procMemoryBound[i] = static_cast<osp::VWorkwT<GraphType>>( Platform::Instance().GetAIVCore().GetMemorySize(MemoryType::MEM_UB) );
        } else { // AI Scalar Cores
            procTypes[i] = GetOspCoreTypeSplit(OpCoreType::AICPU);
            procMemoryBound[i] = std::numeric_limits< osp::VWorkwT<GraphType> >::max();
        }
    }
    bspArch.SetProcessorsWithTypes(procTypes);
    bspArch.SetMemoryBound(procMemoryBound);
    bspArch.SetCommunicationCosts(archParameters_.commCost_);
    bspArch.SetSynchronisationCosts(archParameters_.synchCost_);
}

Status OspPartitioner::ConstructBspArchCVMix(osp::BspArchitecture<GraphType> &bspArch)
{
    const size_t numCubeCores = Platform::Instance().GetSoc().GetAICCoreNum();
    const size_t numVectorCores = Platform::Instance().GetSoc().GetAIVCoreNum();
    const size_t numAiScalarCores = Platform::Instance().GetSoc().GetAICPUNum();

    if (numCubeCores == 0 || ((numVectorCores % numCubeCores) != 0)) {
        APASS_LOG_ERROR_F(Elements::Config, "OSP bsp architecture does not satisfy an 1:N ratio of Cube:Vector cores.");
        return FAILED;
    }

    const size_t numCores = (numCubeCores != 0U ? numCubeCores : numVectorCores) + numAiScalarCores;
    const size_t numVecPerCube = (numCubeCores != 0U ? numVectorCores / numCubeCores : 1U);

    std::vector<osp::VTypeT<GraphType>> procTypes(numCores);
    std::vector<osp::VWorkwT<GraphType>> procMemoryBound(numCores);

    const size_t cubeVecMemoryBound = ((numCubeCores != 0U ? 1U : 0U)) * Platform::Instance().GetAICCore().GetMemorySize(MemoryType::MEM_L1)
                                    + (numVecPerCube * Platform::Instance().GetAIVCore().GetMemorySize(MemoryType::MEM_UB));

    for (size_t i = 0; i < numCores; i++) {
        if (i < numCubeCores) { // Cube Vector Core Mix
            procTypes[i] = GetOspCoreTypeMix(OpCoreType::AIC);
            procMemoryBound[i] = static_cast<osp::VWorkwT<GraphType>>( cubeVecMemoryBound );
        } else { // AI Scalar Cores
            procTypes[i] = GetOspCoreTypeMix(OpCoreType::AICPU);
            procMemoryBound[i] = std::numeric_limits< osp::VWorkwT<GraphType> >::max();
        }
    }
    
    bspArch.SetProcessorsWithTypes(procTypes);
    bspArch.SetMemoryBound(procMemoryBound);
    bspArch.SetCommunicationCosts(archParameters_.commCost_);
    bspArch.SetSynchronisationCosts(archParameters_.synchCost_);

    return SUCCESS;
}

Status OspPartitioner::ConstructDag(GraphType &graph)
{
    if (useCVMixPartition_) {
        ConstructDagCVMix(graph);
    } else {
        if (ConstructDagCVSplit(graph) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate graph with CV split.");
            return FAILED;
        };
    }
    return SUCCESS;
}

Status OspPartitioner::ConstructBspInstance(osp::BspInstance<GraphType> &bspInst)
{
    if (useCVMixPartition_) {
        if (ConstructBspArchCVMix(bspInst.GetArchitecture()) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate bsp architecture with CV mix.");
            return FAILED;
        }
        ConstructDagCVMix(bspInst.GetComputationalDag());
    } else {
        ConstructBspArchCVSplit(bspInst.GetArchitecture());
        if (ConstructDagCVSplit(bspInst.GetComputationalDag()) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate graph with CV split.");
            return FAILED;
        };
    }
    unsigned numTypes = std::max(bspInst.GetArchitecture().GetNumberOfProcessorTypes(), static_cast<unsigned>( bspInst.GetComputationalDag().NumVertexTypes()));
    bspInst.SetDiagonalCompatibilityMatrix(numTypes);
    return SUCCESS;
}

uint64_t OspPartitioner::CombineHash(const uint64_t h1, const uint64_t h2) const
{
    constexpr uint64_t magic = 0x9e3779b9;
    constexpr uint64_t numSix = 6;
    constexpr uint64_t numTwo = 2;

    uint64_t seed = h1;
    seed ^= h2 + magic + (seed << numSix) + (seed >> numTwo);
    return seed;
}

Status OspPartitioner::BuildHashValues()
{
    constexpr uint64_t numThree = 3U;
    constexpr std::size_t numEleven = 11U;

    std::vector<uint64_t> opHashList;
    std::vector<uint64_t> opHashListFront(operationInfo_->opList_.size(), 0);
    std::vector<uint64_t> opHashListBack(operationInfo_->opList_.size(), numThree);
    std::vector<uint64_t> opHashListFrontBack(operationInfo_->opList_.size(), 0);
    for (size_t i = 0; i < operationInfo_->opList_.size(); i++) {
        opHashListFront[i] = operationInfo_->opHashList_[i];
        
        std::vector<uint64_t> hashes;
        for (int32_t j : operationInfo_->inGraph_[i]) {
            hashes.push_back(opHashListFront[j]);
        }
        std::sort(hashes.begin(), hashes.end());
        for (int32_t j = 0; j < static_cast<int32_t>(hashes.size()); j++) {
            opHashListFront[i] = CombineHash(opHashListFront[i], hashes[j]);
        }
    }
    for (int32_t i = static_cast<int32_t>(operationInfo_->opList_.size() - 1); i >= 0; i--) {
        std::vector<uint64_t> hashes;
        for (int32_t j : operationInfo_->outGraph_[i]) {
            hashes.push_back(opHashListBack[j]); 
        }

        std::sort(hashes.begin(), hashes.end());
        for (int32_t j = 0; j < static_cast<int32_t>(hashes.size()); j++) {
            opHashListBack[i] = CombineHash(opHashListBack[i], hashes[j]);
        }
    }
    for (size_t i = 0; i < operationInfo_->opList_.size(); i++) {
        opHashListFrontBack[i] = CombineHash(opHashListFront[i], opHashListBack[i]);
    }
    opHashList.swap(opHashListFrontBack);
    
    if (superNodeInfo_->op2Node_.size() != operationInfo_->opList_.size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation number mismatch in SuperNodeInfo and OperationInfo.");
        return FAILED;
    }
    int32_t numNode = superNodeInfo_->node2Op_.size();
    superNodeInfo_->nodeHashList_.resize(numNode);
    for (int32_t i = 0; i < numNode; i++) {
        superNodeInfo_->nodeHashList_[i] = numEleven;
        std::vector<uint64_t> hashes;
        for (int32_t opIdx : superNodeInfo_->node2Op_[i]) {
            hashes.push_back(opHashList[opIdx]);
        }
        std::sort(hashes.begin(), hashes.end());
        
        for (int32_t j = 0; j < static_cast<int32_t>(hashes.size()); j++) {
            superNodeInfo_->nodeHashList_[i] = CombineHash(superNodeInfo_->nodeHashList_[i], hashes[j]);
        }
    }
    for (int32_t i = 0; i < numNode; i++) {
        superNodeInfo_->hash2NodeMap_[superNodeInfo_->nodeHashList_[i]].push_back(i);
    }
    return SUCCESS;
}