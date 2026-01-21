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

    osp::BspInstance<GraphType> bspInstance;
    if (ConstructBspInstance(bspInstance) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate a bsp instance.");
        return FAILED;
    }
    if (RunOspPartition(function, bspInstance) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate a partition.");
        return FAILED;
    }
    return SUCCESS;
}

Status OspPartitioner::RunOspPartition(Function &function, const osp::BspInstance<GraphType> &bspInst)
{
    Status status = FAILED;
    std::vector<osp::vertex_idx_t<GraphType>> vertexContractionMap;
    CoarseGraphType coarseGraph;

    switch (ospMode_)
    {
        case OspMode::SARKAR:
        {
            status = RunSarkar(bspInst, coarseGraph, vertexContractionMap);
        }
        break;

        case OspMode::MERKLEBSP:
        {
            status = RunMerkleBsp(bspInst, vertexContractionMap);
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

Status OspPartitioner::RunSarkar(const osp::BspInstance<GraphType> &bspInst, CoarseGraphType &coarseGraph, std::vector<osp::vertex_idx_t<GraphType>> &vertexContractionMap)
{
    osp::SarkarParams::MulParameters< osp::v_workw_t<GraphType> > params;
    params.seed = 1729U;
    params.geomDecay = 0.875;
    params.leniency = 0.005;
    params.commCostVec = std::vector<osp::v_workw_t<GraphType>>({1, 2, 5, 10, 20, 50, 100, 200, 500, 1000});
    params.maxWeight = archParameters_.partitionWorkUpperBound_;
    params.smallWeightThreshold = archParameters_.partitionWorkLowerBound_; 
    params.max_num_iteration_without_changes = 3U;
    params.buffer_merge_mode = osp::SarkarParams::BufferMergeMode::FULL;

    osp::SarkarMul<GraphType, CoarseGraphType> coarser;
    coarser.setParameters(params);

    bool coarsen_status = coarser.coarsenDag(bspInst.getComputationalDag(), coarseGraph, vertexContractionMap);
    if (not coarsen_status) {
        APASS_LOG_ERROR_F(Elements::Function, "OSP Sarkar failed to generate a coarse graph.");
        return FAILED;
    }

    return SUCCESS;
}

Status OspPartitioner::RunMerkleBsp(const osp::BspInstance<GraphType> &bspInst, std::vector<osp::vertex_idx_t<GraphType>> &vertexContractionMap) {
    osp::GrowLocalAutoCores<ConstrGraphType> growlocal;
    osp::BspLocking<ConstrGraphType> locking;
    osp::GreedyChildren<ConstrGraphType> children;
    
    osp::kl_total_lambda_comm_improver<ConstrGraphType> kl(42);
    kl.setSuperstepRemoveStrengthParameter(1.0);
    kl.setTimeQualityParameter(1.0);
    
    osp::ComboScheduler<ConstrGraphType> growlocal_kl(growlocal, kl);
    osp::ComboScheduler<ConstrGraphType> locking_kl(locking, kl);
    osp::ComboScheduler<ConstrGraphType> children_kl(children, kl);

    osp::GreedyMetaScheduler<ConstrGraphType> scheduler;
    scheduler.addScheduler(growlocal_kl);
    scheduler.addScheduler(locking_kl);
    scheduler.addScheduler(children_kl);
    scheduler.addSerialScheduler();

    osp::MerkleHashComputer<GraphType, osp::precom_bwd_merkle_node_hash_func<GraphType>> hashComputer(bspInst.getComputationalDag(), bspInst.getComputationalDag(), this->superNodeInfo_->nodeHashList_);
    osp::IsomorphicSubgraphScheduler<GraphType, ConstrGraphType> isoScheduler(scheduler, hashComputer);
    isoScheduler.setWorkThreshold(200);
    isoScheduler.setCriticalPathThreshold(500);
    isoScheduler.setOrbitLockRatio(0.5);
    isoScheduler.setMergeDifferentTypes(false);
    isoScheduler.setAllowTrimmedScheduler(false);
    isoScheduler.setUseMaxBsp(false);
    vertexContractionMap = isoScheduler.compute_partition(bspInst);  
    return SUCCESS;
}

Status OspPartitioner::UpdatePartitionResult(Function &function, std::vector<osp::vertex_idx_t<GraphType>> &vertexContractionMap) 
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
    osp::v_commw_t<GraphType> commWeight = 10;
    osp::v_memw_t<GraphType> memWeight = 10;

    for (const auto &op : operators) {
        const auto operInfo = operationInfo_->opList_[op];
        const std::string opCode = operInfo->GetOpcodeStr();
        const bool isView = opCode.find("View") != std::string::npos;
        if (isView) {
            for (auto &inputLogicalTensor : operationInfo_->opList_[op]->GetIOperands()) {
                const size_t memorySize = inputLogicalTensor->MemorySize();
                memWeight += static_cast<osp::v_memw_t<GraphType>>(memorySize);
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
                commWeight += static_cast<osp::v_commw_t<GraphType>>(memorySize);
                break;
            }
        }
    }

    graph.set_vertex_mem_weight(vertex, memWeight);
    graph.set_vertex_comm_weight(vertex, static_cast<osp::v_commw_t<GraphType>>(commWeight * archParameters_.commCorrectionFactor));
}

Status OspPartitioner::ConstructDagCVSplit(GraphType &graph)
{    
    if (ConstructDagCVMix(graph) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Construct CV Mixed Dag has failed.");
        return FAILED;
    }
    
    for (const auto &superNode : graph.vertices()) {
        const OpCoreType vertexType = superNodeInfo_->nodeCoreType_[superNode];
        if (vertexType != OpCoreType::AIC && vertexType != OpCoreType::AIV && vertexType != OpCoreType::AICPU) {
            APASS_LOG_ERROR_F(Elements::Operation, "SuperNode (%d) has core type (%d) which is neither cube nor vector nor ai-scalar.", superNode, vertexType);
            return FAILED;
        }
        graph.set_vertex_type(superNode, getOspCoreTypeSplit(vertexType));
    }

    return SUCCESS;
}

Status OspPartitioner::ConstructDagCVMix(GraphType &graph)
{    
    graph = GraphType(superNodeInfo_->nodeOutGraphList_, superNodeInfo_->nodeInGraphList_);
    
    for (const auto &superNode : graph.vertices()) {
        graph.set_vertex_work_weight(superNode, superNodeInfo_->nodeCycles_[superNode]); 
        SetVertexCommMemWeight(graph, superNode);

        OpCoreType vertexType = superNodeInfo_->nodeCoreType_[superNode];
        if (vertexType != OpCoreType::AIC && vertexType != OpCoreType::AIV && vertexType != OpCoreType::AICPU) {
            APASS_LOG_ERROR_F(Elements::Operation, "SuperNode (%d) has core type (%d) which is neither cube nor vector nor ai-scalar.", superNode, vertexType);
            return FAILED;
        }
        graph.set_vertex_type(superNode, getOspCoreTypeMix(vertexType));
    }

    return SUCCESS;
}

void OspPartitioner::ConstructBspArchCVSplit(osp::BspArchitecture<GraphType> &bspArch)
{
    const size_t numCubeCores = PassConfigManager::Instance().GetPlatformConfig().GetCoreNum(NpuCoreType::CUBECORE);
    const size_t numVectorCores = PassConfigManager::Instance().GetPlatformConfig().GetCoreNum(NpuCoreType::VECTORCORE);
    const size_t numAiScalarCores = PassConfigManager::Instance().GetPlatformConfig().GetCoreNum(NpuCoreType::AICORE);

    const size_t numCores = numCubeCores + numVectorCores + numAiScalarCores;
    std::vector<osp::v_type_t<GraphType>> procTypes(numCores);
    std::vector<osp::v_workw_t<GraphType>> procMemoryBound(numCores);
    
    for (size_t i = 0; i < numCores; i++) {
        if (i < numCubeCores) { // Cube Cores
            procTypes[i] = getOspCoreTypeSplit(OpCoreType::AIC);
            procMemoryBound[i] = static_cast<osp::v_workw_t<GraphType>>( PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L1) );
        } else if (i < numCubeCores + numVectorCores) { // Vector Cores
            procTypes[i] = getOspCoreTypeSplit(OpCoreType::AIV);
            procMemoryBound[i] = static_cast<osp::v_workw_t<GraphType>>( PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_UB) );
        } else { // AI Scalar Cores
            procTypes[i] = getOspCoreTypeSplit(OpCoreType::AICPU);
            procMemoryBound[i] = std::numeric_limits< osp::v_workw_t<GraphType> >::max();
        }
    }
    bspArch.setProcessorsWithTypes(procTypes);
    bspArch.setMemoryBound(procMemoryBound);
    bspArch.setCommunicationCosts(archParameters_.commCost_);
    bspArch.setSynchronisationCosts(archParameters_.synchCost_);
}

Status OspPartitioner::ConstructBspArchCVMix(osp::BspArchitecture<GraphType> &bspArch)
{
    const size_t numCubeCores = PassConfigManager::Instance().GetPlatformConfig().GetCoreNum(NpuCoreType::CUBECORE);
    const size_t numVectorCores = PassConfigManager::Instance().GetPlatformConfig().GetCoreNum(NpuCoreType::VECTORCORE);
    const size_t numAiScalarCores = PassConfigManager::Instance().GetPlatformConfig().GetCoreNum(NpuCoreType::AICORE);

    if (numCubeCores == 0 || ((numVectorCores % numCubeCores) != 0)) {
        APASS_LOG_ERROR_F(Elements::Config, "OSP bsp architecture does not satisfy an 1:N ratio of Cube:Vector cores.");
        return FAILED;
    }

    const size_t numCores = (numCubeCores != 0U ? numCubeCores : numVectorCores) + numAiScalarCores;
    const size_t numVecPerCube = (numCubeCores != 0U ? numVectorCores / numCubeCores : 1U);

    std::vector<osp::v_type_t<GraphType>> procTypes(numCores);
    std::vector<osp::v_workw_t<GraphType>> procMemoryBound(numCores);

    const size_t cubeVecMemoryBound = ((numCubeCores != 0U ? 1U : 0U)) * PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L1)
                                    + (numVecPerCube * PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_UB));

    for (size_t i = 0; i < numCores; i++) {
        if (i < numCubeCores) { // Cube Vector Core Mix
            procTypes[i] = getOspCoreTypeMix(OpCoreType::AIC);
            procMemoryBound[i] = static_cast<osp::v_workw_t<GraphType>>( cubeVecMemoryBound );
        } else { // AI Scalar Cores
            procTypes[i] = getOspCoreTypeMix(OpCoreType::AICPU);
            procMemoryBound[i] = std::numeric_limits< osp::v_workw_t<GraphType> >::max();
        }
    }
    
    bspArch.setProcessorsWithTypes(procTypes);
    bspArch.setMemoryBound(procMemoryBound);
    bspArch.setCommunicationCosts(archParameters_.commCost_);
    bspArch.setSynchronisationCosts(archParameters_.synchCost_);

    return SUCCESS;
}

Status OspPartitioner::ConstructBspInstance(osp::BspInstance<GraphType> &bspInst)
{
    if (useCVMixPartition_) {
        if (ConstructBspArchCVMix(bspInst.getArchitecture()) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate bsp architecture with CV mix.");
            return FAILED;
        }
        ConstructDagCVMix(bspInst.getComputationalDag());
    } else {
        ConstructBspArchCVSplit(bspInst.getArchitecture());
        if (ConstructDagCVSplit(bspInst.getComputationalDag()) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "OSP failed to generate graph with CV split.");
            return FAILED;
        };
    }
    unsigned numTypes = std::max(bspInst.getArchitecture().getNumberOfProcessorTypes(), static_cast<unsigned>( bspInst.getComputationalDag().num_vertex_types()));
    bspInst.setDiagonalCompatibilityMatrix(numTypes);
    return SUCCESS;
}

uint64_t OspPartitioner::CombineHash(const uint64_t h1, const uint64_t h2) const
{
    uint64_t seed = h1;
    seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

Status OspPartitioner::BuildHashValues()
{
    std::vector<uint64_t> opHashList;
    std::vector<uint64_t> opHashListFront(operationInfo_->opList_.size(), 0);
    std::vector<uint64_t> opHashListBack(operationInfo_->opList_.size(), 3);
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
        superNodeInfo_->nodeHashList_[i] = 11;
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