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
 * \file osp_partitioner.h
 * \brief
 */

#ifndef PASS_OSP_PARTITIONER_H
#define PASS_OSP_PARTITIONER_H
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/dag_vector_adapter.hpp"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
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
#include "tilefwk/platform.h"
#include "supernode_graph_builder.h"
//#include "passes/pass_config/pass_config_manager.h" where is this now?
#include "passes/pass_interface/pass.h"
#include <unordered_map>

namespace npu::tile_fwk {

template<typename GraphT>
struct ArchParameters {
    osp::VWorkwT<GraphT> commCost_  = 1;
    osp::VWorkwT<GraphT> synchCost_ = 8000;
    double commCorrectionFactor_ = 0.01;
    osp::VWorkwT<GraphT> partitionWorkUpperBound_ = std::numeric_limits<osp::VWorkwT<GraphT>>::max();
    osp::VWorkwT<GraphT> partitionWorkLowerBound_ = std::numeric_limits<osp::VWorkwT<GraphT>>::lowest();
};

enum class OspMode {
    SARKAR      = 1, 
    MERKLEBSP   = 2
};

class OspPartitioner : public SuperNodeGraphBuilder {
    using VertType = int32_t;
    using WorkType = int32_t;
    using VTypeType = unsigned;
    using VertexImpl = osp::CDagVertexImpl<VertType, WorkType, WorkType, WorkType, VTypeType>;
    using GraphType = osp::DagVectorAdapter<VertexImpl>;
    using ConstrGraphType = osp::ComputationalDagVectorImpl<VertexImpl>;
    using CoarseGraphType = osp::CompactSparseGraph<VertType, VertType, WorkType, WorkType, WorkType, VTypeType>;

    // Core/Vertex type translation maps
    const std::unordered_map<OpCoreType, VTypeType> ospCoreTypeMapSplit{
        {OpCoreType::AIC,       0U},
        {OpCoreType::AIV,       1U},
        {OpCoreType::AICPU,     2U},
        {OpCoreType::ANY,       3U},
        {OpCoreType::HUB,       4U},
        {OpCoreType::GMATOMIC,  5U}
    };
    const std::unordered_map<OpCoreType, VTypeType> ospCoreTypeMapMix{
        {OpCoreType::AIC,       0U},
        {OpCoreType::AIV,       0U},
        {OpCoreType::AICPU,     1U},
        {OpCoreType::ANY,       2U},
        {OpCoreType::HUB,       3U},
        {OpCoreType::GMATOMIC,  4U}
    };

    // Parameters
    ArchParameters<GraphType> archParameters_;
    OspMode ospMode_;

    // Init
    Status BuildSuperNodeGraph() override;
    
    // Construction of OSP instance
    Status ConstructDagCVSplit(GraphType &graph);
    Status ConstructDagCVMix(GraphType &graph);
    Status ConstructDag(GraphType &graph);
    void ConstructBspArchCVSplit(osp::BspArchitecture<GraphType> &bspArch);
    Status ConstructBspArchCVMix(osp::BspArchitecture<GraphType> &bspArch);
    Status ConstructBspInstance(osp::BspInstance<GraphType> &bspInst);

    // Construction Helpers
    void SetVertexCommMemWeight(GraphType &graph, int32_t vertex);
    inline VTypeType GetOspCoreTypeSplit(OpCoreType coreType) { return ospCoreTypeMapSplit.at(coreType);}
    inline VTypeType GetOspCoreTypeMix(OpCoreType coreType) { return ospCoreTypeMapMix.at(coreType);}
    
    // Run OSP Partition
    Status RunOspPartition(Function &function);
    Status UpdatePartitionResult(Function &function, std::vector<osp::VertexIdxT<GraphType>> &vertexContractionMap);

    // Algorithms
    Status RunSarkar(const GraphType &graph, CoarseGraphType &coarseGraph, std::vector<osp::VertexIdxT<GraphType>> &vertexContractionMap);
    Status RunMerkleBsp(const osp::BspInstance<GraphType> &bspInst, std::vector<osp::VertexIdxT<GraphType>> &vertexContractionMap);
    
    // Helpers
    uint64_t CombineHash(const uint64_t h1, const uint64_t h2) const override ;
    Status BuildHashValues() override;

public:    
    OspPartitioner(OspMode mode) : ospMode_(mode) {};
    inline void SetParameter(const Function &function) {
            archParameters_.partitionWorkUpperBound_ = function.paramConfigs_.sgPgUpperBound;
            archParameters_.partitionWorkLowerBound_ = function.paramConfigs_.sgPgLowerBound;
    }
    ~OspPartitioner() = default;
    Status PartitionGraph(Function &function);
    ArchParameters<GraphType> &GetArchParameters() { return archParameters_; };
};

}  // namespace npu::tile_fwk
#endif  // PASS_OSP_PARTITIONER_H