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
 * \file pass_manager.cpp
 * \brief
 */

#include "pass_manager.h"

#include <cstdlib>
#include <unistd.h>
#include "interface/configs/config_manager.h"
#include "pass_interface/pass.h"
#include "pass_interface/pass_type.h"
#include "pass_registry.h"
#include "interface/tensor/expected_value.h"
#include "tilefwk/error.h"
// tensor graph pass
#include "passes/tensor_graph_pass/remove_redundant_reshape.h"
#include "passes/tensor_graph_pass/remove_redundant_cast.h"
#include "passes/tensor_graph_pass/infer_memory_conflict.h"
#include "passes/tensor_graph_pass/expand_function.h"
//  tile graph pass
#include "passes/tile_graph_pass/generate_move_op.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"
#include "passes/tile_graph_pass/common_operation_eliminate.h"
#include "passes/tile_graph_pass/assign_memory_type.h"
#include "passes/tile_graph_pass/duplicate_view.h"
#include "passes/tile_graph_pass/l1_copy_reuse.h"
#include "passes/tile_graph_pass/merge_view_assemble.h"
#include "passes/tile_graph_pass/insert_copy_op.h"
#include "passes/tile_graph_pass/intra_subgraph_adapter.h"
#include "passes/tile_graph_pass/pad_local_buffer.h"
#include "passes/tile_graph_pass/inplace_process.h"
#include "passes/tile_graph_pass/pre_graph.h"
#include "passes/tile_graph_pass/remove_redundant_op.h"
#include "passes/tile_graph_pass/n_buffer_merge.h"
#include "passes/tile_graph_pass/split_large_local_raw.h"
#include "passes/tile_graph_pass/update_memory_map.h"
#include "passes/tile_graph_pass/split_large_fanout_tensor.h"
#include "passes/tile_graph_pass/cube_process.h"
#include "passes/tile_graph_pass/remove_unaligned_reshape_op.h"
#include "passes/tile_graph_pass/split_reshape.h"
#include "passes/tile_graph_pass/infer_dyn_shape.h"
#include "passes/tile_graph_pass/iso_partitioner.h"
#include "passes/tile_graph_pass/infer_dyn_shape.h"
// execute graph pass
#include "passes/execute_graph_pass/memory_reuse.h"
#include "passes/execute_graph_pass/subgraph_to_function.h"
#include "passes/execute_graph_pass/insert_sync.h"
#include "passes/execute_graph_pass/schedule_ooo.h"
#include "passes/execute_graph_pass/codegen_preproc.h"
#include "passes/execute_graph_pass/infer_param_index.h"
#include "passes/execute_graph_pass/add_alloc.h"
#include "passes/execute_graph_pass/remove_alloc.h"
#include "passes/execute_graph_pass/merge_src_dst_buffer.h"
#include "passes/pass_config/pass_config_manager.h"

namespace npu::tile_fwk {
PassManager &PassManager::Instance() {
    static PassManager instance;
    return instance;
}

void RegPass() {
    REG_PASS(GlobalMemoryReuse);
    REG_PASS(UpdateMemoryMap);
    REG_PASS(SubgraphToFunction);
    REG_PASS(GraphPartition);
    REG_PASS(InsertSync);
    REG_PASS(OoOSchedule);
    REG_PASS(ExpandFunction);
    REG_PASS(CommonOperationEliminate);
    REG_PASS(GenerateMoveOp);
    REG_PASS(AssignMemoryType);
    REG_PASS(DuplicateView);
    REG_PASS(RemoveRedundantReshape);
    REG_PASS(RemoveRedundantCast);
    REG_PASS(InferMemoryConflict);
    REG_PASS(NBufferMerge);
    REG_PASS(L1CopyInReuseMerge);
    REG_PASS(MergeViewAssemble);
    REG_PASS(InsertInterGraphCopy);
    REG_PASS(IntraSubgraphAdapter);
    REG_PASS(PadLocalBuffer);
    REG_PASS(InplaceProcess);
    REG_PASS(PreGraphProcess);
    REG_PASS(RemoveRedundantOp);
    REG_PASS(SplitLargeLocalRawTensor);
    REG_PASS(SplitReshape);
    REG_PASS(RemoveUnalignedReshape);
    REG_PASS(CodegenPreproc);
    REG_PASS(SplitLargeFanoutTensor);
    REG_PASS(CubeProcess);
    REG_PASS(InferDynShape);
    REG_PASS(InferParamIndex);
    REG_PASS(AddAlloc);
    REG_PASS(RemoveAlloc);
    REG_PASS(SrcDstBufferMerge);
}

void PassManager::RegDefaultStrategy() {
    RegisterStrategy(
        "PVC2_OOO", {
            {   "RemoveRedundantReshape",   "RemoveRedundantReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {             "SplitReshape",             "SplitReshape",    PassType::TYPE_TILE_GRAPH},
            {        "RemoveRedundantOp",        "RemoveRedundantOp",    PassType::TYPE_TILE_GRAPH},
            {        "GenerateMoveOp_01",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
            {              "CubeProcess",              "CubeProcess",    PassType::TYPE_TILE_GRAPH},
            {           "GraphPartition",           "GraphPartition",    PassType::TYPE_TILE_GRAPH},
            {             "NBufferMerge",             "NBufferMerge",    PassType::TYPE_TILE_GRAPH},
            {          "UpdateMemoryMap",          "UpdateMemoryMap",    PassType::TYPE_TILE_GRAPH},
            {        "GenerateMoveOp_02",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
            { "SplitLargeLocalRawTensor", "SplitLargeLocalRawTensor",    PassType::TYPE_TILE_GRAPH},
            {     "InsertInterGraphCopy",     "InsertInterGraphCopy",    PassType::TYPE_TILE_GRAPH},
            { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},
            {       "L1CopyInReuseMerge",       "L1CopyInReuseMerge",    PassType::TYPE_TILE_GRAPH},
            {           "InplaceProcess",           "InplaceProcess",    PassType::TYPE_TILE_GRAPH},
            {          "PreGraphProcess",          "PreGraphProcess",    PassType::TYPE_TILE_GRAPH},
            {           "PadLocalBuffer",           "PadLocalBuffer",    PassType::TYPE_TILE_GRAPH},
            {   "RemoveUnalignedReshape",   "RemoveUnalignedReshape",    PassType::TYPE_TILE_GRAPH},
            {            "InferDynShape",            "InferDynShape",    PassType::TYPE_TILE_GRAPH},
            {       "SubgraphToFunction",       "SubgraphToFunction", PassType::TYPE_EXECUTE_GRAPH},
            {          "InferParamIndex",          "InferParamIndex", PassType::TYPE_EXECUTE_GRAPH},
            {        "SrcDstBufferMerge",        "SrcDstBufferMerge", PassType::TYPE_EXECUTE_GRAPH},
            {                 "AddAlloc",                 "AddAlloc", PassType::TYPE_EXECUTE_GRAPH},
            {              "OoOSchedule",              "OoOSchedule", PassType::TYPE_EXECUTE_GRAPH},
            {        "GlobalMemoryReuse",        "GlobalMemoryReuse", PassType::TYPE_EXECUTE_GRAPH},
            {              "RemoveAlloc",              "RemoveAlloc", PassType::TYPE_EXECUTE_GRAPH},
            {               "InsertSync",               "InsertSync", PassType::TYPE_EXECUTE_GRAPH},
            {           "CodegenPreproc",           "CodegenPreproc", PassType::TYPE_EXECUTE_GRAPH},
    });
}

PassManager::PassManager() {
    RegPass();
    // Register strategies
    RegDefaultStrategy();
}

void PassManager::RegisterStrategy(const std::string &strategy, const std::vector<PassEntry> &passEntries) {
    // check identifiers duplication
    std::vector<PassEntry> newPassEntries;
    std::set<std::string> identifiers;
    for (auto &pass : passEntries) {
        if (!(identifiers.insert(pass.identifier).second)) {
            ALOG_WARN_F("Duplicated identifier: %s.", pass.identifier.c_str());
        } else {
            newPassEntries.push_back(pass);
        }
    }
    auto strategyPasses = strategies_.find(strategy);
    if (strategyPasses == strategies_.end()) {
        strategies_.emplace(strategy, newPassEntries);
    } else {
        strategyPasses->second = newPassEntries;
        ALOG_WARN_F("Strategy %s has been changed.", strategy.c_str());
    }
}

std::vector<PassManager::PassEntry> PassManager::GetStrategyPasses(const std::string &strategy) const {
    auto it = strategies_.find(strategy);
    if (it == strategies_.end()) {
        ALOG_WARN_F("Strategy %s does not exist.", strategy.c_str());
        auto emptyPass = std::vector<PassManager::PassEntry>();
        return emptyPass;
    }
    return it->second;
}

std::string PassManager::GetResumePath(const std::string &strategy) {
    auto strategyPasses = GetStrategyPasses(strategy);
    for (size_t i = 0; i < strategyPasses.size(); i++) {
        const auto &identifier = strategyPasses[i].identifier;
        auto passDfxCfg = ConfigManager::Instance().GetPassConfigs(strategy, identifier);
        if (passDfxCfg.resumePath != "") {
            if (access(passDfxCfg.resumePath.c_str(), F_OK) == 0) {
                startIdx = i;
            }
            return passDfxCfg.resumePath;
        }
    }
    startIdx = static_cast<size_t>(0);
    return "";
}

Status PassManager::RunPass(Program &program, Function &function, const std::string &strategy) const {
    PassConfigManager::Instance().Initialize(config::GetDevicePlatform());
    auto strategyPasses = GetStrategyPasses(strategy);
    std::vector<std::string> identifiers;
    std::transform(strategyPasses.begin(), strategyPasses.end(), std::back_inserter(identifiers),
        [](const PassEntry &elem) { return elem.identifier; });
    ConfigManager::Instance().PassConfigsDebugInfo(strategy, identifiers);
    for (size_t i = startIdx; i < strategyPasses.size(); i++) {
        const auto &identifier = strategyPasses[i].identifier;
        const auto &passName = strategyPasses[i].passName;
        auto pass = PassRegistry::GetInstance().CreatePass(passName);
        if (pass == nullptr) {
            ALOG_ERROR_F("Pass [%s] does not exist.", passName.c_str());
            return FAILED;
        }
        std::string originLogOutPath = config::LogFile();
        std::string logFolder = pass->LogFolder(config::LogTopFolder(), i);
        std::string logfilePath = logFolder + "/" + (pass->GetName() + function.GetMagicName() + ".log");
        LoggerManager::FileLoggerReplace(originLogOutPath, logfilePath, true);
        AutoDestructorCallback adc([logfilePath, originLogOutPath]() {
            LoggerManager::FileLoggerReplace(logfilePath, originLogOutPath, true);
        });
        auto passDfxCfg = ConfigManager::Instance().GetPassConfigs(strategy, identifier);
        pass->SetPassConfigs(passDfxCfg);
        ALOG_INFO_F("[PassManager] Apply pass <%s> on function: %s.", identifier.c_str(), function.GetMagicName().c_str());
        auto start = std::chrono::high_resolution_clock::now();
        if (pass->Run(function, strategy, identifier, i) != SUCCESS) {
            ALOG_ERROR_F("Run pass <%s> failed.", identifier.c_str());
            return FAILED;
        }
        if (passDfxCfg.dumpPassTimeCost) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            ALOG_INFO_F("Runtime of pass %s for program %s function %s is %ld us.", identifier.c_str(), program.Name().c_str(),
                function.GetMagicName().c_str(), duration.count());
        }
        if (config::GetPlatformConfig(KEY_VERIFY_PASS, false)) {
            constexpr int SELECT_ALL_PASS = -1;
            int selectIndex = config::GetPlatformConfig(KEY_VERIFY_PASS_SELECT, SELECT_ALL_PASS);
            if ((selectIndex < SELECT_ALL_PASS) || (selectIndex > static_cast<int>(strategyPasses.size()))) {
                ALOG_ERROR_F("Invalid PASS Index %s.", selectIndex);
                return FAILED;
            }
            if (selectIndex == SELECT_ALL_PASS ||
                (((static_cast<size_t>(selectIndex) - 1) <= i) && (i <= static_cast<size_t>(selectIndex)))) {
                Program::GetInstance().VerifyPass(&function, i, identifier);
            }
        }
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
