/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mix_call_operation_builder.h
 * \brief 用于提供所需的接口
 */

#ifndef MIX_CALL_OPERATION_BUILDER_H
#define MIX_CALL_OPERATION_BUILDER_H

#include <set>
#include <vector>
#include <unordered_map>
#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "passes/tile_graph_pass/subgraph_to_function.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_subgraph_split_utils.h"

namespace npu {
namespace tile_fwk {
struct ExtractInfo {
    std::vector<int>& iOffsets;
    std::vector<int>& oOffsets;
    std::set<LogicalTensorPtr>& processedTensors;
};

struct CallOpCreationInfo {
    Function* leafFunc;
    uint64_t newProgramID;
    size_t componentIndex;
    Operation* originalCallOp;
    uint64_t wrapId;
    std::vector<int> iOffsets;
    std::vector<int> oOffsets;
    Operation* createdCallOp = nullptr;
};


class MixCallOperationBuilder {
public:
    Status CreateCallOps(Function& rootFunc,
                         const std::vector<Operation*>& originalCallOps, 
                         Function* originalMixFunc,
                         const std::vector<InternalComponentInfo>& components,
                         const std::vector<uint64_t>& newProgramIDs,
                         SubgraphToFunction& subgraphToFunction,
                         std::vector<Function*>& newFunctions,
                         const std::vector<InternalDependencyInfo>& internalDeps);
private:
    // 在root function中创建call op
    Status CreateCallOpInRootFunction(Function& rootFunc,
                                      Function& leafFunc,
                                      uint64_t newProgramID,
                                      uint64_t componentIndex,
                                      Operation* originalCallOp,
                                      Function* originalMixFunc,
                                      SubgraphToFunction& subgraphToFunction,
                                      CallOpCreationInfo& info);
    int FindTensorIndexInList(int tensorMagic, const std::vector<LogicalTensorPtr>& tensorList) const;
    // 参数提取函数
    std::vector<std::vector<SymbolicScalar>> ExtractArgListForLeafFunction(Function& leafFunc,
                                                                           CallOpAttribute* originalCallAttr,
                                                                           const SubfuncInvokeInfoTy& invokeInfo,
                                                                           std::vector<int>& iOffsets,
                                                                           std::vector<int>& oOffsets,
                                                                           Function* originalMixFunc) const;
    bool ExtractArgListFromIncast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, ExtractInfo& extractInfo) const;
    bool ExtractArgListFromOutcast(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, ExtractInfo& extractInfo) const;
    bool ExtractArgListFromGlobalTensor(const SubfuncInvokeInfoTy& invokeInfo, Function& leafFunc, ExtractInfo& extractInfo) const;
    bool ExtractArgListFromActualIncasts(const std::vector<std::shared_ptr<LogicalTensor>> &actualIncasts, ExtractInfo& extractInfo, Function* originalMixFunc) const; 
    bool ExtractArgListFromActualOutcasts(const std::vector<std::shared_ptr<LogicalTensor>> &actualOutcasts, ExtractInfo& extractInfo, Function* originalMixFunc) const;
    
    int GetOffsetFromIncastParam(const SubfuncInvokeInfoTy::IncastParamPackTy& incastParam, Function& leafFunc) const;
    int GetOffsetFromOutcastParam(const SubfuncInvokeInfoTy::OutcastParamPackTy& outcastParam, Function& leafFunc) const;
    int GetOffsetFromOp(int opMagic, int operandIdx, Function& leafFunc, bool isOutput) const;
    int GetOffsetFromTensorParam(const SubfuncInvokeInfoTy::TensorParamPackTy& tensorParam, Function& leafFunc) const;
    int FindOriginalOffsetInMixFunction(LogicalTensorPtr tensor, Function* originalMixFunc) const;
    // 内部依赖处理相关函数
    void ProcessAllInternalDependencies(Function& rootFunc,
                                        const std::vector<CallOpCreationInfo>& callOpInfos,
                                        const std::vector<InternalDependencyInfo>& internalDeps) const;
    void ProcessInternalDependenciesForWrap(Function& rootFunc,
                                            const std::vector<const CallOpCreationInfo*>& infos,
                                            const std::vector<InternalDependencyInfo>& internalDeps,
                                            uint64_t wrapId) const;

    uint64_t nextWrapId_;
};
} // namespace tile_fwk
} // namespace npu

#endif // MIX_CALL_OPERATION_BUILDER_H