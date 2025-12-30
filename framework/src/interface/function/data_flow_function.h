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
 * \file data_flow_function.h
 * \brief
 */

#pragma once

#include "interface/function/control_flow_function.h"

namespace npu::tile_fwk {

// DataFlowFunction is the dedicated subtype for TENSOR_GRAPH and TILE_GRAPH graphs.
// It contains members and methods specific to data flow (tensor/tile graph) functions.
class DataFlowFunction : public ControlFlowFunction {
public:
    DataFlowFunction(const Program &belongTo, const std::string &funcMagicName,
        const std::string &funcRawName, Function *parentFunc);

    virtual ~DataFlowFunction() = default;
    DataFlowFunction(const DataFlowFunction &other) = delete;
    DataFlowFunction(DataFlowFunction &&other) = delete;
    DataFlowFunction &operator=(const DataFlowFunction &other) = delete;
    DataFlowFunction &operator=(DataFlowFunction &&other) = delete;

    size_t GetTotalSubGraphCount() const override { return totalSubGraphCount_; }
    void SetTotalSubGraphCount(const size_t totalSubGraphCount) override { totalSubGraphCount_ = totalSubGraphCount; }

    int GetParamIndex(const std::shared_ptr<RawTensor> &rawTensor) override;
    void *GetParamAddress(int index) override;

    const SubfuncInvokeInfoTy &GetSubFuncInvokeInfo(const size_t i) const override;

    const std::map<CoreType, std::vector<int>> &GetReadySubGraphIds() const override { return readySubGraphIds_; }
    void SetReadySubGraphIds(CoreType coreType, const std::vector<int> &readySubGraphIds) override {
        readySubGraphIds_[coreType] = readySubGraphIds;
    }
    void EmplaceReadySubGraphIds(CoreType coreType, int readySubGraphId) override {
        readySubGraphIds_[coreType].emplace_back(readySubGraphId);
    }
    void ReplaceReadySubGraphIds(CoreType coreType, int oldIdx, int newId) override {
        readySubGraphIds_[coreType][oldIdx] = newId;
    }
    size_t GetReadySubGraphCount(CoreType coreType) const override {
        auto it = readySubGraphIds_.find(coreType);
        if (it == readySubGraphIds_.end()) {
            return 0;
        }
        return it->second.size();
    }
    int GetReadySubGraphId(CoreType coreType, int index) const override {
        auto it = readySubGraphIds_.find(coreType);
        if (it == readySubGraphIds_.end()) {
            throw std::out_of_range("CoreType not found in readySubGraphIds_");
        }
        if (index >= static_cast<int>(it->second.size())) {
            throw std::out_of_range("Index out of range in readySubGraphIds_");
        }
        return it->second[index];
    }
    int GetAllReadySubGraphCount() const override {
        int size = 0;
        for (auto &ele : readySubGraphIds_) {
            size += ele.second.size();
        }
        return size;
    }

    std::unordered_set<int> LoopCheck() override;

protected:
    auto AnnotateOperation();

private:
    size_t totalSubGraphCount_ = 0; // Total subgraph count
    std::map<CoreType, std::vector<int>> readySubGraphIds_; // Ready subgraph IDs for different core types
};

} // namespace npu::tile_fwk

