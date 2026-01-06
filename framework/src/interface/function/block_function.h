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
 * \file block_function.h
 * \brief
 */

#pragma once

#include "interface/function/function.h"

namespace npu::tile_fwk {

// BlockFunction is the dedicated subtype for BLOCK_GRAPH graphs.
// It contains members and methods specific to kernel (block graph) functions.
class BlockFunction : public Function {
public:
    BlockFunction(const Program &belongTo, const std::string &funcMagicName,
        const std::string &funcRawName, Function *parentFunc);

    virtual ~BlockFunction() = default;
    BlockFunction(const BlockFunction &other) = delete;
    BlockFunction(BlockFunction &&other) = delete;
    BlockFunction &operator=(const BlockFunction &other) = delete;
    BlockFunction &operator=(BlockFunction &&other) = delete;

    std::vector<OperationPtr> &GetProgramOp() override;
    void SetProgramOp(const std::vector<OperationPtr> &operations) override;
    void UpdateBelongToThis() override;
    void ScheduleBy(const std::vector<Operation *> &newList, bool needRefresh = false) override;

    const SubfuncParam &GetParameter() const override { return parameter_; }
    SubfuncParam &GetParameter() override { return parameter_; }
    void SetParameter(const SubfuncParam &parameter) override { parameter_ = parameter; }

    int GetProgramId() const override { return programId_; }
    void SetProgramId(int programId) override { programId_ = programId; }

    void SetLeafFuncAttribute(const std::shared_ptr<LeafFuncAttribute> &attr) override { leafFuncAttr_ = attr; }
    const std::shared_ptr<LeafFuncAttribute> &GetLeafFuncAttribute() const override { return leafFuncAttr_; }
    std::shared_ptr<LeafFuncAttribute> &GetLeafFuncAttribute() override { return leafFuncAttr_; }

    std::vector<std::vector<SymbolicScalar>> NormalizeCoa(
        std::vector<int> &iOffset, std::vector<int> &oOffset) override;
    void NormalizeCoaForInCasts(std::vector<int> &iOffset, std::vector<std::vector<SymbolicScalar>> &coaLists,
        int &coaIndex, std::unordered_map<LogicalTensorPtr, int> &processedOperands,
        const std::unordered_map<int, Operation *> &opmagicToOp) override;
    void NormalizeCoaForOutCasts(std::vector<int> &oOffset, std::vector<std::vector<SymbolicScalar>> &coaLists,
        int &coaIndex, std::unordered_map<LogicalTensorPtr, int> &processedOperands,
        const std::unordered_map<int, Operation *> &opmagicToOp) override;
    void NormalizeCoaForNormalOperands(std::vector<std::vector<SymbolicScalar>> &coaLists, int &coaIndex,
        std::unordered_map<LogicalTensorPtr, int> &processedOperands) override;
    void NormalizeCoaForSpecialInfo(std::vector<std::vector<SymbolicScalar>> &coaLists, int &coaIndex) override;
    void GetOutcastSymbolicExpr(std::map<int, SymbolicScalar>& tabel) override;

    std::pair<bool, Opcode> IsAicpuSubFunction() const override {
        Opcode code = Opcode::OP_UNKNOWN;
        for (size_t i = 0UL; i < operations_.size(); i++) {
            if ((operations_[i]->GetOpcode() != Opcode::OP_VIEW) &&
                (operations_[i]->GetCoreType() != CoreType::AICPU)) {
                    return std::make_pair(false, Opcode::OP_UNKNOWN);
            } else if (operations_[i]->GetCoreType() == CoreType::AICPU) {
                   code = operations_[i]->GetOpcode();
            }
        }
        return std::make_pair(true, code);
    }

    void CreateLeafInAndOutCast(const LogicalTensorPtr &inOrOut, LogicalTensors &inOrOutList) const override;
    GetTensorDataIODescDict GetTensorDataForLeafGraph() override;
    
    void AppendIncast(LogicalTensorPtr tensor, int opmagic, int k) override {
        incastPosition.emplace_back(opmagic, k);
        inCasts_.emplace_back(tensor);
    }

    void AppendOutcast(LogicalTensorPtr tensor, int opmagic, int k) override {
        outcastPosition.emplace_back(opmagic, k);
        outCasts_.emplace_back(tensor);
    }

    DynParamInfo &GetMutableDynParam(std::string dim) override {
        return dynParamTable_[dim];
    }

    void InsertDynParam(std::string dim, DynParamInfo &info) override {
        dynParamTable_.emplace(dim, info);
    }

    const std::map<std::string, DynParamInfo> &GetDynParamTable() const override {
        return dynParamTable_;
    }
private:
    std::map<std::string, DynParamInfo> dynParamTable_;
    SubfuncParam parameter_; // Parameter information for heterogeneous subgraph
    int programId_; // Heterogeneous subgraph ID
    std::shared_ptr<LeafFuncAttribute> leafFuncAttr_; // Leaf function attributes
};
} // namespace npu::tile_fwk