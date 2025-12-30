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
 * \file kernel_function.h
 * \brief
 */

#pragma once

#include "interface/function/function.h"

namespace npu::tile_fwk {

// KernelFunction is the dedicated subtype for BLOCK_GRAPH graphs.
// It contains members and methods specific to kernel (block graph) functions.
class KernelFunction : public Function {
public:
    KernelFunction(const Program &belongTo, const std::string &funcMagicName,
        const std::string &funcRawName, Function *parentFunc);

    virtual ~KernelFunction() = default;
    KernelFunction(const KernelFunction &other) = delete;
    KernelFunction(KernelFunction &&other) = delete;
    KernelFunction &operator=(const KernelFunction &other) = delete;
    KernelFunction &operator=(KernelFunction &&other) = delete;

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
private:
    SubfuncParam parameter_; // Parameter information for heterogeneous subgraph
    int programId_; // Heterogeneous subgraph ID
    std::shared_ptr<LeafFuncAttribute> leafFuncAttr_; // Leaf function attributes
};
} // namespace npu::tile_fwk