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
 * \file pre_graph.h
 * \brief
 */

#ifndef PRE_GRAPH_PASS_H
#define PRE_GRAPH_PASS_H
#include <vector>

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

const std::string MATMUL_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";
const std::string A_MUL_B_SCALE_ATTR = OP_ATTR_PREFIX + "scale_value";
const std::string A_MUL_B_RELU_ATTR = OP_ATTR_PREFIX + "relu_type";

/* L1 Copy In 的内外轴大小 */
const std::string L1_COPY_IN_INNER = OP_ATTR_PREFIX + "inner_value";
const std::string L1_COPY_IN_OUTER = OP_ATTR_PREFIX + "outer_value";

/* L0C Copy Out 的内外轴大小 */
const std::string L0C_COPY_OUT_OUTER = OP_ATTR_PREFIX + "curH";
const std::string L0C_COPY_OUT_INNER = OP_ATTR_PREFIX + "curW";

/* 是否做搬运随路转Nz */
const std::string COPY_IS_NZ = OP_ATTR_PREFIX + "is_nz";

/* Matmul支持的数据类型
key: L0A 和 L0B 的数据类型
vaule: L0C 对应的数据类型
*/
const std::map<std::pair<DataType, DataType>, DataType> supportDtypeMap = {
    {  {DataType::DT_FP16, DataType::DT_FP16},  DataType::DT_FP32},
    {  {DataType::DT_BF16, DataType::DT_BF16},  DataType::DT_FP32},
    {  {DataType::DT_FP32, DataType::DT_FP32},  DataType::DT_FP32},
    {  {DataType::DT_FP32, DataType::DT_FP16},  DataType::DT_FP32},
    {  {DataType::DT_FP32, DataType::DT_BF16},  DataType::DT_FP32},
    {  {DataType::DT_FP16, DataType::DT_FP32},  DataType::DT_FP32},
    {  {DataType::DT_BF16, DataType::DT_FP32},  DataType::DT_FP32},
    {  {DataType::DT_INT8, DataType::DT_INT8}, DataType::DT_INT32},
    {  {DataType::DT_INT4, DataType::DT_INT4}, DataType::DT_INT32},
    {{DataType::DT_INT16, DataType::DT_INT16}, DataType::DT_INT32},
};

struct SubgraphColorInfo {
    std::vector<bool> visited;
    std::vector<int> newColor;
};

class PreGraphProcess : public Pass {
public:
    PreGraphProcess() : Pass("PreGraphProcess") {}
    ~PreGraphProcess() override = default;

private:
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    Status PreColorSort(Function &function);
    bool IsCandidateAssembleOp(Function &function, Operation &op) const;
    void DeleteRedundantAssemble(Function &function) const;
    void ProcessSpecialMTEOperation(Operation &op) const;
    void ProcessMoveInOperation(Operation &op) const;
    void InsertTemporaryCopyIn(Function &function, Operation &op) const;
    void ProcessInplaceOp(Function &function) const;
    void UpdateCopyOpIsCube(Operation &op) const;
    void InitializeTensorColor(Operation &op) const;
    void ProcessSameInOutOp(Function &function) const;
    void SetTensorBoundary(Function &function) const;
    void HandleForAssembleToOutcast(Function &function, std::unordered_set<Operation *> &concurrentAssembles,
        std::set<Operation *, LogicalTensor::CompareOp> &producersBackup) const;
    void HandleForAssembleFromInOut(Function &function, std::unordered_set<Operation *> &concurrentAssembles,
        std::set<Operation *, LogicalTensor::CompareOp> &producersBackup) const;
    void HandleForReshapeToOutcast(Function &function) const;
    Status CheckValidCube(const Operation &op);
    Status UpdateCubeOp(Function &function);
    Status UpdateCopyAttr(Operation &op) const;
    Status AddL1CopyInAttr(
        const std::shared_ptr<LogicalTensor> input, int nzValue, int mValue, int kValue, int nValue) const;
    Status AddL0cCopyOutAttr(const std::shared_ptr<LogicalTensor> output, int nzValue, int mValue, int nValue) const;
    Status UpdateL0cDtype(Operation &op);
    std::pair<Operation *, Operation *> GetLastMmCopyOut(Operation &op);
    Status ReconnectGraph(Operation &mulOp, Operation *copyOutOp);
    Status TransferAttr(Operation &mulOp, Operation *copyOutOp);
    bool IsFloat(const std::shared_ptr<LogicalTensor> tensor) const;
    bool IsInt(const std::shared_ptr<LogicalTensor> tensor) const;
};
} // namespace npu::tile_fwk
#endif // PRE_GRAPH_PASS_H