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
 * \file graph_utils.h
 * \brief
 */

#pragma once
#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H
#include <vector>
#include <queue>
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "pass_common_defs.h"

namespace npu {
namespace tile_fwk {
class GraphUtils {
public:
    static Operation &AddDynOperation(Function &function, const Opcode opCode, LogicalTensors iOperands, const LogicalTensors &oOperands,
                                      const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    static Operation &AddDynRawOperation(Function &function, const Opcode opCode, LogicalTensors iOperands, const LogicalTensors &oOperands,
                                        const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    static Operation &AddViewOperation(Function &function, const ViewOp &view, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    static Operation &AddAssembleOperation(Function &function, const AssembleOp &assemble, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    static Operation &AddReshapeOperation(Function &function, LogicalTensorPtr iOperand, const LogicalTensorPtr &oOperand, const std::vector<SymbolicScalar> &outDynShape = {});
    static Operation &AddCopyInOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    static Operation &AddCopyInRawOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    static Operation &AddCopyOutOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    static Operation &AddCopyOutRawOperation(Function &function, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});   

    static void CopyDynStatus(const LogicalTensorPtr &dstTensor, const LogicalTensorPtr &srcTensor);
    static void UpdateViewAttr(Function &function, Operation &op);
    static void SetDynShape(Operation *newOp, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    static void SetCopyInAttr(Operation *newOp, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
    static void SetCopyOutAttr(Operation *newOp, const CopyInOutOp &copy, const std::vector<std::vector<SymbolicScalar>> &outDynShape = {});
};
}
}
#endif