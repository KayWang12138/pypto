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
 * \file distributed_expand.h
 * \brief
 */

#ifndef DISTRIBUTED_EXPAND_H
#define DISTRIBUTED_EXPAND_H

#include <cstdint>
#include <vector>
#include <memory>
#include "interface/configs/config_storage.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {
namespace Distributed {
constexpr int32_t SHARED_EXPERT_NUM = 1;
constexpr int32_t ROUTING_EXPERT_NUM = 3;
constexpr int32_t TOTAL_EXPERT_NUM = SHARED_EXPERT_NUM + ROUTING_EXPERT_NUM;
constexpr int32_t AIV_NUM = 4;

inline bool IsRoutingExpert(int rankId)
{
    return rankId >= SHARED_EXPERT_NUM;
}
void TiledDistReduce(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledDistScatter(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand, const Operation &op);
void TiledDistGather(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledDistBroadCast(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledMoeFFN2Attn(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand, const Operation &op);
void TiledMoeAttnCombine(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void SendToRoutingExpert(const Tensor &tokenTensor, const Tensor &tokenExpertTable,
    const Tensor &tilingTensor, const Tensor &syncTensor, const char *group);
void SendToSharedExpert(const Tensor &tokenTensor, const Tensor &tilingTensor, const Tensor &syncTensor,
    const char *group);
void CopyToLocalExpert(const Tensor &tokenTensor, const Tensor &tilingTensor, const Tensor &expandX);
Tensor DispatchSetFlag(const Tensor &tokenExpertTable, const Tensor &syncTensor, const Tensor &tilingTensor,
    const char *group);
void TiledSendToRoutingExpert(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledSendToSharedExpert(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledCopyToLocalExpert(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledDispatchSetFlag(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledDispatchFFNSched(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledDispatchFFNBatching(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemPut(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemGet(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemSignal(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemWaitUntil(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);
void TiledShmemReduce(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);

} // namespace npu::tile_fwk
} // namespace Distributed

#endif // DISTRIBUTED_EXPAND_H