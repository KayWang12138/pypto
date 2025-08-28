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
 * \file shmem_expand_funcion.cpp
 * \brief
 */

#include "distributed_expand.h"
#include "distributed_common.h"

namespace npu::tile_fwk::Distributed {
void TiledShmemPut(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    if (iOperand.size() != 2UL || oOperand.size() != 1UL) {
        ALOG_ERROR_F("TiledShmemPut iOperand size=%lu, oOperand size=%lu", iOperand.size(), oOperand.size());
        return;
    }
    (void)tileShape;
    std::shared_ptr<LogicalTensor> in = iOperand[0];
    std::shared_ptr<LogicalTensor> shmDataTile = iOperand[1];
    std::shared_ptr<LogicalTensor> dummy = oOperand[0];
    CommGroupInfo groupInfo;
    TilingInfo tilingInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    op.GetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    std::string atomicType = "";
    op.GetAttr("AtomicType", atomicType);

    std::vector<int64_t> shape{shmDataTile->GetShape()[2] * shmDataTile->GetShape()[3]};
    auto inTile = in->View(function, in->GetShape(), {tilingInfo.rowOffset, tilingInfo.colOffset});
    auto ubTensor = std::make_shared<LogicalTensor>(function, shmDataTile->Datatype(), shape);
    OpArgs<TilingInfo> opArgs = {"SHMEM_PUT", {inTile, shmDataTile}, {dummy, ubTensor}, nullptr, "",
        std::make_optional(tilingInfo), std::nullopt};
    auto& tileop = AddOperation(function, opArgs);
    tileop.SetAttr("AtomicType", atomicType);
}

void TiledShmemSignal(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    if (iOperand.size() != 1UL || oOperand.size() != 1UL) {
        ALOG_ERROR_F("TiledShmemSignal iOperand size=%lu, oOperand size=%lu", iOperand.size(), oOperand.size());
        return;
    }
    (void)tileShape;
    std::shared_ptr<LogicalTensor> dummy = iOperand[0];
    std::shared_ptr<LogicalTensor> shmSignalTile = oOperand[0];
    CommGroupInfo groupInfo;
    TilingInfo tilingInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    op.GetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    std::string atomicType = "";
    std::string value = "";
    op.GetAttr("AtomicType", atomicType);
    op.GetAttr("Value", value);

    std::vector<int64_t> shape{8};
    auto ubTensor = std::make_shared<LogicalTensor>(function, shmSignalTile->Datatype(), shape);
    OpArgs<TilingInfo> opArgs = {"SHMEM_SIGNAL", {dummy}, {shmSignalTile, ubTensor}, nullptr, "",
        std::make_optional(tilingInfo), std::nullopt};
    auto& tileop = AddOperation(function, opArgs);
    tileop.SetAttr("Value", value);
    tileop.SetAttr("AtomicType", atomicType);
}

void TiledShmemWaitUntil(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    if (iOperand.size() != 2UL || oOperand.size() != 1UL) {
        ALOG_ERROR_F("TiledShmemWaitUntil iOperand size=%lu, oOperand size=%lu", iOperand.size(), oOperand.size());
        return;
    }
    (void)tileShape;
    std::shared_ptr<LogicalTensor> in = iOperand[0];
    std::shared_ptr<LogicalTensor> shmSignalTile = iOperand[1];
    std::shared_ptr<LogicalTensor> dummy = oOperand[0];
    CommGroupInfo groupInfo;
    TilingInfo tilingInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    op.GetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    std::string stride = "";
    std::string value = "";
    op.GetAttr("Stride", stride);
    op.GetAttr("Value", value);

    auto inTile = in->View(function, in->GetShape(), {tilingInfo.rowOffset, tilingInfo.colOffset});
    OpArgs<TilingInfo> opArgs = {"SHMEM_WAIT_UNTIL", {inTile, shmSignalTile}, {dummy}, nullptr, "",
        std::make_optional(tilingInfo), std::nullopt};
    auto& tileop = AddOperation(function, opArgs);
    tileop.SetAttr("Value", value);
    tileop.SetAttr("Stride", stride);
}

void TiledShmemGet(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op)
{
    if (iOperand.size() != 2UL || oOperand.size() != 1UL) {
        ALOG_ERROR_F("TiledShmemGet iOperand size=%lu, oOperand size=%lu", iOperand.size(), oOperand.size());
        return;
    }
    (void)tileShape;
    (void)op;
    std::shared_ptr<LogicalTensor> dummy = iOperand[0];
    std::shared_ptr<LogicalTensor> shmDataTile = iOperand[1];
    std::shared_ptr<LogicalTensor> out = oOperand[0];
    CommGroupInfo groupInfo;
    TilingInfo tilingInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    op.GetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    std::string atomicType = "";
    op.GetAttr("AtomicType", atomicType);

    std::vector<int64_t> shape{shmDataTile->GetShape()[2], shmDataTile->GetShape()[3]};
    auto ubTensor = std::make_shared<LogicalTensor>(function, shmDataTile->Datatype(), shape);
    auto outTile = out->View(function, out->GetShape(), out->GetOffset());
    OpArgs<TilingInfo> opArgs = {"SHMEM_GET", {dummy, shmDataTile}, {outTile, ubTensor}, nullptr, "",
        std::make_optional(tilingInfo), std::nullopt};
    auto& tileop = AddOperation(function, opArgs);
    tileop.SetAttr("AtomicType", atomicType);
}
}