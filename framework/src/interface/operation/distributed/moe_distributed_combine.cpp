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
 * \file moe_distributed_combine.cpp
 * \brief
 */

#include "distributed_common.h"
#include "interface/function/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_distributed.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tilefwk.h"

namespace npu::tile_fwk::Distributed {
void MoeDistributedCombineValidateExpandX(const Tensor& expandX, const Tensor& expertScales, int32_t epWorldSize,
    int32_t moeExpertNum)
{
    uint32_t supportedDim = 2;
    ASSERT(expandX.GetShape().size() == supportedDim) << "The dim of \"expandX\" only supports " << supportedDim
        << ", but got " << expandX.GetShape().size();

    int32_t expandXRow = expandX.GetShape(0);
    int32_t topK = expertScales.GetShape(1);
    int32_t batchSize = expertScales.GetShape(0);
    int32_t expectedRow = std::min(topK * batchSize * epWorldSize, batchSize * moeExpertNum);
    ASSERT(expandXRow == expectedRow) << "The first axis of \"expandX\" must be the smaller value between topK (the "
        << "second axis of \"expertScales\") * batchSize (the first axis of \"expertScales\") * epWorldSize and "
        << "batchSize * moeExpertNum, topK=" << topK << ", batchSize=" << batchSize << ", epWorldSize=" << epWorldSize
        << ", moeExpertNum=" << moeExpertNum << ", the expected first axis of \"expandX\" should be " << expectedRow
        << " but got " << expandXRow;

    int32_t expandXCol = expandX.GetShape(1);
    int32_t supportedHiddenSize = 5120;
    ASSERT(expandXCol == supportedHiddenSize) << "The second axis of \"expandX\" only supports " << supportedHiddenSize
        << ", but got " << expandXCol;

    ASSERT(expandX.GetDataType() == DT_BF16) << "The data type of \"expandX\" only supports DT_BF16, but got "
        << DataType2String(expandX.GetDataType());

    ASSERT(expandX.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"expandX\" only supports ND, "
        << "but got NZ";
}

void MoeDistributedCombineValidateAssistInfoForCombine(const Tensor& assistInfoForCombine, const Tensor& expandX)
{
    uint32_t supportedDim = 2;
    ASSERT(assistInfoForCombine.GetShape().size() == supportedDim) << "The dim of \"assistInfoForCombine\" only "
        << "supports " << supportedDim << ", but got " << assistInfoForCombine.GetShape().size();

    int32_t assistInfoForCombineRow = assistInfoForCombine.GetShape(0);
    int32_t expandXRow = expandX.GetShape(0);
    ASSERT(assistInfoForCombineRow == expandXRow) << "The first axis of \"assistInfoForCombine\" must be consistent "
        << "with that of \"expandX\", but expandXRow=" << expandXRow << ", assistInfoForCombineRow="
        << assistInfoForCombineRow;

    int32_t assistInfoForCombineCol = assistInfoForCombine.GetShape(1);
    int32_t supportedAssistInfoForCombineCol = 3;
    ASSERT(assistInfoForCombineCol == supportedAssistInfoForCombineCol) << "The second axis of "
        << "\"assistInfoForCombine\" must be " << supportedAssistInfoForCombineCol << ", but got "
        << assistInfoForCombineCol;

    ASSERT(assistInfoForCombine.GetDataType() == DT_INT32) << "The data type of \"assistInfoForCombine\" only supports "
        << "DT_INT32, but got " << DataType2String(assistInfoForCombine.GetDataType());

    ASSERT(assistInfoForCombine.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of "
        << "\"assistInfoForCombine\" only supports ND, but got NZ";
}

void MoeDistributedCombineValidateRecvCounts(const Tensor& recvCounts)
{
    ASSERT(recvCounts.GetShape().size() == 1) << "The dim of \"recvCounts\" only supports 1, but got "
        << recvCounts.GetShape().size();

    int32_t recvCountsSize = recvCounts.GetShape(0);
    ASSERT(recvCountsSize == 1) << "The size of \"recvCounts\" must be 1, but recvCountsSize=" << recvCountsSize;

    ASSERT(recvCounts.GetDataType() == DT_INT32) << "The data type of \"recvCounts\" only supports DT_INT32, but got "
        << DataType2String(recvCounts.GetDataType());

    ASSERT(recvCounts.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"recvCounts\" only "
        << "supports ND, but got NZ";
}

void MoeDistributedCombineValidateExpertScales(const Tensor& expertScales)
{
    uint32_t supportedDim = 2;
    ASSERT(expertScales.GetShape().size() == supportedDim) << "The dim of \"expertScales\" only supports "
        << supportedDim << ", but got " << expertScales.GetShape().size();

    int32_t expertScalesRow = expertScales.GetShape(0);
    int32_t supportedExpertScalesRow1 = 8;
    int32_t supportedExpertScalesRow2 = 256;
    ASSERT((expertScalesRow == supportedExpertScalesRow1) || (expertScalesRow == supportedExpertScalesRow2)) << "The "
        << "first axis of \"expertScales\" only supports " << supportedExpertScalesRow1 << " or "
        << supportedExpertScalesRow2 << ", but got " << expertScalesRow;

    int32_t expertScalesCol = expertScales.GetShape(1);
    int32_t supportedExpertScalesCol = 8;
    ASSERT(expertScalesCol == supportedExpertScalesCol) << "The second axis of \"expertScales\" only supports "
        << supportedExpertScalesCol << ", but got " << expertScalesCol;

    ASSERT(expertScales.GetDataType() == DT_FP32) << "The data type of \"expertScales\" only supports DT_FP32, but got "
        << DataType2String(expertScales.GetDataType());

    ASSERT(expertScales.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"expertScales\" only "
        << "supports ND, but got NZ";
}

void MoeDistributedCombineValidateOut(const Tensor& out, const Tensor& expertScales, const Tensor& expandX)
{
    uint32_t supportedDim = 2;
    ASSERT(out.GetShape().size() == supportedDim) << "The dim of \"out\" only supports " << supportedDim << ", but got "
        << out.GetShape().size();

    int32_t outRow = out.GetShape(0);
    int32_t expertScalesRow = expertScales.GetShape(0);
    ASSERT(outRow == expertScalesRow) << "The first axis of \"out\" must be consistent with that of \"expertScales\", "
        << "but expertScalesRow=" << expertScalesRow << ", outRow=" << outRow;

    int32_t outCol = out.GetShape(1);
    int32_t expandXCol = expandX.GetShape(1);
    ASSERT(outCol == expandXCol) << "The second axis of \"out\" must be consistent with that of \"expandX\", but "
        << "expandXCol=" << expandXCol << ", outCol=" << outCol;

    ASSERT(out.GetDataType() == expandX.GetDataType()) << "The data type of \"out\" must be consistent with that of "
        << "\"expandX\",  but the data type of \"expandX\" is "<< DataType2String(expandX.GetDataType()) << " and the "
        << "data type of \"out\" is " << DataType2String(out.GetDataType());

    ASSERT(out.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"out\" only supports ND, but got "
        << "NZ";
}

void MoeDistributedCombineValidateGroup(const char* group)
{
    ASSERT(group != nullptr) << "\"group\" cannot be nullptr";
    int32_t groupLen = std::strlen(group);
    int32_t maxGroupLen = 128;
    ASSERT((groupLen >= 1) && (groupLen < maxGroupLen)) << "The length of \"group\" only supports [1, " << maxGroupLen
        << "), but got " << groupLen;
}

void MoeDistributedCombineValidateMoeEpWorldSize(int32_t epWorldSize)
{
    int32_t supportedEpWorldSize1 = 4;
    int32_t supportedEpWorldSize2 = 8;
    ASSERT((epWorldSize == supportedEpWorldSize1) || (epWorldSize == supportedEpWorldSize2)) << "epWorldSize only "
        << "supports " << supportedEpWorldSize1 << " or " << supportedEpWorldSize2 << ", but got " << epWorldSize;
}

void MoeDistributedCombineValidateMoeExpertNum(int32_t moeExpertNum)
{
    int32_t supportedMoeExpertNum = 160;
    ASSERT(moeExpertNum == supportedMoeExpertNum) << "moeExpertNum only supports " << supportedMoeExpertNum << ", but "
        << "got " << moeExpertNum;
}

void CreateShmemTensor(Tensor& shmemTensor, int32_t rankSize, int32_t hcclGroupIndex, DataType dataType,
    const Shape& shape, uint64_t memType = 0)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shmemShape{rankSize};
    shmemShape.insert(shmemShape.end(), shape.begin(), shape.end());
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, dataType, shmemShape);
    shmemTensor = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemTensor, SlotProperty::SHMEM_TENSOR);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, memType,
        BytesOf(dataType) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>())));
}

Tensor MoeDistributedCombineSend(
    const Tensor& in,
    const Tensor& assistInfoForCombine,
    const Tensor& recvCounts,
    const Tensor& shmemData,
    const Tensor& shmemSignal,
    int32_t topK)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1});
    auto& op = function.AddOperation(
        Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND,
        {in.GetStorage(), assistInfoForCombine.GetStorage(), recvCounts.GetStorage(), shmemData.GetStorage(),
            shmemSignal.GetStorage()},
        {out});
    DistOpAttr distOpAttr;
    distOpAttr.topK = topK;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor MoeDistributedCombineReceive(
    const Tensor& predToken,
    const Tensor& expertScales,
    const Tensor& shmemData,
    const Tensor& shmemSignal)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    int32_t batchSize = expertScales.GetShape(0);
    int32_t hiddenSize = shmemData.GetShape(3);
    auto out = std::make_shared<LogicalTensor>(function, shmemData.GetDataType(), Shape{batchSize, hiddenSize});
    function.AddOperation(
        Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE,
        {predToken.GetStorage(), expertScales.GetStorage(), shmemData.GetStorage(), shmemSignal.GetStorage()},
        {out});
    return out;
}

void MoeDistributedCombine(const Tensor& expandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& expertScales, const char* group, uint32_t epWorldSize, uint32_t moeExpertNum,
    uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& out)
{
    MoeDistributedCombineValidateExpandX(expandX, expertScales, epWorldSize, moeExpertNum);
    MoeDistributedCombineValidateAssistInfoForCombine(assistInfoForCombine, expandX);
    MoeDistributedCombineValidateRecvCounts(recvCounts);
    MoeDistributedCombineValidateExpertScales(expertScales);
    MoeDistributedCombineValidateOut(out, expertScales, expandX);
    MoeDistributedCombineValidateGroup(group);
    MoeDistributedCombineValidateMoeEpWorldSize(epWorldSize);
    MoeDistributedCombineValidateMoeExpertNum(moeExpertNum);

    int32_t batchSize = expertScales.GetShape(0);
    int32_t topK = expertScales.GetShape(1);
    int32_t hiddenSize = expandX.GetShape(1);

    int32_t shmemDataRow = topK * batchSize;
    Shape shmemDataShape = {1, shmemDataRow, hiddenSize};
    int32_t shmemSignalCol = SAME_ADDR_BYTE_SIZE / BytesOf(DataType::DT_FP32);
    Shape shmemSignalShape = {batchSize, shmemSignalCol};

    Tensor shmemData;
    Tensor shmemSignal;
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemTensor(shmemData, epWorldSize, hcclGroupIndex, expandX.GetDataType(), shmemDataShape);
        CreateShmemTensor(shmemSignal, epWorldSize, hcclGroupIndex, DT_INT32, shmemSignalShape);
    }
    LOOP("MoeDistributedCombine", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;

        int32_t expandXRow = expandX.GetShape(0);
        TileShape::Current().SetDistTile({expandXRow / AIV_NUM, AIV_NUM, expandXRow % AIV_NUM}, {hiddenSize, 1, 0},
            {0, 0, 0});
        auto sendOut = MoeDistributedCombineSend(expandX, assistInfoForCombine, recvCounts, shmemData, shmemSignal,
            topK);

        SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);
        auto shmemDataThisRank = View(shmemData, {1, 1, shmemDataRow, hiddenSize},
            std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
        auto shmemSignalThisRank = View(shmemSignal, {1, batchSize, shmemSignalCol},
            std::vector<SymbolicScalar>{thisRank, 0, 0});
        TileShape::Current().SetDistTile(
            {batchSize / AIV_NUM, AIV_NUM, batchSize % AIV_NUM}, {hiddenSize, 1, 0}, {0, 0, 0});
        out = MoeDistributedCombineReceive(sendOut, expertScales, shmemDataThisRank, shmemSignalThisRank);
    }
}
}   // namespace npu::tile_fwk::Distributed