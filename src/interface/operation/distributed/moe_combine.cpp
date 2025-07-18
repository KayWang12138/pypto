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
 * \file moe_combine.cpp
 * \brief
 */

#include <functional>
#include <memory>
#include <vector>
#include "interface/operation/operation.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "distributed_common.h"
#include "comm_barrier_manager.h"

namespace npu::tile_fwk {
namespace Distributed {

constexpr int32_t COMBINE_INFO_NUM = 3;

struct FFN2AttnTileArgs {
    Function &function;
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand;
    TilingInfo &tilingInfo;
    const TensorTileInfo &tileInfo;
    const CommGroupInfo &groupInfo;
    const std::string &tilingSymbol;
    const int topk;
};

struct AttnCombineTileArgs {
    Function &function;
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand;
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand;
    TilingInfo &tilingInfo;
    const TensorTileInfo &tileInfo;
    const CommGroupInfo &groupInfo;
    const std::string &tilingSymbol;
};

void DealTileFFN2Attn(FFN2AttnTileArgs &args)
{
    std::vector<int> shape = {args.tilingInfo.rowShape, args.tilingInfo.colShape};
    std::vector<int> offset = {args.tilingInfo.rowOffset, args.tilingInfo.colOffset};

    std::shared_ptr<LogicalTensor> in = args.iOperand[DIST_INDEX_ZERO];
    std::shared_ptr<LogicalTensor> combineInfo = args.iOperand[DIST_INDEX_ONE];
    std::shared_ptr<LogicalTensor> tilingTensor = args.iOperand[DIST_INDEX_TWO];
    auto inTile = in->View(args.function, shape, offset);

    std::vector<int> flagShape =  {1, 64};
    auto flag = std::make_shared<LogicalTensor>(args.function, DataType::DT_INT32, flagShape);

    // CombineInfo不切tile
    OpArgs<TilingInfo> opArgs = {"MOE_FFN_TO_ATTN", {inTile, combineInfo, flag}, {}, tilingTensor, args.tilingSymbol,
        std::make_optional(args.tilingInfo), std::nullopt};
    auto& op = AddOperation(args.function, opArgs);
    std::string extraParam = std::to_string(args.topk);
    op.SetAttr("extraTemplateParam", extraParam);
}

void DealTileAttnCombine(AttnCombineTileArgs &args)
{
    std::vector<int> shape = {args.tilingInfo.rowShape, args.tilingInfo.colShape};
    std::vector<int> offset = {args.tilingInfo.rowOffset, args.tilingInfo.colOffset};

    std::shared_ptr<LogicalTensor> scale = args.iOperand[0];
    std::shared_ptr<LogicalTensor> tilingTensor = args.iOperand[1];
    std::shared_ptr<LogicalTensor> out = args.oOperand[0];
    auto outTile = out->View(args.function, shape, offset);

    // 申请UB, 两块fp32 UB, 用来进行fp32计算
    std::vector<int> ubShape =  {1, args.tilingInfo.colShape};
    auto mulFP32 = std::make_shared<LogicalTensor>(args.function, DataType::DT_FP32, ubShape);
    auto sumFP32 = std::make_shared<LogicalTensor>(args.function, DataType::DT_FP32, ubShape);

    int bs = scale->shape[0];
    int topk = scale->shape[1];
    std::vector<int> scaleFlattenShape = {1, bs * topk};
    // 由于UBCopyIn 拷贝 [8, 4]这种shape有问题，临时做法，传入scale的GM地址，和申请一个UB来拷贝scale
    auto scaleFlatten = std::make_shared<LogicalTensor>(args.function, DataType::DT_FP32, scaleFlattenShape);

    OpArgs<TilingInfo> opArgs = {"MOE_ATTN_COMBINE", {scale, scaleFlatten, mulFP32, sumFP32}, {outTile}, tilingTensor,
        args.tilingSymbol, std::make_optional(args.tilingInfo), std::nullopt};
    auto& op = AddOperation(args.function, opArgs);
    std::string extraParam = std::to_string(topk) + ", " + std::to_string(bs); // 临时，由于在kernel进行copy，需要知道多一个bs的模板参数
    op.SetAttr("extraTemplateParam", extraParam);
}

void TiledMoeFFN2Attn(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand, const Operation &op)
{
    CommGroupInfo groupInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    std::string tilingSymbol;
    op.GetAttr("tiling_tensor_symbol", tilingSymbol);
    TilingInfo tilingInfo;
    op.GetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    int topk;
    op.GetAttr("topk", topk);

    std::shared_ptr<LogicalTensor> in = iOperand[0];

    TensorTileInfo tileInfo;
    std::vector<int> shape = in->GetShape();
    int row = shape[0];
    int col = shape[1];
    CheckAndGetTileInfo(row, col, tileShape, tileInfo);

    FFN2AttnTileArgs args{function, iOperand, tilingInfo, tileInfo, groupInfo, tilingSymbol, topk};
    TileColAndRowProcess<FFN2AttnTileArgs>(DealTileFFN2Attn, args);
}

void MoeFFN2Attn(const Tensor &in, const Tensor &combineInfo, const int tileCnt, const int topk, const char *group)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    std::vector<int32_t> tilingShape = {1, tileCnt * static_cast<int>(sizeof(TilingInfo) / sizeof(int))};
    const std::string tilingSymbol = function.GetDistTilingManager()->CreateTilingStorage("MoeFFN2Attn",
        tilingShape[1]);
    Tensor tilingTensor(DataType::DT_INT32, tilingShape, tilingSymbol);

    auto &oper = function.AddOperation("MOE_FFN_TO_ATTN",
        {in.GetStorage(), combineInfo.GetStorage(), tilingTensor.GetStorage()}, {});
    oper.SetAttr("tiling_tensor_symbol", tilingTensor.GetStorage()->Symbol());
    const TileShape &tileShape = Program::GetInstance().GetTileShape();
    CommGroupInfo groupInfo(group, tileShape);
    TilingInfo tilingInfo;
    oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    oper.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
    oper.SetAttr("topk", topk);
}

void TiledMoeAttnCombine(Function &function, const TileShape &tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>> &oOperand,
    const Operation &op)
{
    CommGroupInfo groupInfo;
    op.GetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    std::string tilingSymbol;
    op.GetAttr("tiling_tensor_symbol", tilingSymbol);
    TilingInfo tilingInfo;
    op.GetAttr(OpAttributeKey::distTilingInfo, tilingInfo);

    std::shared_ptr<LogicalTensor> out = oOperand[0];
    TensorTileInfo tileInfo;
    std::vector<int> shape = out->GetShape();
    int row = shape[0];
    int col = shape[1];
    CheckAndGetTileInfo(row, col, tileShape, tileInfo);

    AttnCombineTileArgs args{function, iOperand, oOperand, tilingInfo, tileInfo, groupInfo, tilingSymbol};
    TileColAndRowProcess<AttnCombineTileArgs>(DealTileAttnCombine, args);
}

void MoeAttnCombine(Tensor &out, const Tensor &scale, int tileCnt, const char *group)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();

    std::vector<int32_t> tilingShape = {1, tileCnt * static_cast<int>(sizeof(TilingInfo) / sizeof(int))};
    const std::string tilingSymbol = function.GetDistTilingManager()->CreateTilingStorage("MoeAttnCombine",
        tilingShape[1]);
    Tensor tilingTensor(DataType::DT_INT32, tilingShape, tilingSymbol);

    auto &oper = function.AddOperation("MOE_ATTN_COMBINE",
        {scale.GetStorage(), tilingTensor.GetStorage()},
        {out.GetStorage()});
    oper.SetAttr("tiling_tensor_symbol", tilingTensor.GetStorage()->Symbol());
    const TileShape &tileShape = Program::GetInstance().GetTileShape();
    CommGroupInfo groupInfo(group, tileShape);
    TilingInfo tilingInfo;
    oper.SetAttr(OpAttributeKey::commGroupInfo, groupInfo);
    oper.SetAttr(OpAttributeKey::distTilingInfo, tilingInfo);
}

Tensor MoeCombine(const Tensor &in, const Tensor &scale, const Tensor &combineInfo, const char *group)
{
    std::vector<int> inShape = in.GetShape();
    int expandBS = inShape[0];
    int h = inShape[1];
    int bs = scale.GetShape(0);
    int topk = scale.GetShape(1);

    Tensor out(in.GetDataType(), {bs, h}, "MoeCombineOut", NodeType::OUTCAST);

    // 只切row
    Program::GetInstance().GetTileShape().SetDistTileShapes({4, expandBS / 4, expandBS % 4}, {h, 1, 0}, {0, 0, 0});
    int tileCnt = expandBS / 4 + (expandBS % 4 ? 1 : 0);
    MoeFFN2Attn(in, combineInfo, tileCnt, topk, group);

    // 一个tile处理一个tokenTensor
    Program::GetInstance().GetTileShape().SetDistTileShapes({1, bs, 0}, {h, 1, 0}, {0, 0, 0});
    MoeAttnCombine(out, scale, bs, group);

    return out;
}
} // namespace Distributed
} // namespace npu::tile_fwk