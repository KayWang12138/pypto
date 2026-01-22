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
 * \file conv_operation_impl.cpp
 * \brief Implementation of convolution operations
 */

#include "interface/configs/config_manager.h"
#include "interface/inner/pre_def.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/operator_tracer.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"

namespace npu {
namespace tile_fwk {
namespace Conv {

#define OP_CHECK(cond, exec_expr) \
    do { \
        if (cond) { \
            exec_expr; \
        } \
    } while (0)

const std::string CONV_OP_MARKER = "op_attr_is_conv_op";
const std::string CONV_FMAP_H = "op_attr_conv_fmap_h";
const std::string CONV_FMAP_W = "op_attr_conv_fmap_w";
const std::string CONV_KERNEL_H = "op_attr_conv_kernel_h";
const std::string CONV_KERNEL_W = "op_attr_conv_kernel_w";

struct ConvTensorInfo {
    std::string name;
    DataType dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> offset;
    NodeType nodeType;
    TileOpFormat format;
    MemoryType memType;

    ConvTensorInfo(const std::string &nameIn, DataType dtypeIn, const std::vector<int64_t> &shapeIn,
                   const std::vector<int64_t> &offsetIn, NodeType nodeTypeIn, TileOpFormat formatIn,
                   MemoryType memTypeIn)
        : name(nameIn),
          dtype(dtypeIn),
          shape(shapeIn),
          offset(offsetIn),
          nodeType(nodeTypeIn),
          format(formatIn),
          memType(memTypeIn)
    {
    }
};

struct ConvAttrParam {
    int64_t fmapH = 0;
    int64_t fmapW = 0;
    int64_t kernelH = 0;
    int64_t kernelW = 0;
    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t padH = 0;
    int64_t padW = 0;
};

struct ConvGraphNodes {
    LogicalTensorPtr fmapTensorPtr = nullptr;
    LogicalTensorPtr weightTensorPtr = nullptr;
    LogicalTensorPtr outTensorPtr = nullptr;
};

struct ConvTileInfo {
    int64_t tileM = 0;   // Output spatial dimension tile
    int64_t tileK = 0;   // Input channel tile
    int64_t tileN = 0;   // Output channel tile
    int64_t tileML1 = 0;
    int64_t tileKL1 = 0;
    int64_t tileNL1 = 0;
};

LogicalTensorPtr AddConvOpView(Function &function, const LogicalTensorPtr &srcTensorPtr,
                                const ConvTensorInfo &dstTensorInfo, bool isConvOp = true)
{
    OP_CHECK(true, {
        ASSERT(srcTensorPtr != nullptr) << "Source tensor for Conv OpView is nullptr." << std::endl;
    });

    LogicalTensorPtr dstTensorPtr =
        std::make_shared<LogicalTensor>(function, dstTensorInfo.dtype, dstTensorInfo.shape,
                                        SymbolicScalar::FromConcrete(dstTensorInfo.shape),
                                        dstTensorInfo.format, dstTensorInfo.name, dstTensorInfo.nodeType);

    dstTensorPtr->UpdateDynValidShape(
        GetViewValidShape(srcTensorPtr->GetDynValidShape(), dstTensorInfo.offset, {}, dstTensorInfo.shape));

    auto &viewOp = function.AddOperation(Opcode::OP_VIEW, {srcTensorPtr}, {dstTensorPtr});
    auto viewAttribute = std::make_shared<ViewOpAttribute>(
        dstTensorInfo.offset, SymbolicScalar::FromConcrete(dstTensorInfo.offset), dstTensorPtr->GetDynValidShape());
    viewAttribute->SetToType(dstTensorInfo.memType);
    viewOp.SetOpAttribute(viewAttribute);

    // Add conv marker attribute
    if (isConvOp) {
        viewOp.SetAttribute(CONV_OP_MARKER, true);
    }

    return dstTensorPtr;
}

void CheckConvOperands(const Tensor &fmap, const Tensor &weight, const ConvAttrParam &param)
{
    // Check fmap is 5D: [N, C1, H, W, C0]
    OP_CHECK(true, {
        ASSERT(fmap.GetShape().size() == 5)
            << "Fmap must be 5D tensor, got: " << fmap.GetShape().size() << "D" << std::endl;
    });

    // Check weight is 4D: [Co, Ci, Kh, Kw]
    OP_CHECK(true, {
        ASSERT(weight.GetShape().size() == 4)
            << "Weight must be 4D tensor, got: " << weight.GetShape().size() << "D" << std::endl;
    });

    // Check shape validity
    OP_CHECK(true, {
        ASSERT(fmap.GetShape()[2] > 0 && fmap.GetShape()[3] > 0)
            << "Invalid fmap spatial dimensions: H=" << fmap.GetShape()[2]
            << ", W=" << fmap.GetShape()[3] << std::endl;
    });

    OP_CHECK(true, {
        ASSERT(weight.GetShape()[2] > 0 && weight.GetShape()[3] > 0)
            << "Invalid kernel dimensions: Kh=" << weight.GetShape()[2]
            << ", Kw=" << weight.GetShape()[3] << std::endl;
    });
}

void ConstructConvTileGraph(Function &function, const TileShape &tileShape,
                            const std::vector<LogicalTensorPtr> &operandVec,
                            const LogicalTensorPtr &outTensorPtr, const Operation &op)
{
    OP_CHECK(true, {
        ASSERT(operandVec.size() >= 2)
            << "Conv operation requires at least 2 operands, got: " << operandVec.size() << std::endl;
    });

    ConvAttrParam convParam;
    if (op.HasAttr(CONV_FMAP_H)) {
        convParam.fmapH = op.GetIntAttribute(CONV_FMAP_H);
    }
    if (op.HasAttr(CONV_FMAP_W)) {
        convParam.fmapW = op.GetIntAttribute(CONV_FMAP_W);
    }
    if (op.HasAttr(CONV_KERNEL_H)) {
        convParam.kernelH = op.GetIntAttribute(CONV_KERNEL_H);
    }
    if (op.HasAttr(CONV_KERNEL_W)) {
        convParam.kernelW = op.GetIntAttribute(CONV_KERNEL_W);
    }

    LogicalTensorPtr fmapPtr = operandVec[0];
    LogicalTensorPtr weightPtr = operandVec[1];

    auto &cubeTile = tileShape.GetCubeTile();
    
    // First level tiling: GM to L1
    // Tile the fmap and weight to L1 memory
    int64_t fmapN = fmapPtr->shape[0];
    int64_t fmapC1 = fmapPtr->shape[1];
    int64_t fmapH = fmapPtr->shape[2];
    int64_t fmapW = fmapPtr->shape[3];
    int64_t fmapC0 = fmapPtr->shape[4];

    int64_t tileML1 = cubeTile.m[1];
    int64_t tileKL1 = cubeTile.k[1];
    int64_t tileNL1 = cubeTile.n[1];

    // Iterate over tiles for GM to L1 transfer
    for (int64_t n = 0; n < fmapN; n++) {
        for (int64_t c1Idx = 0; c1Idx < fmapC1; c1Idx += (tileKL1 / fmapC0)) {
            int64_t c1Size = std::min(fmapC1 - c1Idx, tileKL1 / fmapC0);
            
            // Create L1 view for fmap (GM to L1)
            std::vector<int64_t> fmapL1Shape = {1, c1Size, fmapH, fmapW, fmapC0};
            std::vector<int64_t> fmapL1Offset = {n, c1Idx, 0, 0, 0};
            ConvTensorInfo fmapL1Info("fmap_l1", fmapPtr->Datatype(), fmapL1Shape, fmapL1Offset,
                                      NodeType::LOCAL, fmapPtr->Format(), MemoryType::MEM_L1);
            LogicalTensorPtr fmapL1Ptr = AddConvOpView(function, fmapPtr, fmapL1Info, true);

            // Second level tiling: L1 to L0
            // Tile from L1 to L0A for computation
            int64_t tileML0 = cubeTile.m[0];
            int64_t tileKL0 = cubeTile.k[0];
            int64_t tileNL0 = cubeTile.n[0];

            for (int64_t hIdx = 0; hIdx < fmapH; hIdx += tileML0) {
                for (int64_t wIdx = 0; wIdx < fmapW; wIdx += tileML0) {
                    int64_t hSize = std::min(fmapH - hIdx, tileML0);
                    int64_t wSize = std::min(fmapW - wIdx, tileML0);

                    // Create L0 view for fmap (L1 to L0A)
                    std::vector<int64_t> fmapL0Shape = {1, c1Size, hSize, wSize, fmapC0};
                    std::vector<int64_t> fmapL0Offset = {0, 0, hIdx, wIdx, 0};
                    ConvTensorInfo fmapL0Info("fmap_l0a", fmapL1Ptr->Datatype(), fmapL0Shape, fmapL0Offset,
                                              NodeType::LOCAL, fmapL1Ptr->Format(), MemoryType::MEM_L0A);
                    LogicalTensorPtr fmapL0Ptr = AddConvOpView(function, fmapL1Ptr, fmapL0Info, true);
                }
            }
        }
    }

    // Create similar tiling for weight tensor (GM to L1 to L0B)
    int64_t weightCo = weightPtr->shape[0];
    int64_t weightCi = weightPtr->shape[1];
    int64_t weightKh = weightPtr->shape[2];
    int64_t weightKw = weightPtr->shape[3];

    for (int64_t coIdx = 0; coIdx < weightCo; coIdx += tileNL1) {
        int64_t coSize = std::min(weightCo - coIdx, tileNL1);

        // Create L1 view for weight (GM to L1)
        std::vector<int64_t> weightL1Shape = {coSize, weightCi, weightKh, weightKw};
        std::vector<int64_t> weightL1Offset = {coIdx, 0, 0, 0};
        ConvTensorInfo weightL1Info("weight_l1", weightPtr->Datatype(), weightL1Shape, weightL1Offset,
                                    NodeType::LOCAL, weightPtr->Format(), MemoryType::MEM_L1);
        LogicalTensorPtr weightL1Ptr = AddConvOpView(function, weightPtr, weightL1Info, true);

        // Create L0 view for weight (L1 to L0B)
        std::vector<int64_t> weightL0Shape = {std::min(coSize, tileNL0), weightCi, weightKh, weightKw};
        std::vector<int64_t> weightL0Offset = {0, 0, 0, 0};
        ConvTensorInfo weightL0Info("weight_l0b", weightL1Ptr->Datatype(), weightL0Shape, weightL0Offset,
                                    NodeType::LOCAL, weightL1Ptr->Format(), MemoryType::MEM_L0B);
        LogicalTensorPtr weightL0Ptr = AddConvOpView(function, weightL1Ptr, weightL0Info, true);
    }
}

void AddConvNode(const LogicalTensorPtr &fmapTensorPtr, const LogicalTensorPtr &weightTensorPtr,
                 const LogicalTensorPtr &outTensorPtr, const ConvAttrParam &attrParam)
{
    auto &curFunc = *Program::GetInstance().GetCurrentFunction();
    
    auto &convOp = curFunc.AddOperation(Opcode::OP_CONV, {fmapTensorPtr, weightTensorPtr}, {outTensorPtr});
    
    // Set conv-specific attributes
    convOp.SetAttribute(CONV_FMAP_H, attrParam.fmapH);
    convOp.SetAttribute(CONV_FMAP_W, attrParam.fmapW);
    convOp.SetAttribute(CONV_KERNEL_H, attrParam.kernelH);
    convOp.SetAttribute(CONV_KERNEL_W, attrParam.kernelW);
    convOp.SetAttribute(CONV_OP_MARKER, true);
}

Tensor ConstructConvTensorGraph(DataType outType, const Tensor &fmap, const Tensor &weight,
                                const ConvAttrParam &param)
{
    CheckConvOperands(fmap, weight, param);

    // Calculate output shape
    // fmap: [N, C1, H, W, C0]
    // output: [N, Co1, H_out, W_out, Co0]
    int64_t n = fmap.GetShape()[0];
    int64_t fmapC1 = fmap.GetShape()[1];
    int64_t fmapH = fmap.GetShape()[2];
    int64_t fmapW = fmap.GetShape()[3];
    int64_t fmapC0 = fmap.GetShape()[4];

    int64_t weightCo = weight.GetShape()[0];
    int64_t weightKh = weight.GetShape()[2];
    int64_t weightKw = weight.GetShape()[3];

    // For simplicity, assume same padding (output size = input size)
    int64_t outH = fmapH;
    int64_t outW = fmapW;
    
    // Assuming C0 = 16 (common for NZ format)
    const int64_t CO_0 = 16;
    int64_t outCo1 = (weightCo + CO_0 - 1) / CO_0;

    Tensor output(outType, {n, outCo1, outH, outW, CO_0}, "conv_output");

    ConvAttrParam convParam = param;
    convParam.fmapH = fmapH;
    convParam.fmapW = fmapW;
    convParam.kernelH = weightKh;
    convParam.kernelW = weightKw;

    AddConvNode(fmap.GetStorage(), weight.GetStorage(), output.GetStorage(), convParam);

    return output;
}

Tensor Conv2D(DataType outType, const Tensor &fmap, const Tensor &weight)
{
    ConvAttrParam param;
    return ConstructConvTensorGraph(outType, fmap, weight, param);
}

} // namespace Conv
} // namespace tile_fwk
} // namespace npu
