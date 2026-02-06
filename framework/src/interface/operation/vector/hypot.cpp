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
 * \file hypot.cpp
 * \brief
 */

#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

/**
 * @brief Hypot 的切分实现逻辑
 */
void TiledHypotOperationImpl(Function &function, const TileShape &tileShape, size_t cur, Input &input1,
    Input &input2, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    
    // 1. 递归终止条件（叶子节点）
    if (cur == result->shape.size()) {
        // [FIX Error 1] 使用 .GetStorage()->View(...)
        auto inputTile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor.GetStorage()->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        // --- 开始：合并内存申请逻辑 ---
        
        // 1.1 获取基础信息
        // 为了兼容 FP16 模式下 Cast 到 FP32 进行计算，我们按 float (4 bytes) 来计算空间需求
        size_t element_size = sizeof(float); 

        // 1.2 计算当前 Tile 的元素个数
        int64_t num_elements = 1;
        for (auto dim : resultTileInfo.shape) {
            num_elements *= dim;
        }

        // 1.3 设定对齐大小 (通常 NPU 要求 32 字节对齐)
        const size_t ALIGN_SIZE = 32;

        // 1.4 计算单块内存所需的字节数
        size_t raw_size_bytes = num_elements * element_size;
        // 进行向上对齐：((size + 31) / 32) * 32
        size_t aligned_size_bytes = ((raw_size_bytes + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;

        // 1.5 计算总内存大小
        // 我们需要两块这样的内存：
        // FP32模式: 用于 max 和 min
        // FP16模式: 用于 cast 后的 input1 和 input2 (fp32)
        size_t total_bytes = 2 * aligned_size_bytes;

        // 1.6 创建合并后的临时 Tensor
        // 类型固定为 DT_UINT8，形状为 [total_bytes] 的一维向量
        std::vector<int64_t> tmp_shape = {static_cast<int64_t>(total_bytes)};
        auto tmp_tensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tmp_shape);

        // --- 结束：合并内存申请逻辑 ---

        // 2. 添加算子
        // 输出变为：{resultTile, tmp_tensor}
        // [FIX Error 2] 显式转换或者保持原样，修正了前面的 inputTile 获取方式后，这里通常能自动推导
        function.AddOperation(Opcode::OP_HYPOT, {inputTile1, inputTile2}, {resultTile, tmp_tensor});
        return;
    }

    // 3. 递归切分逻辑
    auto &vecTile = tileShape.GetVecTile();
    // [FIX Logic] 使用 vecTile[cur] 作为步长，注意处理边界
    int64_t step = vecTile[cur];
    
    for (int i = 0; i < result->shape[cur]; i += step) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - i, step);

        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], step);

        input2.tileInfo.offset[cur] = i % input2.tensor.GetShape()[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor.GetShape()[cur] - input2.tileInfo.offset[cur], step);

        TiledHypotOperationImpl(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}

/**
 * @brief Hypot 切分入口函数
 */
void TiledHypotOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result) {
    
    // 广播处理 Lambda，参考 Compare 实现
    auto broadcastOperand = [&](LogicalTensorPtr&operand,LogicalTensorPtr&other) {
        auto dstShape = result->shape;
        if (operand->shape == dstShape) {
            return;
        }
        auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
        Expand(function, tileShape, operand, {other}, expanded);
        operand = expanded;
    };

    CheckBinOpOperandsValid(operand1, operand2);
    broadcastOperand(operand1, operand2);
    broadcastOperand(operand2, operand1);

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    
    auto input1 = Input{operand1, tileInfo1};
    auto input2 = Input{operand2, tileInfo2};

    TiledHypotOperationImpl(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

/**
 * @brief Graph 层构建入口
 */
LogicalTensorPtr TensorHypotOperation(Function &function, const Tensor &self, const Tensor &other) {
    auto operandT1 = self.GetStorage();
    auto operandT2 = other.GetStorage();

    if (operandT1->shape.size() != operandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operandT1, operandT2);
        operandT1 = BinaryOperationBroadCast(operandT1, broadCastShape);
        operandT2 = BinaryOperationBroadCast(operandT2, broadCastShape);
    }

    std::vector<int64_t> resultShape = BinaryOperationResultShape(operandT1, operandT2);
    
    std::vector<SymbolicScalar> resultValidShape;
    if (!operandT1->GetDynValidShape().empty() && !operandT2->GetDynValidShape().empty()) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operandT1->shape[i]) {
                resultValidShape.push_back(operandT1->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operandT2->GetDynValidShape()[i]);
            }
        }
    }

    auto result = std::make_shared<LogicalTensor>(function, operandT1->Datatype(), resultShape, resultValidShape);

    function.AddOperation(Opcode::OP_HYPOT, {operandT1, operandT2}, {result});

    return result;
}

Tensor Hypot(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(HypotOperation, *Program::GetInstance().GetCurrentFunction(), self, other);
}

void HypotOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, [[maybe_unused]] const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledHypotOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_HYPOT, Opcode::OP_HYPOT, HypotOperationTileFunc);

} // namespace npu::tile_fwk