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
 * \file assemble_checker.cpp
 * \brief
 */

#include <utility>
#include "assemble_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "AssembleChecker"

namespace npu {
namespace tile_fwk {
/*
    检查在input->assemble->output的场景中，input是否存在覆盖output中同一数据块的情况。
    （这可能由于两块数据到达时间不同，导致不确定的行为）

    大体判断逻辑：
    遍历每一个tensor(后称output)， 如果tensor的生产者op不是assemble则直接跳过，
    否则依次遍历assemble得到输入tensor(后称input)的形状和assemble的offset，<inputShape, offset>
    并计算出这个tensor在output的rawTensor的覆盖范围：其中每个维度i都覆盖了闭区间[offset[i], inputShape[i] + offset[i] - 1]
    每得到一个覆盖范围都与已记录的覆盖区间进行比较，如果每个维度都存在交集，则意味着这两个形状存在重叠，即assemble存在overlap。
    否则将当前tensor的覆盖范围添加到记录的覆盖范围中进行后续比较。output的所有input都未出现交集则校验通过。
*/
Status AssembleChecker::CheckAssembleOverlap(Function &function) {
    for (const auto &tMap : function.GetTensorMap().tensorMap_) {
        for (const auto &outputTensor : tMap.second) {
            if (outputTensor->GetProducers().size() == 0 || 
                (*outputTensor->GetProducers().begin())->GetOpcode() != Opcode::OP_ASSEMBLE){
                continue;
            }
            coveredAreas.clear();
            for (const auto &assembleOp : outputTensor->GetProducers()) {
                if (assembleOp->GetOpcode() != Opcode::OP_ASSEMBLE) {
                    continue;
                }
                auto assembleOffset = dynamic_cast<AssembleOpAttribute *>(assembleOp->GetOpAttribute().get())->GetToOffset();
                auto inputTensor = assembleOp->GetIOperands().front();
                auto inputShape = inputTensor->GetShape();
                std::vector<std::pair<int64_t, int64_t>> curInputArea;
                if(assembleOffset.size() != inputShape.size()) {
                    APASS_LOG_ERROR_F(Elements::Tensor, "Dimension of assemble op[%d]'s toOffset(%s) "
                        "varies from its input[%d]'s shape(%s); Please check the function graph.",
                        assembleOp->GetOpMagic(), CommonUtils::ContainerToStr(assembleOffset).c_str(),
                        inputTensor->GetMagic(), CommonUtils::ContainerToStr(inputShape).c_str());
                    return FAILED;
                }
                for (size_t i = 0; i < inputShape.size(); i++){
                    curInputArea.emplace_back(assembleOffset[i], assembleOffset[i] + inputShape[i] - 1);
                }

                // 判断是否有重叠
                if (OverlapCurInput(curInputArea)) {
                    APASS_LOG_ERROR_F(Elements::Tensor, "Overlap input2: shape:%s offset:%s; Please check the function graph; Please check Tensor[%d] and its input.",
                        CommonUtils::ContainerToStr(inputShape).c_str(), CommonUtils::ContainerToStr(assembleOffset).c_str(), outputTensor->GetMagic());
                    return FAILED;
                }

                // 将覆盖区域添加到记录
                coveredAreas.emplace_back(std::move(curInputArea));
            }
        }
    }
    return SUCCESS;
}

bool AssembleChecker::OverlapCurInput(const std::vector<std::pair<int64_t, int64_t>> &curInputArea) {
    for (const auto& recordedArea : coveredAreas) {
        bool overlap = std::equal(curInputArea.begin(), curInputArea.end(),
            recordedArea.begin(),
            [](const auto& a, const auto& b) {
                return a.second >= b.first && a.first <= b.second;
            });
        if (overlap) {
            // 计算重叠的input的shape和offset
            Shape recordedShape;
            Shape recordedOffset;
            for (const auto& recordedDim : recordedArea){
                recordedOffset.emplace_back(recordedDim.first);
                recordedShape.emplace_back(recordedDim.second + 1 - recordedDim.first);
            }
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor produced by assemble has overlap inputs. Overlap input1: shape:%s offset:%s.",
                CommonUtils::ContainerToStr(recordedShape).c_str(), CommonUtils::ContainerToStr(recordedOffset).c_str());
            return true;
        }
    }
    return false;
}
} // namespace tile_fwk
} // namespace npu