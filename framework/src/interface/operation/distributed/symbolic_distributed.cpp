/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file symbolic_distributed.cpp
 * \brief
 */

#include <string>
#include "interface/tensor/symbol_handler.h"
#include "interface/tensor/symbolic_scalar.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/symbolic_distributed.h"
#include "tilefwk/comm_group_recorder.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
SymbolicScalar GetHcclRankId(const std::string& groupName)
{
    int32_t hcclGroupIndex =
        static_cast<int32_t>(Distributed::CommGroupRecorder::GetInstance().Input(std::string(groupName)));
    std::string name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetHcclRankId);
    name = AddRuntimePrefix(name);
    SymbolicScalar getHcclRankId(name);
    return getHcclRankId(hcclGroupIndex);
}

SymbolicScalar BindTensor(uint64_t groupIndex, uint64_t memType, uint64_t size, uint64_t maxTileNum)
{
    std::string name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::BindTensor);
    name = AddRuntimePrefix(name);
    SymbolicScalar bindTensor(name);
    static uint64_t index = 0;
    return bindTensor(groupIndex, memType, size, maxTileNum, index++);
}

SymbolicScalar GetHcclRankIdV2(const Tensor &t) {
    std::string name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetHcclRankIdV2);
    name = AddRuntimePrefix(name);
    SymbolicScalar getHcclRankId(name);
    return getHcclRankId(GetInputDataAddr(t));
}

SymbolicScalar BindTensorV2(const Tensor &t, uint64_t memType, uint64_t size, uint64_t maxTileNum) {
    std::string name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::BindTensorV2);
    name = AddRuntimePrefix(name);
    SymbolicScalar bindTensor(name);
    static uint64_t index = 0;
    return bindTensor(GetInputDataAddr(t), memType, size, maxTileNum, index++);
}

SymbolicScalar GetInputDataAddr(const Tensor &t) {
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->TensorRead(t);
    std::string getInputDataAddrName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputDataAddr);
    int inputIndex = slotManager->GetInputIndex(t);
    ASSERT(inputIndex >= 0 && static_cast<size_t>(inputIndex) < slotManager->GetInputNameList().size()) <<
        "Tensor" << t.GetStorage(false)->GetRawTensor()->GetSymbol() << " is not in input tensor list";
    std::string inputName = slotManager->GetInputNameList()[inputIndex];
    getInputDataAddrName = AddRuntimePrefix(getInputDataAddrName);
    inputName =AddArgPrefix(inputName);
    SymbolicScalar getInputDataAddr(getInputDataAddrName);
    SymbolicScalar input(inputName);
    return getInputDataAddr(input);
}
} // namespace npu::tile_fwk
