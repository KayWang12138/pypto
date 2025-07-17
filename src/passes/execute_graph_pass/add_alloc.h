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
 * \file add_alloc.h
 * \brief
 */

#ifndef PASS_ADD_ALLOC_H
#define PASS_ADD_ALLOC_H
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_utils/pass_utils.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace npu::tile_fwk {
class AddAllocPass : public Pass {
public:
    AddAllocPass() : Pass("AddAllocPass") {}
    ~AddAllocPass() override = default;

    Status PreCheck(Function &function) override;

private:
    Status RunOnFunction(Function &function) override {
        ASLOGI("===> Start AddAllocPass.");
        for (auto &program : function.rootFunc_->programs_) {
            if (AddAndCheckAlloc(*program.second) != SUCCESS) { ALOG_ERROR_F("AddAndCheckAlloc failed."); return FAILED; }
        }
        ASLOGI("===> End AddAllocPass.");
        return SUCCESS;
    }
    struct TensorAllocMsg {
        std::vector<Operation *> producer;
        bool isAllocated{false};
        MemoryType memType;
        int memId;
    };
    // 按color去判断是否需要插入alloc
    Status AddAndCheckAlloc(Function &function);
    Status FindTensorAllocMsg(Operation *op, std::unordered_map<int, TensorAllocMsg> &tensorAllocMsgMap) const;
    Status CreateAllocNode(const TensorAllocMsg &tensorAllocMsg, Function &function);
    const std::unordered_map<MemoryType, Opcode> allocOpcodeMap = {
        {MemoryType::MEM_L0A, Opcode::OP_L0A_ALLOC},
        { MemoryType::MEM_UB,  Opcode::OP_UB_ALLOC},
        {MemoryType::MEM_VECTOR_REG, Opcode::OP_REG_ALLOC},
        { MemoryType::MEM_L1,  Opcode::OP_L1_ALLOC},
        {MemoryType::MEM_L0B, Opcode::OP_L0B_ALLOC},
        {MemoryType::MEM_L0C, Opcode::OP_L0C_ALLOC},
        {MemoryType::MEM_BT, Opcode::OP_BT_ALLOC},
        {MemoryType::MEM_FIX, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_QUANT_PRE, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_RELU_PRE, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_RELU_POST, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_QUANT_POST, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_ELT_ANTIQ, Opcode::OP_FIX_ALLOC},
        {MemoryType::MEM_FIX_MTE2_ANTIQ, Opcode::OP_FIX_ALLOC}
    };
};


} // namespace npu::tile_fwk
#endif // PASS_ADD_ALLOC_H