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
 * \file inplace_process.h
 * \brief
 */

#ifndef INPLACE_PROCESS_H
#define INPLACE_PROCESS_H
#include <vector>
#include <climits>

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

const int UB_SIZE = 192 * 1024;

/*
key: Opcode类型
vaule: vector of pair, 每个pair记录了第几个输入和第几个输出存在inplace关系
*/
const std::unordered_map<Opcode, std::vector<std::pair<size_t, size_t>>> inplaceOpMap = {
    {   Opcode::OP_A_MULACC_B, {std::pair<size_t, size_t>{2, 0}}},
    {Opcode::OP_INDEX_OUTCAST, {std::pair<size_t, size_t>{2, 0}}},
    {Opcode::OP_REMOTE_REDUCE, {std::pair<size_t, size_t>{0, 0}}},
};

class InplaceProcess : public Pass {
public:
    InplaceProcess() : Pass("InplaceProcess") {}
    ~InplaceProcess() override = default;

private:
    /*
    补齐: Status PreCheck(Function &function) override;
    补齐: Status PostCheck(Function &function) override;
    */
    Status RunOnFunction(Function &function) override;
    void ProcessView(Operation &op) const;
    void ProcessAssemble(Operation &op) const;
    void AlignCopyInConsumer(std::shared_ptr<LogicalTensor> tensorGm) const;
    void ProcessReshape(Function &function, Operation &op) const;
    void ProcessInplaceOp(Function &function, Operation &op) const;
    bool ValidMeaninglessOp(const Operation &op) const;
};
} // namespace npu::tile_fwk
#endif // INPLACE_PROCESS_H