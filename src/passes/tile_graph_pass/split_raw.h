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
 * \file split_raw.h
 * \brief
 */

#ifndef SPLIT_RAW_PASS_H
#define SPLIT_RAW_PASS_H
#include <vector>

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu {
namespace tile_fwk {

const std::string SYMBOL_PREFIX = "raw_for_";

class SplitRawTensor : public Pass {
public:
    SplitRawTensor() : Pass("SplitRawTensor") {}
    ~SplitRawTensor() override = default;

private:
    Status RunOnFunction(Function &function) override;
    void UpdateConsumerView(Function &function, const LogicalTensorPtr &logicalTensor) const;
    void UpdateProducerAssemble(Function &function, const LogicalTensorPtr &logicalTensor) const;
    void SplitRaw(Function &function) const;
    bool ShouldProcessTensor(Function& function, const LogicalTensorPtr& singleTensor) const;
    std::vector<int64_t> UpdateOffset(std::vector<int64_t> &offset, const std::vector<int64_t> &diff) const;
    std::vector<SymbolicScalar> UpdateDynOffset(std::vector<SymbolicScalar> &offset, const std::vector<SymbolicScalar> &diff) const;
};
} // namespace tile_fwk
} // namespace npu
#endif // SPLIT_RAW_PASS_H