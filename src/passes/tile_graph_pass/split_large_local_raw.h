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
 * \file split_large_local_raw.h
 * \brief
 */

#ifndef SPLIT_LARGE_LOCAL_RAW_PASS_H
#define SPLIT_LARGE_LOCAL_RAW_PASS_H
#include <vector>

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {

const std::string LOCAL_RAW_SYMBOL_PREFIX = "raw_for_";

class SplitLargeLocalRawTensor : public Pass {
public:
    SplitLargeLocalRawTensor() : Pass("SplitLargeLocalRawTensor") {}
    ~SplitLargeLocalRawTensor() override = default;

private:
    Status RunOnFunction(Function &function) override;
    void UpdateConsumerView(Function &function, const LogicalTensorPtr &logicalTensor, std::vector<int64_t> &diff) const;
    void UpdateProducerAssemble(Function &function, const LogicalTensorPtr &logicalTensor, std::vector<int64_t> &diff) const;
    void SplitLargeLocalRaw(Function &function) const;
    bool ShouldProcessTensor(Function& function, const LogicalTensorPtr& tensor) const;
    std::vector<int64_t> UpdateOffset(std::vector<int64_t> &offset, std::vector<int64_t> &diff) const;
};
} // namespace npu::tile_fwk
#endif // SPLIT_LARGE_LOCAL_RAW_PASS_H