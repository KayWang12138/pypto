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
 * \file comm_barrier_manager.h
 * \brief
 */

#ifndef COMM_BARRIER_MANAGER_H
#define COMM_BARRIER_MANAGER_H

#include <memory>
#include <vector>
#include <cstdint>
#include <optional>
#include "tilefwk/data_type.h"
#include "interface/utils/common.h"
#include "tilefwk/tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {
namespace Distributed {

class PerGroupManager {
public:
    PerGroupManager() {};
    ~PerGroupManager() {};

    void GenBarrierBeforeOp(Function &function, std::shared_ptr<LogicalTensor> &inTensor)
    {
        if (lastBarrierTensor_.has_value()) {
            (void)function.AddOperation("DEPEND_ON", {lastBarrierTensor_.value()}, {inTensor});
        }
    }

    void GenBarrierAfterOp(Function &function, std::shared_ptr<LogicalTensor> &outTensor)
    {
        if (!nextBarrierTensor_.has_value()) {
            auto tensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, std::vector<int>{1});
            tensor->SetIsDummy(true);
            nextBarrierTensor_ = std::make_optional(tensor);
        }
        (void)function.AddOperation("DEPEND_ON", {outTensor}, {nextBarrierTensor_.value()});
    }

    void BarrierFinish()
    {
        lastBarrierTensor_ = nextBarrierTensor_;
        nextBarrierTensor_ = std::nullopt;
    }

private:
    std::optional<std::shared_ptr<LogicalTensor>> lastBarrierTensor_{std::nullopt};
    std::optional<std::shared_ptr<LogicalTensor>> nextBarrierTensor_{std::nullopt};
};

class CommBarrierManager {
public:
    CommBarrierManager() {};
    ~CommBarrierManager() {};

    PerGroupManager &PerGroup(uint32_t groupIndex) { return perGroupMgr_[groupIndex]; }

private:
    PerGroupManager perGroupMgr_[DIST_COMM_GROUP_NUM];
};

} // namespace Distributed
} // namespace npu::tile_fwk

#endif
