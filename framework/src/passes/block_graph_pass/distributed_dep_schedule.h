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
 * \file distributed_dep_schedule.h
 * \brief Annotate distributed tile ops with dependency scheduling metadata.
 */

#ifndef PASS_DISTRIBUTED_DEP_SCHEDULE_H
#define PASS_DISTRIBUTED_DEP_SCHEDULE_H

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {
class DistributedDepSchedule : public Pass {
public:
    DistributedDepSchedule() : Pass("DistributedDepSchedule") {}
    ~DistributedDepSchedule() override = default;

    Status RunOnFunction(Function &function) override;

private:
    Status AnnotateProgram(Function &program, uint64_t programId) const;
};
} // namespace npu::tile_fwk

#endif // PASS_DISTRIBUTED_DEP_SCHEDULE_H
