/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicpu_call.h
 * \brief
 */

#ifndef TILE_FWK_AICPU_CALL_H
#define TILE_FWK_AICPU_CALL_H

#include "tileop_common.h"

#include <type_traits>

namespace TileOp {

#define GET_CURRENT_TASKID()            ((param)->taskId)

#define AICPU_CALL_NUM_COPYOUT_RESOLVE 1
#define AICPU_CALL_NUM_BIT 16
#define AICPU_CALL_ARG_BIT 16
#define AICPU_CALL_TASK_BIT 32

INLINE uint64_t AicpuCallCreate(uint16_t callNum, uint16_t callArg, uint32_t taskId) {
    uint64_t callCode = ((uint64_t)callNum << AICPU_CALL_ARG_BIT) | (uint64_t)callArg;
    return (callCode << AICPU_CALL_TASK_BIT) | (uint64_t)taskId;
}

template <uint16_t callNum, uint16_t callArg>
TILEOP void AicpuCall(uint32_t taskId) {
    if constexpr (callNum == AICPU_CALL_NUM_COPYOUT_RESOLVE) {
        uint64_t cond = AicpuCallCreate(callNum, callArg, taskId);
        set_cond(cond);
    }
}

} // namespace TileOp

#endif // TILE_FWK_AICPU_CALL_H
