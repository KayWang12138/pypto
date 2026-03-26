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
 * \file calc_distributed.cpp
 * \brief 通信算子执行函数实现
 */

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../utils/string_utils.h"
#include "function.h"
#include "shmem_tensor_data.h"
#include "operation.h"
#include "../operation/distributed/distributed_common.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/tilefwk.h"
#include "calc.h"

namespace npu::tile_fwk {

static int GetRankId() {
    static int rankId = -1;
    if (rankId == -1) {
        const char* rankStr = std::getenv("PYPTO_RANK_ID");
        if (rankStr != nullptr) {
            rankId = std::atoi(rankStr);
        } else {
            rankId = 0;
        }
    }
    return rankId;
}

static std::string GenerateShmNameName(int tensorMagic) {
    return "/pypto_shmem_" + std::to_string(tensorMagic);
}

void ExecuteOpShmemSet(ExecuteOperationContext *ctx) {
    (void)ctx;
}

void ExecuteOpShmemPut(ExecuteOperationContext *ctx) {
    (void)ctx;
}

void ExecuteOpShmemPutUb2Gm(ExecuteOperationContext *ctx) {
    (void)ctx;
}

void ExecuteOpShmemSignal(ExecuteOperationContext *ctx) {
    (void)ctx;
}

void ExecuteOpShmemWaitUntil(ExecuteOperationContext *ctx) {
    (void)ctx;
}

void ExecuteOpShmemGet(ExecuteOperationContext *ctx) {
    (void)ctx;
}

void ExecuteOpShmemGetGm2Ub(ExecuteOperationContext *ctx) {
    (void)ctx;
}

REGISTER_CALC_OP(OP_SHMEM_SET, Opcode::OP_SHMEM_SET, ExecuteOpShmemSet);
REGISTER_CALC_OP(OP_SHMEM_PUT, Opcode::OP_SHMEM_PUT, ExecuteOpShmemPut);
REGISTER_CALC_OP(OP_SHMEM_PUT_UB2GM, Opcode::OP_SHMEM_PUT_UB2GM, ExecuteOpShmemPutUb2Gm);
REGISTER_CALC_OP(OP_SHMEM_SIGNAL, Opcode::OP_SHMEM_SIGNAL, ExecuteOpShmemSignal);
REGISTER_CALC_OP(OP_SHMEM_WAIT_UNTIL, Opcode::OP_SHMEM_WAIT_UNTIL, ExecuteOpShmemWaitUntil);
REGISTER_CALC_OP(OP_SHMEM_GET, Opcode::OP_SHMEM_GET, ExecuteOpShmemGet);
REGISTER_CALC_OP(OP_SHMEM_GET_GM2UB, Opcode::OP_SHMEM_GET_GM2UB, ExecuteOpShmemGetGm2Ub);

} // namespace npu::tile_fwk
