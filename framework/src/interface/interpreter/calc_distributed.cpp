/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS EXPRESS OR IMPLIED,
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

static void GetShmemOffset(const std::vector<int64_t> &shape, 
                           const std::vector<int64_t> &tileShape, 
                           std::vector<int64_t> &offset) {
    offset.clear();
    offset.resize(shape.size(), 0);
    for (size_t i = 0; i < std::min(shape.size(), tileShape.size()); i++) {
        offset[i] = 0;
    }
}

void ExecuteOpShmemSet(ExecuteOperationContext *ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() == 2);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 1);
    
    auto &predToken = ctx->ioperandDataViewList->at(0);
    auto &shmemData = ctx->ioperandDataViewList->at(1);
    auto &output = ctx->ooperandInplaceDataViewList->at(0);
    
    auto attr = std::static_pointer_cast<Distributed::ShmemSetAttr>(
        ctx->op->GetOpAttribute());
    
    ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE,
           shmemData->GetData()->IsSharedMemory());
    
    if (attr->setType == 0) {
        calc::Copy(shmemData, predToken);
    }
    
    calc::Copy(output, shmemData);
}

void ExecuteOpShmemPut(ExecuteOperationContext *ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() == 3);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 1);
    
    auto &predToken = ctx->ioperandDataViewList->at(0);
    auto &inputData = ctx->ioperandDataViewList->at(1);
    auto &shmemData = ctx->ioperandDataViewList->at(2);
    auto &output = ctx->ooperandInplaceDataViewList->at(0);
    
    auto attr = std::static_pointer_cast<Distributed::ShmemPutAttr>(
        ctx->op->GetOpAttribute());
    
    ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE,
           shmemData->GetData()->IsSharedMemory());
    
    std::vector<int64_t> offset;
    GetShmemOffset(shmemData->GetShape(), attr->copyBufferShape, offset);
    
    if (attr->atomicType == Distributed::AtomicType::SET) {
        shmemData->GetData()->shmemData_->Set(
            offset,
            inputData->GetData()->data(),
            inputData->GetData()->size()
        );
    } else if (attr->atomicType == Distributed::AtomicType::ADD) {
        shmemData->GetData()->shmemData_->Add(
            offset,
            inputData->GetData()->data(),
            inputData->GetData()->size()
        );
    }
    
    calc::Copy(output, predToken);
}

void ExecuteOpShmemPutUb2Gm(ExecuteOperationContext *ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() == 3);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 1);
    
    auto &inputData = ctx->ioperandDataViewList->at(0);
    auto &shmemData = ctx->ioperandDataViewList->at(1);
    auto &barrierDummy = ctx->ioperandDataViewList->at(2);
    auto &output = ctx->ooperandInplaceDataViewList->at(0);
    
    ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE,
           shmemData->GetData()->IsSharedMemory());
    
    std::vector<int64_t> offset;
    GetShmemOffset(shmemData->GetShape(), inputData->GetShape(), offset);
    
    shmemData->GetData()->shmemData_->Set(
        offset,
        inputData->GetData()->data(),
        inputData->GetData()->size()
    );
    
    calc::Copy(output, barrierDummy);
}

void ExecuteOpShmemSignal(ExecuteOperationContext *ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() == 2);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 1);
    
    auto &predToken = ctx->ioperandDataViewList->at(0);
    auto &shmemSignal = ctx->ioperandDataViewList->at(1);
    auto &output = ctx->ooperandInplaceDataViewList->at(0);
    
    auto attr = std::static_pointer_cast<Distributed::ShmemSignalAttr>(
        ctx->op->GetOpAttribute());
    
    ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE,
           shmemSignal->GetData()->IsSharedMemory());
    
    std::vector<int64_t> offset;
    std::vector<int64_t> tileShape = {attr->tileRowShape, attr->tileColShape};
    GetShmemOffset(shmemSignal->GetShape(), tileShape, offset);
    
    shmemSignal->GetData()->shmemData_->Signal(offset, attr->signalValue);
    
    calc::Copy(output, predToken);
}

void ExecuteOpShmemWaitUntil(ExecuteOperationContext *ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() == 2);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 1);
    
    auto &predToken = ctx->ioperandDataViewList->at(0);
    auto &shmemSignal = ctx->ioperandDataViewList->at(1);
    auto &output = ctx->ooperandInplaceDataViewList->at(0);
    
    auto attr = std::static_pointer_cast<Distributed::ShmemWaitUntilAttr>(
        ctx->op->GetOpAttribute());
    
    ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE,
           shmemSignal->GetData()->IsSharedMemory());
    
    std::vector<int64_t> offset;
    std::vector<int64_t> tileShape = {attr->tileRowShape, attr->tileColShape};
    GetShmemOffset(shmemSignal->GetShape(), tileShape, offset);
    
    shmemSignal->GetData()->shmemData_->WaitUntil(offset, attr->expectedSum);
    
    calc::Copy(output, predToken);
}

void ExecuteOpShmemGet(ExecuteOperationContext *ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() == 2);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 2);
    
    auto &predToken = ctx->ioperandDataViewList->at(0);
    auto &shmemData = ctx->ioperandDataViewList->at(1);
    auto &output = ctx->ooperandInplaceDataViewList->at(0);
    auto &ubTensor = ctx->ooperandInplaceDataViewList->at(1);
    
    ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE,
           shmemData->GetData()->IsSharedMemory());
    
    std::vector<int64_t> offset;
    GetShmemOffset(shmemData->GetShape(), output->GetShape(), offset);
    
    shmemData->GetData()->shmemData_->Get(
        offset,
        output->GetData()->data(),
        output->GetData()->size()
    );
    
    calc::Copy(ubTensor, predToken);
}

void ExecuteOpShmemGetGm2Ub(ExecuteOperationContext *ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() == 2);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 2);
    
    auto &dummy = ctx->ioperandDataViewList->at(0);
    auto &shmemData = ctx->ioperandDataViewList->at(1);
    auto &output = ctx->ooperandInplaceDataViewList->at(0);
    auto &ubTensor = ctx->ooperandInplaceDataViewList->at(1);
    
    ASSERT(ExecuteOperationScene::INVALID_TENSOR_DTYPE,
           shmemData->GetData()->IsSharedMemory());
    
    std::vector<int64_t> offset;
    GetShmemOffset(shmemData->GetShape(), output->GetShape(), offset);
    
    shmemData->GetData()->shmemData_->Get(
        offset,
        output->GetData()->data(),
        output->GetData()->size()
    );
    
    calc::Copy(ubTensor, dummy);
}

REGISTER_CALC_OP(OP_SHMEM_SET, Opcode::OP_SHMEM_SET, ExecuteOpShmemSet);
REGISTER_CALC_OP(OP_SHMEM_PUT, Opcode::OP_SHMEM_PUT, ExecuteOpShmemPut);
REGISTER_CALC_OP(OP_SHMEM_PUT_UB2GM, Opcode::OP_SHMEM_PUT_UB2GM, ExecuteOpShmemPutUb2Gm);
REGISTER_CALC_OP(OP_SHMEM_SIGNAL, Opcode::OP_SHMEM_SIGNAL, ExecuteOpShmemSignal);
REGISTER_CALC_OP(OP_SHMEM_WAIT_UNTIL, Opcode::OP_SHMEM_WAIT_UNTIL, ExecuteOpShmemWaitUntil);
REGISTER_CALC_OP(OP_SHMEM_GET, Opcode::OP_SHMEM_GET, ExecuteOpShmemGet);
REGISTER_CALC_OP(OP_SHMEM_GET_GM2UB, Opcode::OP_SHMEM_GET_GM2UB, ExecuteOpShmemGetGm2Ub);

} // namespace npu::tile_fwk
