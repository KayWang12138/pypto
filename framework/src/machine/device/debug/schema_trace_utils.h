/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "interface/schema/schema.h"
#include "tilefwk/aikernel_data.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/utils/dynamic/dev_encode_types.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"

namespace npu::tile_fwk::dynamic{

#define RAW_TENSOR_ADDR_MASK ((1UL << 63) - 1)

class SchemaDumpUtil {
public:
    static void DumpSchemaOperationInfo(DynDeviceTask *dyntask, DevAscendFunctionDuppedData *duppedData, int coreIdx, uint64_t taskId, uint32_t deviceTaskId);

private:
    static DynFuncData* GetDynFuncData(DynDeviceTask *dyntask, uint64_t taskId);
    static uint64_t GetTensorAddr(DynFuncData *dynFuncData, uint64_t rawTensorIndex);
    static uint64_t GetCoa(DynFuncData *dynFuncData, const SymInt *attrs, int idx);
    static schema::shape SchemaGetShape(DynFuncData *dynFuncData, const SymInt *attrs, const DevAscendOperationOperandInfo &info);
    static schema::offset SchemaGetOffset(DynFuncData *dynFuncData, const SymInt *attrs, const DevAscendOperationOperandInfo &info);

    static inline uint32_t FuncID(uint32_t taskId) {
        return taskId >> TASKID_TASK_BITS;
    }

    static inline uint32_t TaskID(uint32_t taskId) {
        return taskId & TASKID_TASK_MASK;
    }

    static inline int GetRootIndex(DynDeviceTask *dyntask, uint64_t taskId) {
        uint32_t funcId = FuncID(taskId);
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        return func->GetRootIndex();
    }

    static inline int GetLeafIndex(DynDeviceTask *dyntask, uint64_t taskId) {
        uint32_t funcId = FuncID(taskId);
        uint32_t opIndex = TaskID(taskId);
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return callList[opIndex];
    }
};
}