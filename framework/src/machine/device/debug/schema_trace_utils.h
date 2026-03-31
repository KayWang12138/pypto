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