#include "schema_trace_utils.h"
namespace npu::tile_fwk {
namespace dynamic {
DynFuncData* SchemaDumpUtil::GetDynFuncData(DynDeviceTask *dyntask, uint64_t taskId) {
    DynFuncHeader *head = (DynFuncHeader *) dyntask->GetDynFuncDataList();
    auto funcDataList = (DynFuncData *)(head + 1);
    auto funcData = &funcDataList[FuncID(taskId)];
    return funcData;
}

uint64_t SchemaDumpUtil::GetTensorAddr(DynFuncData *dynFuncData, uint64_t rawTensorIndex) {
    auto desc = &dynFuncData->rawTensorDesc[rawTensorIndex];
    if (desc->location == npu::tile_fwk::RAW_TENSOR_LOCATION_LOCAL) {
        return dynFuncData->workspaceAddr + desc->offsetOrIndex;
    } else {
        return dynFuncData->rawTensorAddr[desc->offsetOrIndex] & RAW_TENSOR_ADDR_MASK;
    }
}

uint64_t SchemaDumpUtil::GetCoa(DynFuncData *dynFuncData, const SymInt *attrs, int idx) {
    return attrs[idx].IsExpression() ? dynFuncData->exprTbl[attrs[idx].Value()] : attrs[idx].Value();
}

schema::shape SchemaDumpUtil::SchemaGetShape(DynFuncData *dynFuncData, const SymInt *attrs, const DevAscendOperationOperandInfo &info) {
    auto attrOffset = info.staticOffsetAttrBeginIndex;
    std::vector<schema::Int64Type> shapeList;
    for (int d = 0; d < info.GetDim(); d++) {
        auto shapeIdx = attrOffset + d + info.GetDim() * 3;
        auto actualShape = GetCoa(dynFuncData, attrs, shapeIdx);
        shapeList.push_back(actualShape);
    }
    return schema::shape(schema::shapeList(shapeList));
}

schema::offset SchemaDumpUtil::SchemaGetOffset(DynFuncData *dynFuncData, const SymInt *attrs, const DevAscendOperationOperandInfo &info) {
    auto attrOffset = info.staticOffsetAttrBeginIndex;
    std::vector<schema::Int64Type> offsetList;
    for (int d = 0; d < info.GetDim(); d++) {
        auto offsetIdx = attrOffset + d;
        auto actualOffset = GetCoa(dynFuncData, attrs, offsetIdx);
        offsetList.push_back(actualOffset);
    }
    return schema::offset(schema::offsetList(offsetList));
}

void SchemaDumpUtil::DumpSchemaOperationInfo(DynDeviceTask *dyntask, DevAscendFunctionDuppedData *duppedData, int coreIdx, uint64_t taskId, uint32_t deviceTaskId) {
#if ENABLE_DUMP_OPERATION
    uint32_t funcId = FuncID(taskId);
    int rootIndex = GetRootIndex(dyntask, taskId);
    int leafIndex = GetLeafIndex(dyntask, taskId);
    uint32_t opIdx = TaskID(taskId);
    auto dynFuncData = GetDynFuncData(dyntask, taskId);
    auto attrBase = &duppedData->GetSource()->GetOperationAttr(opIdx, 0);

    DEV_TRACE_DEBUG(LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex),LActStart(coreIdx)));
    DEV_TRACE_DEBUG_SPLIT(LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex),
        duppedData->GetSource()->SchemaGetCoa(opIdx)));

    auto iOperandSize = duppedData->GetSource()->GetOperationIOperandSize(opIdx);
    DEV_TRACE_DEBUG(LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex),LActIncastCount(iOperandSize)));
    for (size_t i = 0; i < iOperandSize; i++) {
        auto iOperand = duppedData->GetSource()->GetOperationIOperand(opIdx, i);
        auto base = GetTensorAddr(dynFuncData, iOperand->rawIndex);
        auto size = duppedData->GetRawTensorDataSize(iOperand->rawIndex);
        auto opInfo = duppedData->GetSource()->GetOperationIOperandInfo(opIdx, i);
        DEV_TRACE_DEBUG(LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex),
            LActIncast(SchemaGetShape(dynFuncData, attrBase, opInfo), SchemaGetOffset(dynFuncData, attrBase, opInfo), Range(base, base + size))));
    }

    auto oOperandSize = duppedData->GetSource()->GetOperationOOperandSize(opIdx);
    DEV_TRACE_DEBUG(LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex), LActOutcastCount(oOperandSize)));
    for (size_t i = 0; i < oOperandSize; i++) {
        auto oOperand = duppedData->GetSource()->GetOperationOOperand(opIdx, i);
        auto base = GetTensorAddr(dynFuncData, oOperand->rawIndex);
        auto size = duppedData->GetRawTensorDataSize(oOperand->rawIndex);
        auto opInfo = duppedData->GetSource()->GetOperationOOperandInfo(opIdx, i);
        DEV_TRACE_DEBUG(LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex),
            LActOutcast(SchemaGetShape(dynFuncData, attrBase, opInfo), SchemaGetOffset(dynFuncData, attrBase, opInfo), Range(base, base + size))));
    }
#endif
}
}
}