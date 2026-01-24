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
 * \file dev_encode_function_dupped_data.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#include "machine/utils/dynamic/allocator/allocators.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk::dynamic {
const uint32_t RAW_TENSOR_OFFSET_SIZE = 63;

struct DevAscendFunctionDuppedData {
    DevAscendFunction *source_;
    DevAscendFunctionDuppedOperation operationList_;
    DevAscendFunctionDuppedVector incastList_;
    DevAscendFunctionDuppedVector outcastList_;
    DevAscendFunctionDuppedVector expressionList_;
    uintdevptr_t runtimeWorkspace_;
    RuntimeReuseInfo runtimeWsReuseInfo_;
    uintdevptr_t runtimeOutcastWorkspace_;
    uint8_t data_[0];
    /*
     *  Duplicated:
     *      predcount_t                                         predCountListData[];
     *  Allocated (& zero-ed):
     *      AddressDescriptor                                   incastAddressListData[];
     *      AddressDescriptor                                   outcastAddressListData[];
     *      uint64_t                                            expressionListData[];
     *      DevAscendFunctionDuppedStitchList                   stitchListData[];
     */
#define GET_DATA(type, data, base, index) ((reinterpret_cast<type *>(const_cast<uint8_t *>((data) + (base)))[index]))
    uint32_t GetOperationSize() const { return operationList_.size; }
    const predcount_t &GetOperationCurrPredCount(int index) const { return GET_DATA(predcount_t, data_, operationList_.predCountBase, index); }
    predcount_t &GetOperationCurrPredCount(int index) { return GET_DATA(predcount_t, data_, operationList_.predCountBase, index); }

    uint32_t GetStitchSize() const { return operationList_.stitchCount; }
    const DevAscendFunctionDuppedStitchList &GetStitch(int index) const { return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, index); }
    DevAscendFunctionDuppedStitchList &GetStitch(int index) { return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, index); }

    uint64_t GetExpressionSize() const { return expressionList_.size; }
    const uint64_t &GetExpression(int index) const { return GET_DATA(uint64_t, data_, expressionList_.base, index); }
    uint64_t &GetExpression(int index) { return GET_DATA(uint64_t, data_, expressionList_.base, index); }

    uint64_t *GetExpressionAddr() const {
        return &GET_DATA(uint64_t, data_, expressionList_.base, 0);
    }

    uint64_t GetIncastSize() const { return incastList_.size; }
    AddressDescriptor GetIncastAddress(int index) const { return GET_DATA(AddressDescriptor, data_, incastList_.base, index); }
    AddressDescriptor &GetIncastAddress(int index) { return GET_DATA(AddressDescriptor, data_, incastList_.base, index); }

    uint64_t GetOutcastSize() const { return outcastList_.size; }
    AddressDescriptor GetOutcastAddress(int index) const { return GET_DATA(AddressDescriptor, data_, outcastList_.base, index); }
    AddressDescriptor &GetOutcastAddress(int index) { return GET_DATA(AddressDescriptor, data_, outcastList_.base, index); }

    RuntimeReuseInfo GetRuntimeReuseInfo() const { return runtimeWsReuseInfo_; }
    RuntimeReuseInfo &GetRuntimeReuseInfo() { return runtimeWsReuseInfo_; }

    uintdevptr_t GetRuntimeWorkspace() const { return runtimeWorkspace_; }
    uintdevptr_t &GetRuntimeWorkspace() { return runtimeWorkspace_; }

    uintdevptr_t GetRuntimeOutcastWorkspace() const { return runtimeOutcastWorkspace_; }
    uintdevptr_t &GetRuntimeOutcastWorkspace() { return runtimeOutcastWorkspace_; }

    DevAscendFunction *GetSource() const { return source_; }
    DevAscendFunction *&GetSource() { return source_; }

    const DevAscendFunctionDuppedStitchList &GetOperationStitch(int operationIndex, bool maybeNull = true) const {
        int outcastStitchIndex = GetSource()->GetOperationOutcastStitchIndex(operationIndex);
        DEV_IF_NONDEVICE {
            if (!maybeNull && outcastStitchIndex == 0) {
                DEV_ERROR("GetOperationStitch: operation %d has invalid outcast stitch index 0", operationIndex);
            }
            DEV_ASSERT(maybeNull || outcastStitchIndex != 0);
        }
        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, outcastStitchIndex);
    }
    DevAscendFunctionDuppedStitchList &GetOperationStitch(int operationIndex, bool maybeNull = true) {
        int outcastStitchIndex = GetSource()->GetOperationOutcastStitchIndex(operationIndex);
        DEV_IF_NONDEVICE {
            if (!maybeNull && outcastStitchIndex == 0) {
                DEV_ERROR("GetOperationStitch: operation %d has invalid outcast stitch index 0", operationIndex);
            }
            DEV_ASSERT(maybeNull || outcastStitchIndex != 0);
        }
        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, outcastStitchIndex);
    }

    inline uint64_t GetIncastDataSize(int incastIndex) const {
        auto rawTensor = GetSource()->GetIncastRawTensor(incastIndex);
        auto size = rawTensor->GetMemoryRequirement(GetExpressionAddr());
        return size;
    }

    inline uint64_t GetOutcastDataSize(int outcastIndex) const {
        auto rawTensor = GetSource()->GetOutcastRawTensor(outcastIndex);
        auto size = rawTensor->GetMemoryRequirement(GetExpressionAddr());
        return size;
    }

    schema::range SchemaGetIncastRange(int arg) const {
        auto base = GetIncastAddress(arg).GetAddress();
        auto size = GetIncastDataSize(arg);
        return schema::Range(base, base + size);
    }
    schema::range SchemaGetOutcastRange(int arg) const {
        auto base = GetOutcastAddress(arg).GetAddress();
        auto size = GetOutcastDataSize(arg);
        return schema::Range(base, base + size);
    }

    schema::RActWorkspace SchemaGetWorkspace() const {
        auto workspaceBegin = GetRuntimeWorkspace();
        auto workspaceEnd = GetRuntimeWorkspace() + GetSource()->rootInnerTensorWsMemoryRequirement;
        return schema::RActWorkspace(schema::Range(workspaceBegin, workspaceEnd));
    }

    std::string Dump(int indent = 0) const;
};

struct DevAscendFunctionDupped {
    DevAscendFunctionDupped() = default;
    explicit DevAscendFunctionDupped(WsAllocation tinyAlloc) : dupTiny_(tinyAlloc) {}

    static DevAscendFunctionDupped DuplicateRoot(DevAscendFunction *func, WsAllocation tinyAlloc) {
        DevAscendFunctionDuppedData *dupData = tinyAlloc.As<DevAscendFunctionDuppedData>();
        DevAscendFunctionDuppedData *sourceData = func->GetDuppedData();
        (void)memcpy_s(reinterpret_cast<uint8_t *>(dupData),
            func->GetDuppedDataCopySize(),
            sourceData,
            func->GetDuppedDataCopySize());
        (void)memset_s(reinterpret_cast<uint8_t *>(dupData) + func->GetDuppedDataCopySize(),
            func->GetDuppedDataAllocSize() - func->GetDuppedDataCopySize(),
            0,
            func->GetDuppedDataAllocSize() - func->GetDuppedDataCopySize());
        dupData->GetSource() = func;

        DevAscendFunctionDupped dup(tinyAlloc);
        return dup;
    }

    void ReleaseDuppedMemory(WsMetadataAllocator &allocator) {
        (void)allocator;
    }

    RuntimeReuseInfo GetRuntimeReuseInfo() const { return DupData()->GetRuntimeReuseInfo(); }
    RuntimeReuseInfo &GetRuntimeReuseInfo() { return DupData()->GetRuntimeReuseInfo(); }

    uintdevptr_t RuntimeWorkspace() const { return DupData()->GetRuntimeWorkspace(); }
    uintdevptr_t &RuntimeWorkspace() { return DupData()->GetRuntimeWorkspace(); }

    uintdevptr_t RuntimeOutcastBase() const { return DupData()->GetRuntimeOutcastWorkspace(); }
    uintdevptr_t &RuntimeOutcastBase() { return DupData()->GetRuntimeOutcastWorkspace(); }

    const DevAscendFunction *GetSource() const { return DupData()->GetSource(); }
    DevAscendFunction *GetSource() { return DupData()->GetSource(); }

    inline const uint64_t &GetExpression(int arg) const { return DupData()->GetExpression(arg); };
    inline uint64_t &GetExpression(int arg) { return DupData()->GetExpression(arg); };
    inline uint64_t GetExpressionSize() const { return DupData()->GetExpressionSize(); }
    inline uint64_t *GetExpressionAddr() const { return DupData()->GetExpressionAddr(); }

    inline auto GetOperationSize() const { return DupData()->GetOperationSize(); }
    inline const predcount_t &GetOperationCurrPredCount(int arg) const { return DupData()->GetOperationCurrPredCount(arg); };
    inline predcount_t &GetOperationCurrPredCount(int arg) { return DupData()->GetOperationCurrPredCount(arg); };
    inline const auto &GetOperationStitch(int arg, bool maybeNull = true) const { return DupData()->GetOperationStitch(arg, maybeNull); };
    inline auto &GetOperationStitch(int arg, bool maybeNull = true) { return DupData()->GetOperationStitch(arg, maybeNull); };

    inline AddressDescriptor GetIncastAddress(int arg) const { return DupData()->GetIncastAddress(arg); };
    inline AddressDescriptor &GetIncastAddress(int arg) { return DupData()->GetIncastAddress(arg); };

    inline AddressDescriptor GetOutcastAddress(int arg) const { return DupData()->GetOutcastAddress(arg); };
    inline AddressDescriptor &GetOutcastAddress(int arg) { return DupData()->GetOutcastAddress(arg); };

    schema::expr SchemaGetExpressionTable() const {
        std::vector<schema::Int64Type> exprTable;
        uint64_t *exprAddr = GetExpressionAddr();
        uint64_t exprSize = GetExpressionSize();
        for (uint64_t i = 0; i < exprSize; i++) {
            exprTable.push_back(exprAddr[i]);
        }
        return schema::expr(exprTable);
    }

    inline uintdevptr_t GetRawTensorAddr(int rawIndex) const {
        uintdevptr_t addr = 0ULL;
        const DevAscendRawTensor *rawTensor = GetSource()->GetRawTensor(rawIndex);
        if (rawTensor->ioProperty == DevIOProperty::ROOT_INCAST) {
            AddressDescriptor incast = GetIncastAddress(rawTensor->ioIndex);
            DEV_ASSERT_MSG(!incast.IsNullAddress(),
                "Null incast: root [%s], rawIndex [%d], ioIndex [%d]",
                GetSource()->GetRawName(), rawIndex, rawTensor->ioIndex);
            addr = incast.addr;
        } else if (rawTensor->ioProperty == DevIOProperty::ROOT_OUTCAST) {
            AddressDescriptor outcast = GetOutcastAddress(rawTensor->ioIndex);
            DEV_ASSERT_MSG(!outcast.IsNullAddress(),
                "Null outcast: root [%s], rawIndex [%d], ioIndex [%d]",
                GetSource()->GetRawName(), rawIndex, rawTensor->ioIndex);
            addr = outcast.addr;
        } else {
            uintdevptr_t runtimeWorkspace = RuntimeWorkspace();
            DEV_ASSERT_MSG(runtimeWorkspace != 0,
                "Trying to access inner tensor addr with zero runtime workspace: root [%s], rawIndex [%d]",
                GetSource()->GetRawName(), rawIndex);
            addr = runtimeWorkspace + rawTensor->addrOffset;
        }
        return addr;
    }

    // for stitch
    inline void GetInTensorOffset(int32v8 &offset, int operationIndex, int operandIndex) const {
        auto func = GetSource();
        auto &operandInfo = func->GetOperationIOperandInfo(operationIndex, operandIndex);

        const SymInt *offsetSymList = &func->GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        offset[0] = offsetSymList[0].IsExpression() ? GetExpression(offsetSymList[0].Value()) : offsetSymList[0].Value();
        offset[1] = offsetSymList[1].IsExpression() ? GetExpression(offsetSymList[1].Value()) : offsetSymList[1].Value();
    }

    inline void GetOutTensorOffset(int32v8 &offset, int operationIndex, int operandIndex) const {
        auto func = GetSource();
        auto &operandInfo = func->GetOperationOOperandInfo(operationIndex, operandIndex);

        const SymInt *offsetSymList = &func->GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        offset[0] = offsetSymList[0].IsExpression() ? GetExpression(offsetSymList[0].Value()) : offsetSymList[0].Value();
        offset[1] = offsetSymList[1].IsExpression() ? GetExpression(offsetSymList[1].Value()) : offsetSymList[1].Value();
    }

    inline void GetFuncTensorOffsetAndShape(uint64_t offset[DEV_SHAPE_DIM_MAX], uint64_t shape[DEV_SHAPE_DIM_MAX], int dims,
        int operationIndex, int operandIndex, bool isIOperand = true) const {
        auto func = GetSource();
        GetTensorOffsetAndShape<false>(func, offset, shape, &GetExpression(0), dims, operationIndex, operandIndex, isIOperand);
    }

    std::string Dump(int indent = 0) const {
        return DupData()->Dump(indent);
    }

    inline int64_t GetValue(const SymInt *attrs, int idx) const {
        return attrs[idx].IsExpression() ? funcData->exprTbl[attrs[idx].Value()] : attrs[idx].Value();
    }

    inline uint64_t GetRawTensorAddrEx(int idx) const {
        auto desc = funcData->rawTensorDesc[idx];
        if (desc.location == RAW_TENSOR_LOCATION_LOCAL)
            return funcData->workspaceAddr + desc.offsetOrIndex;
        else
            return funcData->rawTensorAddr[desc.offsetOrIndex] & ((1UL << RAW_TENSOR_OFFSET_SIZE) - 1);
    }

    std::string DumpDyn(int funcIdx, int operIdx, const DevCceBinary *cceBinary) const;

    void DumpTopo(std::ofstream &os, int seqNo, int funcIdx, const DevCceBinary *cceBinary) const;

#if DEBUG_INFINITE_LIFETIME
    void DumpTensorAddrInfo(std::vector<std::string> &infos, uint32_t seqNo, uint32_t funcIdx);
#endif // DEBUG_INFINITE_LIFETIME

    std::vector<std::string> DumpLeafs(uint32_t seqNo, uint32_t funcIdx) const;

    std::string DumpDyn(int funcIdx, const DevCceBinary *cceBinary) const;

    bool IsNull() const { return !dupTiny_; }
    void ResetNull() { dupTiny_.Invalidate(); }
    DynFuncData *GetFuncData() { return funcData; }
    void SetFuncData(DynFuncData *data) { funcData = data; }

    DevAscendFunctionDuppedData *DupDataForDynFuncData() { return DupData(); }

private:
    void DumpRawShape(const DevAscendRawTensor *rawTensor, uint32_t dimSize, std::vector<std::string> &lines,
                      std::stringstream &oss) const;

    void DumpOperandShape(uint32_t dimSize, size_t opIdx, size_t operandIdx, bool isIn, std::vector<std::string> &lines,
                          std::stringstream &oss) const;

    void DumpAttr(const DevAscendFunction *func, const SymInt *attrs, const DevAscendOperationOperandInfo &info,
                  std::stringstream &oss) const;

    void DumpFuncData(const DevAscendFunction *func, int funcIdx, const DevCceBinary *cceBinary,
                      std::stringstream &oss) const;

    const DevAscendFunctionDuppedData *DupData() const { return dupTiny_.As<DevAscendFunctionDuppedData>(); }
    DevAscendFunctionDuppedData *DupData() { return dupTiny_.As<DevAscendFunctionDuppedData>(); }

private:
    DynFuncData *funcData{nullptr}; // used by aicore
    WsAllocation dupTiny_;
};
}