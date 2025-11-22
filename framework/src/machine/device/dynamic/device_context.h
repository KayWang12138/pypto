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
 * \file device_context.h
 * \brief
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <vector>
#include <cstring>
#include <utility>
#include "machine/utils/device_log.h"
#include "interface/utils/common.h"
#ifndef __DEVICE__
#include "interface/configs/config_manager.h"
#endif

#ifndef CONFIG_BAREMETAL
#include <sys/mman.h>
#endif

#include "tilefwk/core_func_data.h"
#include "interface/schema/schema.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/allocator/allocators.h"
#include "machine/utils/dynamic/vector.h"
#include "machine/utils/dynamic/item_pool.h"
#include "device_utils.h"
#include "securec.h"
#include "costmodel_utils.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/dynamic/spsc_queue.h"
#ifndef __DEVICE__
#include <map>
#endif

#ifndef STR
#define STR_(n)         #n
#define STR(n)          STR_(n)
#endif

#define AOT_CODE_POOL_CODE_SIZE     (4096 * 0x200)
extern uint8_t aotCodePoolCode[];

namespace npu::tile_fwk {

namespace dynamic {
struct DeviceExecuteContext;

#define POOL_PAGE_COUNT             0x100
#define ENABLE_STITCH 1

asm(
    "\n\t.pushsection .bss." STR(aotCodePoolCode) ",\"axwG\",@nobits," STR(aotCodePoolCode) ",comdat"
    "\n\t.p2align 12"
    "\n\t.weak " STR(aotCodePoolCode)
    "\n\t.type " STR(aotCodePoolCode) ", @gnu_unique_object"
    "\n\t.size " STR(aotCodePoolCode) ", " STR(AOT_CODE_POOL_CODE_SIZE)
    "\n" STR(aotCodePoolCode) ":"
    "\n\t.zero " STR(AOT_CODE_POOL_CODE_SIZE)
    "\n\t.popsection"
);

struct AOTCodePool {
    uintptr_t base{0};
    uintptr_t offset{0};

    void MapExec() {}

    static AOTCodePool &GetCodePool() {
        static AOTCodePool pool = {0};
        if (pool.base == 0) {
            pool.base = (uintptr_t)aotCodePoolCode;
        }
        return pool;
    };
};
struct AOTBinary {
    AOTBinary() {}

    void InitCodeSize(const void *data, uint64_t size) {
        auto &pool = AOTCodePool::GetCodePool();
        PerfBegin(PERF_EVT_CONTROL_FLOW_MAPEXE_MEMCPY);
        memcpy_s((void *)pool.base , size, data, size);
        __builtin___clear_cache((void *)pool.base, (uint8_t*)pool.base + size);
        PerfEnd(PERF_EVT_CONTROL_FLOW_MAPEXE_MEMCPY);
        code_ = (unsigned char *)pool.base;
        size_ = size;
    }
    void InitCode(const void *data) {
        code_ = (const unsigned char *)data;
    }

    const unsigned char *code_{nullptr};
    size_t size_{0};
};

struct AOTBinaryControlFlow : AOTBinary {
    typedef void (*controlFlowEntry)(
            struct DeviceExecuteContext *ctx, uint64_t *symbolTable, CallRootEntryType callRootList[T_CALLROOT_MAX], DevStartArgsBase *startArgsBase);

    AOTBinaryControlFlow() = default;

    AOTBinaryControlFlow(const std::tuple<const void *, uint64_t> &code, controlFlowEntry entry = nullptr)
        : AOTBinaryControlFlow(std::get<0>(code), std::get<1>(code), entry) {}

    AOTBinaryControlFlow(const std::vector<uint8_t> &code, controlFlowEntry entry = nullptr)
        : AOTBinaryControlFlow(code.data(), code.size(), entry) {}

    AOTBinaryControlFlow(const void *code, uint64_t codeSize, controlFlowEntry entry = nullptr) {
        if (entry != nullptr) {
            InitCode((void *)entry);
        } else {
            InitCodeSize(code, codeSize);
        }
    }

    void CallControlFlow(
            struct DeviceExecuteContext *ctx, uint64_t *symbolTable, CallRootEntryType callRootList[T_CALLROOT_MAX], DevStartArgsBase *startArgsBase) {
        ((controlFlowEntry)code_)(ctx, symbolTable, callRootList, startArgsBase);
    }
};
const size_t TUBLE_INDEX_2 = 2;
const size_t TUBLE_INDEX_3 = 3;
struct AOTBinaryExpressionTable : AOTBinary {
    using exprEntry = uint64_t (*)(struct DeviceExecuteContext *ctx, uint64_t *symbolTable);
    AOTBinaryExpressionTable() {}
    AOTBinaryExpressionTable(const std::tuple<const void *, uint64_t, const uint64_t *, uint64_t> &table)
        : offsetList(std::get<TUBLE_INDEX_2>(table)), offsetSize(std::get<TUBLE_INDEX_3>(table)) {
        InitCodeSize(std::get<0>(table), std::get<1>(table));
    }

    uint64_t CallExpr(struct DeviceExecuteContext *ctx, uint64_t *symbolTable, uint64_t index) {
        return ((exprEntry)(code_ + offsetList[index]))(ctx, symbolTable);
    }

    const uint64_t *offsetList{nullptr};
    uint64_t offsetSize{0};
};

struct DeviceExecuteProgram {
    DevAscendProgram *prog{nullptr};

    AOTBinaryControlFlow controlFlowBinary;
    AOTBinaryExpressionTable exprBinary;

    DeviceExecuteProgram() {}
    DeviceExecuteProgram(DevAscendProgram *prog_, AOTBinaryControlFlow::controlFlowEntry entry = nullptr)
        : prog(prog_),
          controlFlowBinary(IsDeviceMode() ? prog_->GetDevControlFlowBinary() : prog_->GetHostControlFlowBinary(), entry),
          exprBinary(prog_->GetExpressionTableBinary()) {}

    const void *GetControlFlowEntry() {
        return controlFlowBinary.code_;
    }
};

using StitchedList = Vector<DevAscendFunctionDupped, WsMemCategory::VECTOR_STITCHED_LIST, DeviceWorkspaceAllocator>;

struct DeviceSlotContext {
    void InitAllocator(DeviceWorkspaceAllocator &workspace, uint64_t slotSize) {
        workspace.SetupVector(slotList_);
        workspace.SetupItemPool(slotRefCntPool_, slotSize);
        workspace_ = &workspace;
        slotList_.resize(slotSize);
    }

    void FillInputOutputSlot(DevAscendProgram *devProg, DevStartArgs *args) {
        FillInputOutputSlot(slotList_.data(), slotList_.size(), devProg, args);
    }

    void UpdateSlots(DevAscendFunctionDupped &devRootDup, const StitchedList &stitchedList, uint32_t devTaskId, uint32_t devNextIdx) {
        UpdateSlots(workspace_, slotList_.data(), stitchedList, slotList_.size(), slotRefCntPool_, devRootDup, devTaskId, devNextIdx);
    }

    DeviceExecuteSlot *GetSlotList() { return slotList_.data(); }
    size_t GetSlotSize() { return slotList_.size(); }

    auto &GetSlotRefCntPool() { return slotRefCntPool_; }

    void ClearDirty() {
        for (size_t i = 0; i < slotList_.size(); i++) {
            slotList_[i].stitchDupIdx = INVALID_STITCH_IDX;
        }
    }

public:
    void FillInputOutputSlot(DeviceExecuteSlot *slotList, size_t slotSize, DevAscendProgram *devProg, DevStartArgs *args) {
        DEV_TRACE_DEBUG(CtrlEvent(none(), InputTensorCount(args->GetInputTensorSize())));
        for (int i = 0; i < args->GetInputTensorSize(); ++i) {
            DevTensorData &param = args->GetInputTensor(i);
            int slotIndex = devProg->startArgsInputTensorSlotIndexList[i];
            slotList[slotIndex].desc = AddressDescriptor(param.address);
            DEV_INFO("Param %d Input Slot %d = %lx.", i, slotIndex, param.address);
            DEV_TRACE_DEBUG(CtrlEvent(none(), InputTensorElement(i, param.address, param.shape.GetSize())));
        }
        DEV_TRACE_DEBUG(CtrlEvent(none(), OutputTensorCount(args->GetOutputTensorSize())));
        for (int i = 0; i < args->GetOutputTensorSize(); ++i) {
            DevTensorData &param = args->GetOutputTensor(i);
            int slotIndex = devProg->startArgsOutputTensorSlotIndexList[i];
            slotList[slotIndex].desc = AddressDescriptor(param.address);
            slotList[slotIndex].isOutputSlot = true;
            DEV_INFO("Param %d Output Slot %d = %lx.", i, slotIndex, param.address);
            DEV_TRACE_DEBUG(CtrlEvent(none(), OutputTensorElement(i, param.address, param.shape.GetSize())));
        }
        for (size_t i = static_cast<size_t>(args->GetOutputTensorSize()); i < devProg->startArgsOutputTensorSlotIndexList.size(); ++i) {
            int outSlot = devProg->startArgsOutputTensorSlotIndexList[i];
            int inSlot = devProg->outputInplaceSlotList[i];
            if (inSlot != -1) {
                slotList[outSlot].desc = slotList[inSlot].desc;
                slotList[outSlot].isOutputSlot = true;
                DEV_VERBOSE_DEBUG("Param %zu Output Slot %d = inSlot %d.", i, outSlot, inSlot);
            }
        }
        for (size_t i = 0; i < devProg->assembleSlotIndexList.size(); ++i) {
            int slotIndex = devProg->assembleSlotIndexList[i];
            slotList[slotIndex].isAssembleSlot = true;
            DEV_VERBOSE_DEBUG("Assemble Slot %d.", slotIndex);
        }
        for (size_t i = 0, ie = devProg->partialUpdateList.size(); i < ie; i++) {
            auto &partialUpdate = devProg->At(devProg->partialUpdateList, i);
            int slotIndex = i;
            if (!partialUpdate.Empty()) {
                slotList[slotIndex].isPartialUpdateStitch = true;
                slotList[slotIndex].partialUpdate = &partialUpdate;
                DEV_VERBOSE_DEBUG("Partial Update Slot %d\n", slotIndex);
            }
        }
        (void)slotSize;
    }

    static void UpdateSlotsForStitch(int slotIdx, DeviceExecuteSlot &slot, DevAscendFunction *devRootSrc, DevAscendFunctionOutcast &outcast,
                                     uint32_t devTaskId, uint32_t devNextIdx, uint32_t outcastIndex, uint64_t *expressionList) {
        slot.stitchDupIdx = devNextIdx;
        slot.stitchOutcastIdx = outcastIndex;
        UNUSED(slotIdx);

        auto producerList = &devRootSrc->At(outcast.producerList, 0);
        if (slot.isPartialUpdateStitch) {
            auto &cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
            auto tableData = &slot.partialUpdate->cellMatchRuntimePartialUpdateTable[0];
            auto producerSize = outcast.producerList.size();
            if (producerSize != 0) {
                devRootSrc->CellMatchFillIncastOutcast<false>(
                        producerList, producerSize, expressionList, false, cellMatchTableDesc, tableData, devTaskId, devNextIdx);
            } else {
                // maybe is fullcover producer, dassemble full shape
                devRootSrc->CellMatchFillIncastOutcast<false>(
                        &devRootSrc->At(outcast.stitchPolicyFullCoverProducerList, 0), outcast.stitchPolicyFullCoverProducerList.size(),
                        expressionList, false, cellMatchTableDesc, tableData, devTaskId, devNextIdx);
            }

            DEV_VERBOSE_DEBUG("[UpdateSlots]  slot %d CellMatchPartial=%s\n", slotIdx,
                DevAscendFunctionDuppedStitchList::DumpTask<uint64_t>(tableData, slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size()).c_str());
            slot.isPartialUpdateDirty = true;
        } else {
            auto &cellMatchTableDesc = outcast.cellMatchTableDesc;
            auto tableData = &devRootSrc->At(outcast.cellMatchRuntimeFullUpdateTable, 0);
            devRootSrc->CellMatchFillIncastOutcast<false>(
                    producerList, outcast.producerList.size(), expressionList, false, cellMatchTableDesc, tableData);
            DEV_VERBOSE_DEBUG("[UpdateSlots] slot %d  CellMatchFull=%s\n", slotIdx,
                DevAscendFunctionDuppedStitchList::DumpTask(tableData, outcast.cellMatchRuntimeFullUpdateTable.size()).c_str());
        }
    }

    template <WsMemCategory category>
    static void UpdateSlots(DeviceWorkspaceAllocator *workspace, DeviceExecuteSlot *slotList, const StitchedList &stitchedList, int slotSize,
        ItemPool<uint32_t, category> &slotRefCntPool, DevAscendFunctionDupped &devRootDup, uint32_t devTaskId, uint32_t devNextIdx) {
        UNUSED(slotSize);

        AutoScopedPerf asp(PERF_EVT_UPDATE_SLOT);
        DevAscendFunction *devRootSrc = devRootDup.GetSource();
        uint64_t *expressionList = &devRootDup.GetExpression(0);
        size_t outcastSize = devRootSrc->GetOutcastSize();

        std::vector<int64_t> newRefCntIndex;
        newRefCntIndex.resize(outcastSize, itemPoolInvalidIndex);

        // Increase refCnt for linked incasts
        for (size_t i = 0; i < outcastSize; ++i) {
            auto &outcast = devRootSrc->GetOutcast(i);
            auto *rawTensor = devRootSrc->GetOutcastRawTensor(i);
            if (rawTensor->linkedIncastId != -1) {
                auto &incast = devRootSrc->GetIncast(rawTensor->linkedIncastId);
                DEV_DEBUG_ASSERT(incast.fromSlotList.size() > 0);
                int slotIndex = devRootSrc->At(incast.fromSlotList, 0);
                auto &slot = slotList[slotIndex];
                int64_t refCntIndex = slot.refCntIndex;
                if (refCntIndex != itemPoolInvalidIndex) {
                    slot.RefCntInc(slotRefCntPool, outcast.toSlotList.size());
                }
                newRefCntIndex[i] = refCntIndex;
            }
        }

        // Update slot address
        for (size_t i = 0; i < outcastSize; ++i) {
            auto &srcDesc = devRootDup.GetOutcastAddress(i);
            auto &outcast = devRootSrc->GetOutcast(i);
            for (size_t j = 0; j < outcast.toSlotList.size(); ++j) {
                int slotIdx = devRootSrc->At(outcast.toSlotList, j);
                auto &slot = slotList[slotIdx];
                UpdateSlotsForStitch(slotIdx, slot, devRootSrc, outcast, devTaskId, devNextIdx, i, expressionList);
                if (!slot.RefCntIsNull() && slot.RefCntDec(slotRefCntPool)) {
                    DEV_DEBUG_ASSERT(!slot.desc.IsNullAddress());
                    auto freeAddr = slot.desc.addr;
                    if (!slot.desc.IsAddress()) {
                        freeAddr = stitchedList[slot.desc.dupIdx].GetOutcastAddress(slot.desc.outcastIdx).GetAddress();
                    }
                    workspace->DelayedRecycleSlotMem(freeAddr);
                }

                if (!srcDesc.IsAddress() /* Unroll secondary placeholder */) {
                    slot.desc = srcDesc;
                } else {
                    slot.desc = AddressDescriptor(devNextIdx, i);
                }
                slot.refCntIndex = newRefCntIndex[i];
                DEV_VERBOSE_DEBUG("[UpdateSlots]   Outcast [%3zu] to slot [%3d], address %s.", i, slotIdx, slot.desc.Dump().c_str());
            }
        }
    }
private:
    Vector<DeviceExecuteSlot, WsMemCategory::VECTOR_SLOT_LIST> slotList_;
    ItemPool<uint32_t, WsMemCategory::ITEMPOOL_SLOT_REF_CNT> slotRefCntPool_;
    DeviceWorkspaceAllocator *workspace_{nullptr};
};

struct DeviceStitchContext {
    struct StitchReuseContext {
        // changing with stitching progress
        uint32_t firstDupIdx{0};
        int32_t lastNonEmptyDupIdx{-1};
    } stitchReuseContext_;

    void Init(DevAscendProgram *devProg, DeviceWorkspaceAllocator &workspace) {
        workspace.SetupVector(stitchedList_);
        workspace_ = &workspace;

        workspace_->SetupVector(slotInfosInDecidingSlotMem_);
        slotInfosInDecidingSlotMem_.resize(devProg->slotSize); // need pre alloc , left memory for slab allocator

        Reset();
    }
    void DumpStitchInfo() {
        DumpStitchInfo(stitchedList_.data(), stitchedList_.size());
    }
    size_t Size() const { return stitchedList_.size(); }
    bool Empty() const { return stitchedList_.empty(); }
    void Reset() {
        stitchedList_.clear();
        stitchReuseContext_.firstDupIdx = 0;
        stitchReuseContext_.lastNonEmptyDupIdx = -1;
    }
    void Append(DevAscendFunctionDupped &devRootDup) { stitchedList_.push_back(devRootDup); }

    const auto &GetStitchedList() const { return stitchedList_; }

    static void CheckStitch(DevAscendFunctionDupped *stitchedList, int size, DevAscendFunctionDupped *nextDup) {
        DEV_IF_NONDEVICE {
            uint32_t dynPredCount = 0;
            uint32_t dynSuccCount = 0;
            for (int k = 0; k <= size; k++) {
                DevAscendFunctionDupped *dup = nullptr;
                if (k < size) {
                    dup = &stitchedList[k];
                } else if (nextDup != nullptr) {
                    dup = nextDup;
                } else {
                    break;
                }
                auto src = dup->GetSource();
                for (size_t i = 0; i < dup->GetOperationSize(); i++) {
                    auto opPredCount = src->GetOperationDepGraphPredCount(i);
                    auto opDynPredCount = dup->GetOperationCurrPredCount(i);
                    dynPredCount += opDynPredCount - opPredCount;
                    auto succStitchList = dup->GetOperationStitch(i);
                    for (auto p = succStitchList.Head(); p != nullptr; p = p->Next()) {
                        dynSuccCount += p->Size();
                    }
                }
            }
            DEV_ASSERT(dynPredCount == dynSuccCount);
        }
    }

    static void CheckStitch(DynDeviceTask *dyntask) {
        DevAscendFunctionDupped *stitchedList = &dyntask->stitchedList[0];
        int stitchedSize = dyntask->stitchedList.size();
        CheckStitch(stitchedList, stitchedSize, nullptr);
    }

    uint64_t Stitch(DeviceSlotContext &slotContext, DevAscendFunctionDupped &nextDup, size_t devTaskId, size_t devNextIdx) {
        uint64_t count = FastStitch(slotContext.GetSlotList(), slotContext.GetSlotSize(), nextDup, devTaskId, devNextIdx);
        if (stitchedList_.capacity() == 0) {
            /* This stitchedList_ vector can only allocate sufficient space once,
               during a single device task construction process.*/
            stitchedList_.reserve(MAX_CACHED_FUNC_NUM);
        }
        Append(nextDup);
        stitchedCallOpSize_ += nextDup.GetSource()->GetOperationSize();
        return count;
    }

    void RecycleTensorWorkspace() {
        // recycle submitted tasks' workspace memory
        workspace_->RecycleDevFuncWorkspace();
        workspace_->TriggerDelayedRecycle();
    }

    void DumpSlotInfo(const char *label, DeviceExecuteSlot *slotList, size_t slotSize) {
        UNUSED(label);
        UNUSED(slotList);
        UNUSED(slotSize);
        DEV_IF_VERBOSE_DEBUG {
            DEV_VERBOSE_DEBUG("[DecideSlotAddress] %s.", label);
            for (size_t slotIdx = 0; slotIdx < slotSize; slotIdx++) {
                auto &desc = slotList[slotIdx].desc;
                UNUSED(desc);
                const char *extraAttr = "";
                UNUSED(extraAttr);
                if (slotList[slotIdx].isOutputSlot) {
                    extraAttr = " <output>";
                } else if (slotList[slotIdx].isAssembleSlot) {
                    extraAttr = " <assemble>";
                }
                DEV_VERBOSE_DEBUG("[DecideSlotAddress]   Slot [%3lu]: addr %s%s.",
                    slotIdx, desc.Dump().c_str(), extraAttr);
            }
        }
    }

    void DecideSlotAddress(DeviceExecuteSlot *slotList, size_t slotSize,
                           ItemPool<uint32_t, WsMemCategory::ITEMPOOL_SLOT_REF_CNT> &slotRefCntPool) {
        static constexpr uint64_t NON_ADDR_MASK = UINT64_C(1) << 62;

        UNUSED(slotRefCntPool);
        UNUSED(NON_ADDR_MASK);

        DumpSlotInfo("Update before", slotList, slotSize);
        for (size_t slotIdx = 0; slotIdx < slotSize; ++slotIdx) {
            auto &slot = slotList[slotIdx];
            auto &desc = slot.desc;
            if (desc.IsAddress()) {
                continue;
            }

            auto &dup = stitchedList_[desc.dupIdx];
            auto &outcastDesc = dup.GetOutcastAddress(desc.outcastIdx);
            DEV_DEBUG_ASSERT(outcastDesc.IsAddress());
#if DEBUG_INFINITE_LIFETIME
            desc = outcastDesc;
#else
            auto *outcastRawTensor = dup.GetSource()->GetOutcastRawTensor(desc.outcastIdx);
            if (slot.IsFixedAddress() || (outcastRawTensor->linkedIncastId != -1) ||
                (dup.GetSource()->GetOutcast(desc.outcastIdx).exprListIndex != -1)) {
                 desc = outcastDesc;
                 continue;
             }

            uintdevptr_t outcastWsStandardAddr = dup.RuntimeOutcastBase() + outcastRawTensor->addrOffset;
            bool isStandardOutcastSlot = outcastDesc.addr == outcastWsStandardAddr ||
                                         (outcastDesc.addr & NON_ADDR_MASK) == NON_ADDR_MASK;
            if (!isStandardOutcastSlot) {
                desc = outcastDesc;
                continue;
            }
            if (outcastDesc.addr == outcastWsStandardAddr) {
                // First time meet this unsolved slot
                slotInfosInDecidingSlotMem_[slotIdx].slotPtr = workspace_->AllocateSlot(dup.GetSource()->GetRawName());
                slotInfosInDecidingSlotMem_[slotIdx].RefCntInit(slotRefCntPool);
                outcastDesc = AddressDescriptor(slotIdx ^ NON_ADDR_MASK); // mark as first slot
            } else {
                DEV_DEBUG_ASSERT((outcastDesc.addr & NON_ADDR_MASK) == NON_ADDR_MASK);
                size_t firstSlotIdx = outcastDesc.addr ^ NON_ADDR_MASK;
                slotInfosInDecidingSlotMem_[firstSlotIdx].RefCntInc(slotRefCntPool);
                slotInfosInDecidingSlotMem_[slotIdx] = slotInfosInDecidingSlotMem_[firstSlotIdx];
            }
#endif
        }
        for (size_t slotIdx = 0; slotIdx < slotSize; ++slotIdx) {
            auto &slot = slotList[slotIdx];
            auto &desc = slot.desc;
            if (desc.IsAddress()) {
                continue;
            }
#if DEBUG_INFINITE_LIFETIME
            DEV_ASSERT(false);
#endif
            auto &dup = stitchedList_[desc.dupIdx];
            auto &outcastDesc = dup.GetOutcastAddress(desc.outcastIdx);
            if (size_t firstSlotIdx = outcastDesc.addr ^ NON_ADDR_MASK; firstSlotIdx == slotIdx) {
                // First time meet this unsolved slot
                outcastDesc = AddressDescriptor(slotInfosInDecidingSlotMem_[firstSlotIdx].slotPtr);
            }

            slot.desc = outcastDesc;
            slot.RefCntCopyFrom(slotInfosInDecidingSlotMem_[slotIdx]);
        }

        DumpSlotInfo("Update after", slotList, slotSize);
    }

    void DecideIncastOutcast(uint64_t taskId) {
        (void)taskId;
        for (size_t funcIdx = 0; funcIdx < stitchedList_.size(); ++funcIdx) {
            auto &dup = stitchedList_[funcIdx];
            // decide incast address
            size_t incastSize = dup.GetSource()->GetIncastSize();
            for (size_t i = 0; i < incastSize; ++i) {
                auto &desc = dup.GetIncastAddress(i);
                if (!desc.IsAddress()) {
                    desc = stitchedList_[desc.dupIdx].GetOutcastAddress(desc.outcastIdx);;
                }
                DEV_DEBUG_ASSERT(desc.IsAddress());
            }

            // decide outcast address
            size_t outcastSize = dup.GetSource()->GetOutcastSize();
            for (size_t i = 0; i < outcastSize; ++i) {
                auto &desc = dup.GetOutcastAddress(i);
                if (!desc.IsAddress()) {
                    desc = stitchedList_[desc.dupIdx].GetOutcastAddress(desc.outcastIdx);
                }
                DEV_DEBUG_ASSERT(desc.IsAddress());
            }
        }
    }

    void MoveTo(DynDeviceTask *dynTask) {
        dynTask->stitchedList = std::move(stitchedList_);
        stitchedList_.clear();
        dynTask->devTask.coreFunctionCnt = stitchedCallOpSize_;
        stitchedCallOpSize_ = 0;

        DEV_ASSERT(dynTask->stitchedList.size() <= MAX_CACHED_FUNC_NUM);
        int size = static_cast<int>(dynTask->stitchedList.size());
        for (int i = 0; i < size; ++i) {
            auto &funcDup = dynTask->stitchedList[i];
            dynTask->dynFuncDataCacheList[i] = {
                funcDup.GetSource(), &funcDup.GetOperationCurrPredCount(0), funcDup.GetSource()->GetCalleeIndexAddr(), funcDup.DupDataForDynFuncData()};
        }
        dynTask->dynFuncDataCacheListSize = size;
    }

    void VerifyStitchedListMemory(DevStartArgs &args) const {
        workspace_->VerifyStitchedListMemory(args, stitchedList_.data(), stitchedList_.size());
    }

    static void PushBackTask(DevAscendFunctionDuppedStitchList &stitch, uint32_t coreTask,
                             DeviceWorkspaceAllocator *workspace) {
        stitch.PushBack(coreTask, [workspace] { return workspace->AllocateStitch(); });
    }

    uint32_t stitchedCallOpSize() { return stitchedCallOpSize_; }

private:
    struct SlotAdditionalInfo {
        uintdevptr_t slotPtr{0};
        int64_t refCntIndex{itemPoolInvalidIndex};

        template<WsMemCategory category>
        void RefCntInit(ItemPool<uint32_t, category> &slotRefCntPool) {
            refCntIndex = slotRefCntPool.Allocate(1);
        }

        template<WsMemCategory category>
        void RefCntInc(ItemPool<uint32_t, category> &slotRefCntPool) {
            (void)slotRefCntPool;
            ++slotRefCntPool.At(refCntIndex);
        }
    };
    uint32_t stitchedCallOpSize_{0};
    StitchedList stitchedList_;
    Vector<SlotAdditionalInfo, WsMemCategory::VECTOR_TEMPORARY> slotInfosInDecidingSlotMem_;
    DeviceWorkspaceAllocator *workspace_{nullptr};

public:
    enum class StitchKind {
        StitchDefault,
        StitchPartial,
        StitchFullCover,
        StitchReuse,
    };

    static std::string GetStitchKindName(StitchKind kind) {
        static std::unordered_map<StitchKind, std::string> stitchNameDict = {
            {StitchKind::StitchDefault, "default"},
            {StitchKind::StitchPartial, "partial"},
            {StitchKind::StitchFullCover, "fullCover"},
            {StitchKind::StitchReuse, "reuse"},
        };
        DEV_ASSERT(stitchNameDict.count(kind));
        return stitchNameDict.find(kind)->second;
    }

    static inline
    void HandleOneStitch(
            DevAscendFunctionDupped &producerDup, DevAscendFunctionDupped &consumerDup,
            DevAscendFunctionDuppedStitchList &producerStitchList, size_t producerOperationIdx,
            size_t consumerIdx, size_t consumerOperationIdx,
            DeviceWorkspaceAllocator *workspace,
            StitchKind debugStitchKind, int debugSlotIdx) {
        (void)debugStitchKind;
        (void)debugSlotIdx;

        PushBackTask(producerStitchList, MakeTaskID(consumerIdx, consumerOperationIdx), workspace);
        consumerDup.GetOperationCurrPredCount(consumerOperationIdx)++;

        DEV_IF_NONDEVICE {
            DEV_ASSERT(producerOperationIdx < producerDup.GetSource()->GetOperationSize());
            DEV_ASSERT(consumerOperationIdx < consumerDup.GetSource()->GetOperationSize());
            DEV_VERBOSE_DEBUG("[Stitch] slot:%d kind:%s dupIdx:%d funcKey:%d,op:%d -> funcKey:%d,op:%d\n",
                        debugSlotIdx, GetStitchKindName(debugStitchKind).c_str(), (int)consumerIdx,
                        producerDup.GetSource()->GetFuncKey(), (int)producerOperationIdx,
                        consumerDup.GetSource()->GetFuncKey(), (int)consumerOperationIdx);
        }
    }

    static inline
    void HandleOneStitch(
            DevAscendFunctionDupped &producerDup, DevAscendFunctionDupped &consumerDup,
            size_t producerOperationIdx, size_t consumerIdx, size_t consumerOperationIdx,
            DeviceWorkspaceAllocator *workspace, StitchKind debugStitchKind, int debugSlotIdx) {
        auto &producerStitchList = producerDup.GetOperationStitch(producerOperationIdx, false);
        HandleOneStitch(producerDup, consumerDup, producerStitchList, producerOperationIdx,
            consumerIdx, consumerOperationIdx, workspace, debugStitchKind, debugSlotIdx);
    }

    template<typename T>
    static inline std::string IntVecToStr(DevAscendFunctionDupped &dup, DevLocalVector<T> &vec) {
        std::stringstream ss;
        ss << "[";
        ss << dup.GetSource()->At(vec, 0);
        for (size_t i = 1; i < vec.size(); ++i) {
            ss << ", " << dup.GetSource()->At(vec, i);
        }
        ss << "]";
        return ss.str();
    }
    static inline std::string IntVecToStr(const uint64_t shape[DEV_SHAPE_DIM_MAX], int dim) {
        std::stringstream ss;
        ss << "[";
        ss << shape[0];
        for (size_t i = 1; i < static_cast<size_t>(dim); ++i) {
            ss << ", " << shape[i];
        }
        ss << "]";
        return ss.str();
    }

    uint64_t PartialUpdateStitch(DevAscendFunctionDupped &nextDup, size_t devTaskId, size_t devNextIdx,
            DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionIncast& incast) {
        uint64_t matchCount = 0;
        auto *nextSrc = nextDup.GetSource();
        auto expressionList = &nextDup.GetExpression(0);
        auto &cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
        auto partialUpdateTableData = &slot.partialUpdate->cellMatchRuntimePartialUpdateTable[0];
        struct HandleCellMatchPartial {
            static inline void Process(int index, uint64_t *cellMatchTableData, uint64_t *matchCount,
                    DevAscendFunctionDupped *stitchingList, int stitchingSize, DevAscendFunctionDupped *nextDup,
                    size_t devTaskId, size_t devNextIdx, int consumerOperationIdx,
                    DeviceWorkspaceAllocator *workspace,
                    int debugSlotIdx) {
                uint64_t id = cellMatchTableData[index];
                if (id != AICORE_TASK_INIT && devTaskId == (uint32_t)(id >> TASKID_SHIFT32)) {
                    auto funcId = FuncID(static_cast<uint32_t>(id));
                    auto producerOperationIdx = TaskID(static_cast<uint32_t>(id));
                    DevAscendFunctionDupped &prevDup = stitchingList[funcId];
                    (*matchCount)++;
                    DEV_VERBOSE_DEBUG("nextindex %lu stitch depend slot table cell[%d] = taskid(%u ! %u),", devNextIdx, index, funcId, producerOperationIdx);
                    DeviceStitchContext::HandleOneStitch(prevDup, *nextDup, producerOperationIdx, devNextIdx, consumerOperationIdx,
                        workspace, StitchKind::StitchPartial, debugSlotIdx);
                    DeviceStitchContext::CheckStitch(stitchingList, stitchingSize, nextDup);
                }
            }
        };
        for (size_t n = 0; n < incast.consumerList.size(); n++) {
            auto &consumer = nextSrc->At(incast.consumerList, n);
            uint64_t consumerOffset[DEV_SHAPE_DIM_MAX];
            uint64_t consumerShape[DEV_SHAPE_DIM_MAX];
            nextSrc->GetTensorOffsetAndShape<false>(
                    consumerOffset, consumerShape, expressionList, incast.dim, consumer.operationIdx, consumer.operandIdx, true);

            DEV_IF_VERBOSE_DEBUG {
                for (int j = 0; j < cellMatchTableDesc.GetDimensionSize(); j++) {
                    DEV_VERBOSE_DEBUG("PartialUpdateStitch cell match, operation[%d] -> dimension[%d] = (offset:%lu ,shape:%lu, cellshape:%d)",
                            consumer.operationIdx, j, consumerOffset[j], consumerShape[j], cellMatchTableDesc.cellShape.dim[j]);
                }
            }

            nextSrc->CellMatchHandle<HandleCellMatchPartial>(
                    consumerOffset, consumerShape, cellMatchTableDesc,
                    partialUpdateTableData, &matchCount, stitchedList_.data(), stitchedList_.size(), &nextDup,
                    devTaskId, devNextIdx, consumer.operationIdx, workspace_, slotIdx);
        }
        return matchCount;
    }

    uint64_t FullCoverDefaultUpdateStitch(DevAscendFunctionDupped &nextDup, size_t devNextIdx, DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionIncast& incast) {
        uint64_t matchCount = 0;
        DevAscendFunctionDupped &prevDup = stitchedList_[slot.stitchDupIdx];
        auto *prevSrc = prevDup.GetSource();
        auto &outcast = prevSrc->GetOutcast(slot.stitchOutcastIdx);
        auto *nextSrc = nextDup.GetSource();
        auto expressionList = &nextDup.GetExpression(0);
        auto &cellMatchTableDesc = outcast.cellMatchTableDesc;
        auto fullUpdateTableData = &prevSrc->At(outcast.cellMatchRuntimeFullUpdateTable, 0);
        struct HandleCellMatchFull {
            static inline void Process(
                    int index,
                    uint32_t *cellMatchTableData,
                    uint64_t *matchCount,
                    DevAscendFunctionDupped *prevDup, DevAscendFunctionDupped *nextDup,
                    size_t devNextIdx, int consumerOperationIdx,
                    DeviceWorkspaceAllocator *workspace,
                    int debugSlotIdx) {
                auto producerOperationIdx = cellMatchTableData[index];
                if (producerOperationIdx != (uint32_t)-1) {
                    (*matchCount)++;
                    DEV_TRACE_DEBUG(DEvent(DUid(none()), DActStitchEdge(
                        Producer(LUid(none(), 0, none(), producerOperationIdx, none()), none(), none(), debugSlotIdx, none(), none()),
                        Consumer(LUid(none(), 0, none(), consumerOperationIdx, none()), none(), none(), debugSlotIdx, none(), none()),
                        StitchReasonUniqueMatch())));
                    DeviceStitchContext::HandleOneStitch(*prevDup, *nextDup, producerOperationIdx, devNextIdx, consumerOperationIdx,
                        workspace, StitchKind::StitchDefault, debugSlotIdx);
                }
            }
        };
        for (size_t n = 0; n < incast.consumerList.size(); n++) {
            auto &consumer = nextSrc->At(incast.consumerList, n);
            uint64_t consumerOffset[DEV_SHAPE_DIM_MAX];
            uint64_t consumerShape[DEV_SHAPE_DIM_MAX];
            nextSrc->GetTensorOffsetAndShape<false>(
                    consumerOffset, consumerShape, expressionList, incast.dim, consumer.operationIdx, consumer.operandIdx, true);
            nextSrc->CellMatchHandle<HandleCellMatchFull>(
                    consumerOffset, consumerShape, cellMatchTableDesc,
                    fullUpdateTableData,
                    &matchCount,
                    &prevDup, &nextDup,
                    devNextIdx, consumer.operationIdx,
                    workspace_,
                    slotIdx);
            DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup);
        }
        return matchCount;
    }

    uint64_t FullCoverUpdateStitch(DevAscendFunctionDupped &nextDup, size_t devNextIdx, DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionIncast& incast) {
        DevAscendFunctionDupped &prevDup = stitchedList_[slot.stitchDupIdx];
        auto *prevSrc = prevDup.GetSource();
        auto &outcast = prevSrc->GetOutcast(slot.stitchOutcastIdx);
        auto *nextSrc = nextDup.GetSource();
        DEV_VERBOSE_DEBUG("outcast %lu is %d, cellMatchStaticOutcastTable is %s\n", (unsigned long)slot.stitchOutcastIdx,
            outcast.stitchByAllFullMatch, IntVecToStr(prevDup, outcast.cellMatchStaticOutcastTable).c_str());
        DEV_VERBOSE_DEBUG("=================FullCoverUpdateStitch %zu %zu %zu %zu %d %d===========================\n",
            outcast.producerList.size(), incast.consumerList.size(),
            outcast.cellMatchStaticOutcastTable.size(), incast.cellMatchStaticIncastTable.size(),
            outcast.stitchByAllFullMatch, incast.stitchByAllFullMatch);

        // stitchPolicyFullCover hub
        auto producerHubOpIdx = outcast.stitchPolicyFullCoverProducerHubOpIdx;
        if (producerHubOpIdx != -1) {
            auto consumerAllOpIdxList = &nextSrc->At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, 0);
            for (size_t conIndex = 0, conSize = incast.stitchPolicyFullCoverConsumerAllOpIdxList.size(); conIndex < conSize; conIndex++) {
                auto &consumerOpIdx = consumerAllOpIdxList[conIndex];
                DeviceStitchContext::HandleOneStitch(prevDup, nextDup, producerHubOpIdx, devNextIdx, consumerOpIdx,
                    workspace_, StitchKind::StitchFullCover, slotIdx);
            }
            DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup);
        } else {
            // stitchPolicyFullCover producer
            auto producerList = &prevSrc->At(outcast.stitchPolicyFullCoverProducerList, 0);
            auto consumerAllOpIdxList = &nextSrc->At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, 0);
            for (size_t prodIndex = 0, prodSize = outcast.stitchPolicyFullCoverProducerList.size(); prodIndex < prodSize; prodIndex++) {
                auto &producer = producerList[prodIndex];
                auto producerOperationIdx = producer.operationIdx;

                for (size_t conIndex = 0, conSize = incast.stitchPolicyFullCoverConsumerAllOpIdxList.size(); conIndex < conSize; conIndex++) {
                    auto &consumerOpIdx = consumerAllOpIdxList[conIndex];
                    DeviceStitchContext::HandleOneStitch(prevDup, nextDup, producerOperationIdx, devNextIdx, consumerOpIdx,
                        workspace_, StitchKind::StitchFullCover, slotIdx);
                }
            }
            DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup);
        }

        return FullCoverDefaultUpdateStitch(nextDup, devNextIdx, slot, slotIdx, incast);
    }

    void ReuseStitch(DevAscendFunctionDupped &nextDup, size_t devNextIdx) {
        if (nextDup.GetSource()->rawTensorWsMemoryRequirement == 0) {
            // 0 length workspace, no dependency in need
            return;
        }

        uintdevptr_t nextAddrL = nextDup.RuntimeWorkspace();
        uintdevptr_t nextAddrR = nextAddrL + nextDup.GetSource()->rawTensorWsMemoryRequirement;
        auto nextReuseInfo = nextDup.GetRuntimeReuseInfo();
        if (auto &firstDup = stitchedList_[stitchReuseContext_.firstDupIdx];
            firstDup.GetRuntimeReuseInfo().poolResetTimes >= nextReuseInfo.poolResetTimes) {
            return;
        }

        enum { SKIP_EMPTY = -2, INVALID_TOO_AHEAD = -1, NO_DEP = 0, NEEDS_DEP = 1 };

        auto needsDependency = [&](uint32_t prevIdx) -> int {
            if (prevIdx >= devNextIdx) {
                // invalid idx
                return INVALID_TOO_AHEAD;
            }

            auto &prevDup = stitchedList_[prevIdx];
            if (prevDup.GetSource()->rawTensorWsMemoryRequirement == 0) {
                // empty workspace
                return SKIP_EMPTY;
            }

            auto prevReuseInfo = prevDup.GetRuntimeReuseInfo();
            if (prevReuseInfo.poolResetTimes + 1 != nextReuseInfo.poolResetTimes) {
                return prevReuseInfo.poolResetTimes >= nextReuseInfo.poolResetTimes ? INVALID_TOO_AHEAD : NO_DEP;
            }

            // proper poolResetTimes
            stitchReuseContext_.lastNonEmptyDupIdx = prevIdx;

            uintdevptr_t prevAddrL = prevDup.RuntimeWorkspace();
            uintdevptr_t prevAddrR = prevAddrL + prevDup.GetSource()->rawTensorWsMemoryRequirement;
            return !(prevAddrR <= nextAddrL || prevAddrL >= nextAddrR) ? NEEDS_DEP : NO_DEP;
        };

        auto skipBefore = [](int result) { return result == NO_DEP || result == SKIP_EMPTY; };
        for (; skipBefore(needsDependency(stitchReuseContext_.firstDupIdx)); stitchReuseContext_.firstDupIdx++) {}

        if (needsDependency(stitchReuseContext_.firstDupIdx) == NEEDS_DEP) {
            for (uint32_t prevIdx = stitchReuseContext_.firstDupIdx; ; prevIdx++) {
                int res = needsDependency(prevIdx);
                if (res == NO_DEP || res == INVALID_TOO_AHEAD) {
                    break;
                }
                if (res != SKIP_EMPTY) {
                    auto &prevDup = stitchedList_[prevIdx];
                    StitchForWorkspaceReuse(stitchedList_.data(), stitchedList_.size(), prevDup, nextDup, devNextIdx, workspace_);
                    stitchReuseContext_.firstDupIdx = prevIdx; // Risk on time complexity: Duplicated access to empty-workspace funcs
                }
            }
        } else {
            if (stitchReuseContext_.lastNonEmptyDupIdx != -1) {
                auto &prevDup = stitchedList_[stitchReuseContext_.lastNonEmptyDupIdx];
                StitchForWorkspaceReuse(stitchedList_.data(), stitchedList_.size(), prevDup, nextDup, devNextIdx, workspace_);
            }
        }
    }

    uint64_t FastStitch(DeviceExecuteSlot *slotList, size_t slotSize, DevAscendFunctionDupped &nextDup, size_t devTaskId, size_t devNextIdx) {
        AutoScopedPerf asp(PERF_EVT_FAST_STITCH);
#if !ENABLE_STITCH
        return 0;
#endif
        auto *nextSrc = nextDup.GetSource();
        nextDup.GetSource()->GetFuncidx() = static_cast<int>(devNextIdx);
        if (devNextIdx == 0) {
            // The only function, don't need stitch
            return 0;
        }
        uint64_t matchCount = 0;
        for (size_t incastIdx = 0; incastIdx < nextSrc->GetIncastSize(); ++incastIdx) {
            auto &incast = nextSrc->GetIncast(incastIdx);

            for (size_t j = 0; j < incast.fromSlotList.size(); ++j) {
                auto slotIdx = nextSrc->At(incast.fromSlotList, j);
                if (slotIdx >= (int)slotSize) {
                    DEV_ERROR("slotIdx %d is larger than slotSize %zu!.", slotIdx, slotSize);
                    continue;
                }

                auto &slot = slotList[slotIdx];
                DEV_VERBOSE_DEBUG("FastStitch slot %d, incastindex %zu, ispartial %d, stitchDupIdx %u",
                    slotIdx, incastIdx, slot.isPartialUpdateStitch, slot.stitchDupIdx);
                if (slot.stitchDupIdx == INVALID_STITCH_IDX) {
                    // Slot never output
                    continue;
                }

                if (slot.isPartialUpdateStitch) {
                    matchCount = PartialUpdateStitch(nextDup, devTaskId, devNextIdx, slot, slotIdx, incast);
                    continue;
                }

                if (slot.desc.IsNullAddress()) {
                    continue;
                }
                DEV_VERBOSE_DEBUG("incast %zu is %d, cellMatchStaticIncastTable is %s\n", incastIdx, incast.stitchByAllFullMatch,
                    IntVecToStr(nextDup, incast.cellMatchStaticIncastTable).c_str());
                matchCount = FullCoverUpdateStitch(nextDup, devNextIdx, slot, slotIdx, incast);
            }
        }
#if !DEBUG_INFINITE_LIFETIME
        ReuseStitch(nextDup, devNextIdx);
#endif // !DEBUG_INFINITE_LIFETIME
        return matchCount;
    }

    static
    void DumpStitchInfo(DevAscendFunctionDupped *stitchedList, int stitchedSize) {
        int funcId = 0;
        for (int i = 0; i < stitchedSize; i++) {
            auto &funcDup = stitchedList[i];
            for (size_t opIndex = 0; opIndex < funcDup.GetSource()->GetOperationSize(); opIndex++) {
                auto &stitch = funcDup.GetOperationStitch(opIndex);
                if (stitch.IsNull()) {
                    continue;
                }
                std::stringstream oss;
                oss << stitch.Dump();
                DEV_VERBOSE_DEBUG("func %d opIndex %zu stitch list: %s.", funcId, opIndex, oss.str().c_str());
            }
            funcId++;
        }
    }

private:
    static
    bool MemOverlap(uint64_t ahead, uint64_t alength, uint64_t bhead, uint64_t blength) {
        return !(ahead + alength <= bhead || bhead + blength <= ahead);
    }

    static
    void StitchForWorkspaceReuse(
            DevAscendFunctionDupped *stitchingList, int stitchingSize,
            DevAscendFunctionDupped &prevDup, DevAscendFunctionDupped &currDup,
        size_t devCurrIdx, DeviceWorkspaceAllocator *workspace) {
        // Add dependency between root functions
        auto *prevSrc = prevDup.GetSource();
        auto *currSrc = currDup.GetSource();

        size_t prevNoSuccOpSize = prevSrc->GetNoSuccOpSize();
        size_t currNoPredOpSize = currSrc->GetNoPredOpSize();
        if (unlikely(prevNoSuccOpSize == 0 || currNoPredOpSize == 0)) {
            // Empty root function
            return;
        }

        // Graph has been optimized when encoding, we just put trivial full connection logics here
        for (size_t i = 0; i < prevNoSuccOpSize; ++i) {
            int prevNoSucc = prevSrc->GetNoSuccOpIdx(i);
            auto &stitch = prevDup.GetOperationStitch(prevNoSucc);
            for (size_t j = 0; j < currNoPredOpSize; ++j) {
                int currNoPred = currSrc->GetNoPredOpIdx(j);
                DEV_TRACE_DEBUG(DEvent(DUid(none()), DActStitchEdge(
                    Producer(LUid(none(), 0, none(), prevNoSucc, none()), none(), none(), none(), none(), none()),
                    Consumer(LUid(none(), 0, none(), currNoPred, none()), none(), none(), none(), none(), none()),
                    StitchReasonWorkspaceReuse())));
                DeviceStitchContext::HandleOneStitch(prevDup, currDup, stitch, prevNoSucc, devCurrIdx, currNoPred, workspace, DeviceStitchContext::StitchKind::StitchReuse, -1);
                DeviceStitchContext::CheckStitch(stitchingList, stitchingSize, &currDup);
            }
        }
    }
};
const uint32_t OP_ATTRS_PRE_NUM = 8;
const uint32_t OP_ATTRS_OFFSET_PRE_NUM = 4;
const uint32_t EXPR_TABLE_PRE_NUM = 8;
const uint32_t RAW_TENSOR_ADDR_MASK = 8;
const uint32_t CCE_BINARY_MOD = 8;
const size_t DUP_PRED_COUNT_LOOP_MAX = 8;
const size_t DUP_PRED_COUNT_PRE_LOOP_CNT = 4;
struct DeviceTaskContext {
    void InitAllocator(DevAscendProgram *devProg, DeviceWorkspaceAllocator &workspace, npu::tile_fwk::DevStartArgsBase *startArgs) {
        devProg_ = devProg;
        workspace_ = &workspace;
        startArgs_ = startArgs;
    }

    DynDeviceTask *BuildDeviceTaskData(DeviceStitchContext &stitchContext, uint32_t taskId, DevAscendProgram *devProg, bool withoutTail) {
        PerfBegin(PERF_EVT_ALLOCATE_TASK);
        DynDeviceTask *dynTask = workspace_->MakeDynDeviceTask();
        stitchContext.MoveTo(dynTask);
        PerfEnd(PERF_EVT_ALLOCATE_TASK);

        PerfBegin(PERF_EVT_BUILD_TASK_DATA);
        BuildDeviceTaskDataAndReadyQueue(dynTask, taskId, devProg);
        PerfEnd(PERF_EVT_BUILD_TASK_DATA);

        PerfBegin(PERF_EVT_SLAB_MEM_SUBMIT);
        // cache allocated memory , when task finish will recycle
        dynTask->taskStageAllocMem = workspace_->SlabGetStageAllocMem(withoutTail, WsAicpuSlabMemType::DUPPED_FUNC_DATA);
        workspace_->SlabStageAllocMemSubmmit(&dynTask->taskStageAllocMem);
        PerfEnd(PERF_EVT_SLAB_MEM_SUBMIT);
        return dynTask;
    }

    void ReleaseFinishedTasks(int perfEvtReleaseFinishTask, int perfEvtDeallocateTask) {
        (void)perfEvtReleaseFinishTask;
        (void)perfEvtDeallocateTask;
    }

    void AppendFinishTask(DynDeviceTask *dynTask) {
        (void)dynTask;
    }

    void ShowStats() {
        DEV_ERROR("   Stitched function count: %10lu.", stitchedFuncNum);
        DEV_ERROR("       Root function count: %10lu.", rootFuncNum);
        DEV_ERROR("       Leaf function count: %10lu.", leafFuncNum);
        DEV_ERROR("   Inital ready task count: %10lu.", readyTaskNum);
        DEV_ERROR(" Static function data size: %10lu bytes.", dynFuncDataSize);
        DEV_ERROR("   Leaf function data size: %10lu bytes.", leafFuncDataSize);
    }

    void UpdateReadyTaskNum(uint64_t cnt) { readyTaskNum += cnt; }
private:
    uint64_t stitchedFuncNum{0};
    uint64_t rootFuncNum{0};
    uint64_t leafFuncNum{0};
    uint64_t readyTaskNum {0};
    uint64_t dynFuncDataSize {0};
    uint64_t leafFuncDataSize {0};
private:
    DevAscendProgram *devProg_{nullptr};
    DeviceWorkspaceAllocator *workspace_{nullptr};
    npu::tile_fwk::DevStartArgsBase *startArgs_{nullptr};
private:
    void BuildReadyQueue(DynDeviceTask *dyntask, DevAscendProgram *devProg) {
        PerfBegin(PERF_EVT_READY_QUEUE_IN);
        uint32_t size = sizeof(ReadyCoreFunctionQueue) + dyntask->devTask.coreFunctionCnt * sizeof(taskid_t);
        DEV_ASSERT(dyntask->devTask.coreFunctionCnt <= devProg->singleLoopCallopMaxNum);
        ReadyCoreFunctionQueue *queue[READY_QUEUE_SIZE];
        for (size_t index = 0; index < READY_QUEUE_SIZE; ++index) {
            WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size, workspace_->SlabAlloc(size, WsAicpuSlabMemType::READY_QUE));
            ReadyCoreFunctionQueue *q = qalloc.As<ReadyCoreFunctionQueue>();
            q->head = 0;
            q->tail = 0;
            q->lock = 0;
            q->capacity = dyntask->devTask.coreFunctionCnt;
            q->elem = reinterpret_cast<taskid_t *>(q + 1);
            queue[index] = q;
            dyntask->readyQueue[index] = q;
        }

        ReadyCoreFunctionQueue *aivQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)];
        ReadyCoreFunctionQueue *aicQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)];
        ReadyCoreFunctionQueue *aicpuQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AICPU)];
        int aivQueueTail = 0;
        int aicQueueTail = 0;
        int aicpuQueueTail = 0;

        DynFuncDataCache *dynFuncDataCacheList = dyntask->GetDynFuncDataCacheList();
        uint32v8 one = {1, 1, 1, 1, 1, 1, 1, 1};
        uint32v8 base = {0, 1, 2, 3, 4, 5, 6, 7};
        size_t funcSize = dyntask->dynFuncDataCacheListSize;
        for (size_t funcIndex = 0; funcIndex < funcSize; ++funcIndex) {
            DevAscendFunctionDuppedData *duppedData = dynFuncDataCacheList->At(funcIndex).duppedData;
            predcount_t *dupPredCountList = &duppedData->GetOperationCurrPredCount(0);

            auto &predInfo = duppedData->GetSource()->GetPredInfo();
            size_t totalZeroPredAIVBatchEnd = predInfo.totalZeroPredAIV & ~0x7;
            taskid_t *aivQueueElemList = reinterpret_cast<taskid_t *>(aivQueue->elem);
            for (size_t opIndex = 0; opIndex < totalZeroPredAIVBatchEnd; opIndex += DUP_PRED_COUNT_LOOP_MAX) {
                if (likely((*reinterpret_cast<uint64_t *>(&dupPredCountList[opIndex]) | *reinterpret_cast<uint64_t *>(&dupPredCountList[opIndex + DUP_PRED_COUNT_PRE_LOOP_CNT]))) == 0) {
                    uint32v8 taskidv8 = (one * MakeTaskID(funcIndex, 0)) | (base + (uint32_t)opIndex);
#ifdef __x86_64__
                    memcpy_s(&aivQueueElemList[aivQueueTail], sizeof(taskidv8), &taskidv8, sizeof(taskidv8));
#else
                    *(uint32v8 *)&aivQueueElemList[aivQueueTail] = taskidv8;
#endif
                    aivQueueTail += DUP_PRED_COUNT_LOOP_MAX;
                } else {
                    for (size_t idx = 0; idx < DUP_PRED_COUNT_LOOP_MAX; ++idx) {
                        if (likely(dupPredCountList[opIndex + idx] == 0)) {
                            aivQueueElemList[aivQueueTail] = MakeTaskID(funcIndex, opIndex + idx);
                            aivQueueTail++;
                        }
                    }
                }
            }
            for (size_t opIndex = totalZeroPredAIVBatchEnd; opIndex < predInfo.totalZeroPredAIV; ++opIndex) {
                if (likely(dupPredCountList[opIndex] == 0)) {
                    aivQueue->elem[aivQueueTail] = MakeTaskID(funcIndex, opIndex);
                    aivQueueTail++;
                }
            }

            auto aicEnd = predInfo.totalZeroPredAIV + predInfo.totalZeroPredAIC;
            for (size_t opIndex = predInfo.totalZeroPredAIV; opIndex < aicEnd; ++opIndex) {
                if (likely(dupPredCountList[opIndex] == 0)) {
                    aicQueue->elem[aicQueueTail] = MakeTaskID(funcIndex, opIndex);
                    aicQueueTail++;
                }
            }
            auto aicpuEnd = predInfo.totalZeroPredAIV + predInfo.totalZeroPredAIC + predInfo.totalZeroPredAicpu;
            for (size_t opIndex = aicEnd; opIndex < aicpuEnd; ++opIndex) {
                if (likely(dupPredCountList[opIndex] == 0)) {
                    aicpuQueue->elem[aicpuQueueTail] = MakeTaskID(funcIndex, opIndex);
                    aicpuQueueTail++;
                }
            }
        }

        aivQueue->tail = static_cast<uint32_t>(aivQueueTail);
        aicQueue->tail = static_cast<uint32_t>(aicQueueTail);
        aicpuQueue->tail = static_cast<uint32_t>(aicpuQueueTail);
        dyntask->devTask.readyAivCoreFunctionQue = PtrToValue(aivQueue);
        dyntask->devTask.readyAicCoreFunctionQue = PtrToValue(aicQueue);
        dyntask->devTask.readyAicpuFunctionQue = PtrToValue(aicpuQueue);
        readyTaskNum += static_cast<uint64_t>(aivQueueTail + aicQueueTail + aicpuQueueTail);
        PerfEnd(PERF_EVT_READY_QUEUE_IN);
    }

    void BuildDynFuncData(DynDeviceTask *dyntask, uint32_t taskId, DevAscendProgram *devProg, DevAscendFunctionDupped *stitchedList, uint64_t stitchedSize) {
        size_t headerSize = sizeof(DynFuncHeader) + stitchedSize * sizeof(DynFuncData);
        auto header = workspace_->AllocateDynFuncData(headerSize);
        dyntask->dynFuncDataList = header;
        auto dyndata = &header->At(0);

        stitchedFuncNum++;

        header->funcSize = headerSize;
        header->seqNo = taskId;
        header->funcNum = stitchedSize;
        header->cceBinary = (DynFuncBin *) const_cast<DevCceBinary *>(dyntask->cceBinary);
        DEV_ASSERT((uint64_t)header->cceBinary % CCE_BINARY_MOD == 0);

        rootFuncNum += stitchedSize;
        for (size_t funcIndex = 0; funcIndex < stitchedSize; ++funcIndex) {
            auto &funcDup = stitchedList[funcIndex];
            dyndata->opAttrs = (uint64_t *) const_cast<SymInt *>(funcDup.GetSource()->GetSymoffset(0));
            dyndata->opAtrrOffsets = funcDup.GetSource()->GetOpAttrOffsetAddr();
            dyndata->exprNum = funcDup.GetSource()->expressionList.size();
            dyndata->exprTbl = funcDup.GetExpressionAddr();
            dyndata->rawTensorAddr = (uint64_t *)&funcDup.GetIncastAddress(0);
            dyndata->rawTensorDesc = funcDup.GetSource()->GetRawTensorDesc(0);
            dyndata->startArgs = this->startArgs_;
            dyndata->workspaceAddr = funcDup.RuntimeWorkspace();
            dyndata->stackWorkSpaceSize = workspace_->StandardStackWorkspacePerCore();
            dyndata->stackWorkSpaceAddr = workspace_->StackWorkspaceAddr();
            dyndata->opAttrSize = funcDup.GetSource()->GetOpAttrSize();
            dyndata->rawTensorAddrSize = funcDup.GetSource()->GetIncastSize();
            dyndata->rawTensorDescSize = funcDup.GetSource()->GetRawTensorDescSize();
            dyndata->commGroupNum = devProg->commGroupNum;
            DEV_ASSERT(sizeof(dyndata->hcclContext) == sizeof(devProg->hcclContext));
            (void)memcpy_s(dyndata->hcclContext, sizeof(dyndata->hcclContext), devProg->hcclContext, sizeof(devProg->hcclContext));
            DEV_ASSERT((uint64_t)dyndata->opAttrs % OP_ATTRS_PRE_NUM == 0);
            DEV_ASSERT((uint64_t)dyndata->opAtrrOffsets % OP_ATTRS_OFFSET_PRE_NUM == 0);
            DEV_ASSERT((uint64_t)dyndata->exprTbl % EXPR_TABLE_PRE_NUM == 0);
            DEV_ASSERT((uint64_t)dyndata->rawTensorAddr % RAW_TENSOR_ADDR_MASK == 0);

            leafFuncDataSize += funcDup.GetSource()->GetOpAttrSize() * sizeof(SymInt); // opAttrs
            leafFuncDataSize += funcDup.GetSource()->GetOperationSize() * sizeof(int32_t); // opAttrOffsts;
            leafFuncDataSize += dyndata->exprNum * sizeof(int64_t);
            leafFuncDataSize += funcDup.GetSource()->GetRawTensorSize() * sizeof(DevRawTensorDesc);

            leafFuncNum += funcDup.GetSource()->GetOperationSize();
            funcDup.SetFuncData(dyndata);
            dyndata++;
        }
        dynFuncDataSize += headerSize * sizeof(int64_t);
    }

    inline void doResolve(DynDeviceTask *dyntask, int coreType, size_t funcIdx, size_t succIdx, predcount_t *predList) {
        predList[succIdx] -= 1;
        if (predList[succIdx] != 0)
            return;

        if (coreType == static_cast<int>(CoreType::HUB)) {
            ResolveEarlyDepends(dyntask, funcIdx, succIdx);
        } else {
            auto q = dyntask->readyQueue[dyntask->GetReadyQueueIndexByCoreType(static_cast<CoreType>(coreType))];
            q->elem[q->tail++] = MakeTaskID(funcIdx, succIdx);
            readyTaskNum++;
        }
    }

    void ResolveEarlyDepends(DynDeviceTask *dyntask, size_t funcIdx, size_t opIdx) {
        size_t succSize;

        dyntask->devTask.coreFunctionCnt--;

        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->dynFuncDataCacheList[funcIdx].devFunc;
        auto predList = dyntask->dynFuncDataCacheList[funcIdx].predCount;
        auto succList = func->GetOperationDepGraphSuccAddr(opIdx, succSize);
        auto callList = dyntask->dynFuncDataCacheList[funcIdx].calleeList;

        for (size_t i = 0; i < succSize; ++i) {
            auto succIdx = succList[i];
            doResolve(dyntask, cceBinary[callList[succIdx]].coreType, funcIdx, succIdx, predList);
        }

        auto &funcDup = dyntask->stitchedList[funcIdx];
        auto &stitchList = funcDup.GetOperationStitch(opIdx);
        for (auto *node = stitchList.Head(); node != nullptr; node = node->Next()) {
            uint32_t listSize = node->Size();
            for (uint32_t i = 0; i < listSize; ++i) {
                uint32_t id = node->At(i);
                auto succFuncIdx = FuncID(id);
                auto succIdx = TaskID(id);
                predList = dyntask->dynFuncDataCacheList[succFuncIdx].predCount;
                callList = dyntask->dynFuncDataCacheList[succFuncIdx].calleeList;
                doResolve(dyntask, cceBinary[callList[succIdx]].coreType, succFuncIdx, succIdx, predList);
            }
        }
    }

    void ResolveEarlyDepends(DynDeviceTask *dyntask) {
        size_t funcSize = dyntask->stitchedList.size();
        for (size_t funcIdx = 0; funcIdx < funcSize; ++funcIdx) {
            auto func = dyntask->dynFuncDataCacheList[funcIdx].devFunc;
            auto predList = dyntask->dynFuncDataCacheList[funcIdx].predCount;
            auto &predInfo = func->GetPredInfo();
            auto opIdx = predInfo.totalZeroPredAIC + predInfo.totalZeroPredAIV + predInfo.totalZeroPredAicpu;
            while (opIdx < predInfo.totalZeroPred) {
                if (predList[opIdx] == 0) {
                    ResolveEarlyDepends(dyntask, funcIdx, opIdx);
                }
                opIdx++;
            }
        }
    }

public:
    static void DumpReadyQueue(DynDeviceTask *dynTask, const char *prefix) {
        DEV_ERROR("%s: coreFunctionCnt: %d", prefix, (int)dynTask->devTask.coreFunctionCnt);
        int aivIndex = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV);
        int aicIndex = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC);
        int aicpuIndex = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AICPU);

        DEV_ERROR("%s: ready queue aiv: %d-%d", prefix, (int)dynTask->readyQueue[aivIndex]->head, (int)dynTask->readyQueue[aivIndex]->tail);
        for (uint32_t i = dynTask->readyQueue[aivIndex]->head; i < dynTask->readyQueue[aivIndex]->tail; i++) {
            DEV_ERROR("%s: ready queue aiv[%d]: %x", prefix, (int)i, dynTask->readyQueue[aivIndex]->elem[i]);
        }

        DEV_ERROR("%s: ready queue aic: %d-%d", prefix, (int)dynTask->readyQueue[aicIndex]->head, (int)dynTask->readyQueue[aicIndex]->tail);
        for (uint32_t i = dynTask->readyQueue[aicIndex]->head; i < dynTask->readyQueue[aicIndex]->tail; i++) {
            DEV_ERROR("%s: ready queue aic[%d]: %x", prefix, (int)i, dynTask->readyQueue[aicIndex]->elem[i]);
        }

        DEV_ERROR("%s: ready queue aicpu: %d-%d", prefix, (int)dynTask->readyQueue[aicpuIndex]->head, (int)dynTask->readyQueue[aicpuIndex]->tail);
        for (uint32_t i = dynTask->readyQueue[aicpuIndex]->head; i < dynTask->readyQueue[aicpuIndex]->tail; i++) {
            DEV_ERROR("%s: ready queue aicpu[%d]: %x", prefix, (int)i, dynTask->readyQueue[aicpuIndex]->elem[i]);
        }
    }
    static void DumpDepend(DynDeviceTask *dyntask, DevAscendProgram *devProg, DevStartArgs *startArgs, const char *prefix) {
        (void)devProg;
        (void)startArgs;
        int total = 0;
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            ReadyCoreFunctionQueue *q = dyntask->readyQueue[i];
            total += q->tail - q->head;
        }
        DEV_ERROR("%s: ready total:%d", prefix, total);
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            ReadyCoreFunctionQueue *q = dyntask->readyQueue[i];
            for (uint32_t k = q->head; k < q->tail; k++) {
                uint32_t taskId = q->elem[k];
                uint32_t dupIndex = FuncID(taskId);
                uint32_t opIndex = TaskID(taskId);
                DEV_ERROR("%s: ready %d-%d:L(%d,%d,%d)\n",
                    prefix,
                    (int)i, (int)k,
                    (int)dyntask->GetDynFuncDataList()->seqNo, (int)dupIndex, (int)opIndex);
            }
        }
        DEV_ERROR("%s: workspace:%llx", prefix, (unsigned long long)startArgs->contextWorkspaceAddr);
        for (size_t i = 0; i < startArgs->inputTensorSize; i++) {
            DEV_ERROR("%s: input-%d:%llx", prefix, (int)i, (unsigned long long)startArgs->GetInputTensor(i).address);
        }
        for (size_t i = 0; i < startArgs->outputTensorSize; i++) {
            DEV_ERROR("%s: output-%d:%llx", prefix, (int)i, (unsigned long long)startArgs->GetOutputTensor(i).address);
        }
        std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
        DevProgramControlFlowCache::RelocBuildInputOutputDesc(cacheInputOutputDict, startArgs);
        RelocRange relocWorkspace(startArgs->contextWorkspaceAddr, 0);

        DynFuncHeader *dynFuncDataList = dyntask->GetDynFuncDataList();
        int deviceIndex = dynFuncDataList->seqNo;
        DynFuncDataCache *dynFuncDataCacheList = dyntask->GetDynFuncDataCacheList();
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); dupIndex++) {
            DynFuncData &dynFuncData = dynFuncDataList->At(dupIndex);
            DynFuncDataCache &dynFuncDataCache = dynFuncDataCacheList->At(dupIndex);

            DevAscendFunctionDuppedData *duppedData = dynFuncDataCache.duppedData;

            predcount_t *pred = &duppedData->GetOperationCurrPredCount(0);
            for (size_t opIndex = 0; opIndex < duppedData->GetOperationSize(); opIndex++) {
                DEV_ERROR("%s: L(%d,%d,%d) pred:%d\n",
                    prefix,
                    (int)deviceIndex, (int)dupIndex, (int)opIndex,
                    (int)pred[opIndex]);
            }
            for (size_t stitchIndex = 1; stitchIndex < duppedData->GetStitchSize(); stitchIndex++) {
                DevAscendFunctionDuppedStitchList stitchList = duppedData->GetStitch(stitchIndex);
                stitchList.ForEach([&](int succTaskId){
                    uint32_t succDupIndex = FuncID(succTaskId);
                    uint32_t succOpIndex = TaskID(succTaskId);
                    DEV_ERROR("%s: R(%d,%d).succ-%d: L(%d,%d,%d)\n",
                        prefix, (int)deviceIndex, (int)dupIndex, (int)stitchIndex,
                        (int)deviceIndex, (int)succDupIndex, (int)succOpIndex);
                });
            }
            for (size_t exprIndex = 0; exprIndex < duppedData->GetExpressionSize(); exprIndex++) {
                DEV_ERROR("%s: R(%d,%d).expr-%d: %lld\n",
                    prefix, (int)deviceIndex, (int)dupIndex, (int)exprIndex,
                    (long long)duppedData->GetExpression(exprIndex));
            }
            for (size_t incastIndex = 0; incastIndex < duppedData->GetIncastSize(); incastIndex++) {
                AddressDescriptor addr = duppedData->GetIncastAddress(incastIndex);
                AddressDescriptor addrDesc = addr;
                DevProgramControlFlowCache::RelocDescToCache(addrDesc, relocWorkspace, cacheInputOutputDict);

                DEV_ERROR("%s: R(%d,%d).incast-%d: 0x%llx - 0x%llx\n",
                    prefix, (int)deviceIndex, (int)dupIndex, (int)incastIndex,
                    (unsigned long long)addrDesc.GetAddressValue(),
                    (unsigned long long)addr.GetAddressValue());
            }
            for (size_t outcastIndex = 0; outcastIndex < duppedData->GetOutcastSize(); outcastIndex++) {
                AddressDescriptor addr = duppedData->GetOutcastAddress(outcastIndex);
                AddressDescriptor addrDesc = addr;
                DevProgramControlFlowCache::RelocDescToCache(addrDesc, relocWorkspace, cacheInputOutputDict);
                DEV_ERROR("%s: R(%d,%d).outcast-%d: 0x%llx - 0x%llx\n",
                    prefix, (int)deviceIndex, (int)dupIndex, (int)outcastIndex,
                    (unsigned long long)addrDesc.GetAddressValue(),
                    (unsigned long long)addr.GetAddressValue());
            }
            DEV_ERROR("%s: R(%d,%d).workspace: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetRuntimeWorkspace());
            DEV_ERROR("%s: R(%d,%d).outcastWorkspace: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetRuntimeOutcastWorkspace());

            DEV_ERROR("%s: R(%d,%d).opAttrList: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)dynFuncData.opAttrs);
            DEV_ERROR("%s: R(%d,%d).opAttrList:Dupped: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetSource()->GetSymoffset(0));

            DEV_ERROR("%s: R(%d,%d).opAttrOffsetList: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)dynFuncData.opAtrrOffsets);
            DEV_ERROR("%s: R(%d,%d).opAttrOffsetList:Dupped: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetSource()->GetOpAttrOffsetAddr());

            DEV_ERROR("%s: R(%d,%d).exprTbl: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)dynFuncData.exprTbl);
            DEV_ERROR("%s: R(%d,%d).exprTbl:Dupped: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetExpressionAddr());

            DEV_ERROR("%s: R(%d,%d).rawTensorDesc: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)dynFuncData.rawTensorDesc);
            DEV_ERROR("%s: R(%d,%d).rawTensorDesc:Dupped: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetSource()->GetRawTensorDesc(0));

            DEV_ERROR("%s: R(%d,%d).rawTensorDesc: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)dynFuncData.rawTensorAddr);
            DEV_ERROR("%s: R(%d,%d).rawTensorDesc:Dupped: 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)&duppedData->GetIncastAddress(0));
        }
    }

    void BuildDeviceTaskDataAndReadyQueue(DynDeviceTask *dyntask, uint32_t taskId, DevAscendProgram *devProg) {
        dyntask->cceBinary = devProg->GetCceBinary(0);
        dyntask->aicpuLeafBinary = devProg->GetAicpuLeafBinary(0);
        DeviceStitchContext::CheckStitch(dyntask);

        DEV_VERBOSE_DEBUG("build ready queue.");
        PerfBegin(PERF_EVT_READY_QUEUE);
        BuildReadyQueue(dyntask, devProg);
        PerfEnd(PERF_EVT_READY_QUEUE);

        PerfBegin(PERF_EVT_RESOLVE_EARLY);
        ResolveEarlyDepends(dyntask);
        PerfEnd(PERF_EVT_RESOLVE_EARLY);

        DEV_VERBOSE_DEBUG("build func data.");
        PerfBegin(PERF_EVT_CORE_FUNCDATA);
        BuildDynFuncData(dyntask, taskId, devProg, &dyntask->stitchedList[0], dyntask->stitchedList.size());
        PerfEnd(PERF_EVT_CORE_FUNCDATA);
        DEV_INFO("Finish build a new device task.");

        DEV_IF_NONDEVICE {
            dyntask->DumpTopo();
        }

#if DEBUG_INFINITE_LIFETIME
        DEV_IF_DEVICE {
            dyntask->DumpTensorAddrInfo(workspace_->DumpTensorWsBaseAddr(), workspace_->DumpTensorWsSize());
        }
#endif
        DEV_IF_VERBOSE_DEBUG {
            dyntask->DumpLeafs();
        }
        dyntask->stitchedList.clear();
    }
};
const uint64_t SLEEP_TIME_US = 10000;
struct DeviceExecuteContext {
    typedef std::function<void(DynDeviceTask *, DeviceExecuteContext *)> PushTaskEntry;
    PushTaskEntry pushTask;

    DevStartArgs *args{nullptr};
    uint64_t taskId{0};
    bool isFirstTaskSend{true};

    DevAscendProgram *devProg{nullptr};
    DeviceExecuteProgram execProg;
    uint16_t stitchTaskLoopNumThreshold{MAX_CACHED_FUNC_NUM};

    DeviceWorkspaceAllocator workspace;

    DeviceSlotContext slotContext;

    DeviceStitchContext stitchContext;

    DeviceTaskContext taskContext;

    Vector<uint64_t, WsMemCategory::VECTOR_SYMBOL_TABLE> symbolTable;

    DevAscendFunctionDupped currDevRootDup;

    CostModel::ModelData *costModelData{nullptr};

    void *aicoreModel{nullptr};

    SPSCQueue<DynDeviceTask *, SUBMMIT_TASK_QUE_SIZE> submmitTaskQueue_;

    uint64_t duppedRootCount{0};
    bool controlFlowCacheActivated{false};

    bool DuppedRootCached() {
        if (!controlFlowCacheActivated) {
            return false;
        }
        return duppedRootCount < devProg->controlFlowCache.rootTaskCount;
    }

    bool DuppedRootUpdateAndCachedAllSubmitted() {
        if (!controlFlowCacheActivated) {
            return false;
        }
        duppedRootCount++;
        return duppedRootCount == devProg->controlFlowCache.rootTaskCount;
    }

    static uint64_t GetInputShapeDimSize(DeviceExecuteContext *ctx, uint64_t inputIndex) {
        DevTensorData *input = &ctx->args->devTensorList[inputIndex];
        return input->shape.dimSize;
    }
    static uint64_t GetInputShapeDim(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t n) {
        DevTensorData *input = &ctx->args->devTensorList[inputIndex];
        return input->shape.dim[n];
    }
    static int64_t GetInputDataInt32Dim1(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0) {
        DevTensorData *input = &ctx->args->devTensorList[inputIndex];
        return ((int32_t *)input->address)[off0];
    }
    static int64_t GetInputDataInt32Dim2(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1) {
        DevTensorData *input = &ctx->args->devTensorList[inputIndex];
        return ((int32_t *)input->address)[off0 * input->shape.dim[1] + off1];
    }
    static int64_t GetInputDataInt32Dim3(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1, uint64_t off2) {
        DevTensorData *input = &ctx->args->devTensorList[inputIndex];
        return ((int32_t *)input->address)[off0 * input->shape.dim[1] * input->shape.dim[2] + off1 * input->shape.dim[2] + off2]; // 2: dim 2
    }
    static int64_t GetInputDataInt32Dim4(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1,
        uint64_t off2, uint64_t off3) {
        DevTensorData *input = &ctx->args->devTensorList[inputIndex];
        return ((int32_t *)input->address)[((off0 * input->shape.dim[1] + off1) * input->shape.dim[2] + off2) * input->shape.dim[3] + off3]; // 2: dim 2, 3: dim 3
    }

    static void *SymbolHandlerIdToHandler(SymbolHandlerId id) {
        switch (id) {
            case SymbolHandlerId::GetInputShapeDimSize:
                return (void *)GetInputShapeDimSize;
            case SymbolHandlerId::GetInputShapeDim:
                return (void *)GetInputShapeDim;
            case SymbolHandlerId::GetInputDataInt32Dim1:
                return (void *)GetInputDataInt32Dim1;
            case SymbolHandlerId::GetInputDataInt32Dim2:
                return (void *)GetInputDataInt32Dim2;
            case SymbolHandlerId::GetInputDataInt32Dim3:
                return (void *)GetInputDataInt32Dim3;
            case SymbolHandlerId::GetInputDataInt32Dim4:
                return (void *)GetInputDataInt32Dim4;
            default:
                DEV_ASSERT(0);
                return nullptr;
        }
        return nullptr;
    }

    DeviceExecuteContext(DevStartArgs *startArgs) {
        PerfBegin(PERF_EVT_INIT);
        this->devProg = startArgs->devProg;

        DEV_IF_VERBOSE_DEBUG {
            std::string dump = devProg->Dump(0, true);
            DEV_VERBOSE_DEBUG("[DEVICE] %s.", dump.c_str());
        }

        PerfBegin(PERF_EVT_CONTROL_FLOW_MAPEXE);
        execProg = DeviceExecuteProgram(devProg, (AOTBinaryControlFlow::controlFlowEntry)startArgs->controlFlowEntry);
        AOTCodePool::GetCodePool().MapExec();
        PerfEnd(PERF_EVT_CONTROL_FLOW_MAPEXE);
        PerfEnd(PERF_EVT_INIT);
    }

    void ShowStats() {
        taskContext.ShowStats();
        workspace.DumpMemoryUsage("End ExecDyn");
    }

    void RunInit(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
        PerfBegin(PERF_EVT_CONTROL_FLOW_INIT);
        this->pushTask = tPushTask;
        this->args = startArgs;
        this->devProg = startArgs->devProg;

        workspace.Init(startArgs);
        if (devProg->firstStitchTaskLoopNum > 0) {
            stitchTaskLoopNumThreshold = std::min<uint16_t>(devProg->firstStitchTaskLoopNum, MAX_CACHED_FUNC_NUM);
            DEV_INFO("first stitch task loop num threshold is %u.", stitchTaskLoopNumThreshold);
        }

        slotContext.InitAllocator(workspace, devProg->slotSize);
        slotContext.FillInputOutputSlot(devProg, startArgs);

        stitchContext.Init(devProg, workspace);

        taskContext.InitAllocator(devProg, workspace, startArgs);

        workspace.SetupVector(symbolTable);
        symbolTable.resize(devProg->symbolTable.size());
        for (int i = 0; i < startArgs->GetInputSymbolSize(); ++i) {
            DevInputSymbol &param = startArgs->GetInputSymbol(i);
            int inputSymbolIndex = this->devProg->startArgsInputSymbolIndexList[i];
            symbolTable[inputSymbolIndex] = param.value;
            DEV_INFO("Param %d Symbol Table %d = %lu.", i, inputSymbolIndex, param.value);
        }

        for (size_t i = 0; i < this->devProg->startArgsSymbolHandlerList.size(); ++i) {
            SymbolHandler &symbolHandler = this->devProg->startArgsSymbolHandlerList[i];
            void *handler = SymbolHandlerIdToHandler(symbolHandler.handlerId);
            DEV_ASSERT_MSG(handler, "handler not found.");
            symbolTable[symbolHandler.symIndex] = PtrToValue(handler);
        }

        /* This initialization must only occur after all other AICPU workspace meta memory allocations have completed.
           The remaining portion of AICPU workspace meta memory must support reclamation. */
        workspace.InitMetadataSlabAllocator();

        PerfEnd(PERF_EVT_CONTROL_FLOW_INIT);
        DEV_INFO("Image size = %lu.", devProg->GetSize());
    }

    void PushTask(DynDeviceTask *dynTask) {
        pushTask(dynTask, this);
        taskId++;
    }

    void GELaunchRunCached(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
        PerfBegin(PERF_EVT_CONTROL_FLOW_INIT);
        this->pushTask = tPushTask;
        this->args = startArgs;
        this->devProg = startArgs->devProg;
        PerfEnd(PERF_EVT_CONTROL_FLOW_INIT);
        PerfMtTrace(PERF_TRACE_INIT, CTRL_CPU_THREAD_IDX);
        PerfBegin(PERF_EVT_CONTROL_FLOW);
        for (size_t i = 0; i < devProg->controlFlowCache.deviceTaskCount; i++) {
            DynDeviceTask *dynTask = (DynDeviceTask *)devProg->controlFlowCache.deviceTaskCacheList[i].dynTaskBase;
            devProg->controlFlowCache.PredCountDataRestore(dynTask);
            devProg->controlFlowCache.ReadyQueueDataRestore(dynTask);
            taskContext.UpdateReadyTaskNum(dynTask->readyQueueBackup->readyTaskNum);

            PROF_STAGE_BEGIN(PERF_EVT_STAGE_PUSH_TASK, "push.before\n");
            DumpDeviceTask(taskId, dynTask);
            PushTask(dynTask);
            PerfMtTrace(PERF_TRACE_DEV_TASK_BUILD, CTRL_CPU_THREAD_IDX);
            PROF_STAGE_END(PERF_EVT_STAGE_PUSH_TASK, "push.after\n");
        }
        PerfEnd(PERF_EVT_CONTROL_FLOW);
    }

    void RunControlFlow(DevStartArgs *startArgs) {
        PerfBegin(PERF_EVT_CONTROL_FLOW);
        CallRootEntryType callRootList[static_cast<uint32_t>(CallRootStage::T_CALLROOT_MAX)] = {
            DeviceExecuteCallAlloc,
            DeviceExecuteCallStitch,
            DeviceExecuteRuntimerLog,
            DeviceExecuteShmemAlloctor,
        };
        execProg.controlFlowBinary.CallControlFlow(this, symbolTable.data(), callRootList, startArgs);
        PerfEnd(PERF_EVT_CONTROL_FLOW);
    }

    void GELaunchFullCacheRunControlFlow(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
        RunInit(startArgs, tPushTask);
        RunControlFlow(startArgs);
    }

    void GELaunchFullCache(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
        if (devProg->controlFlowCache.IsActivatedFullCache(startArgs)) {
            DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCacheFullRunCache()));
            GELaunchRunCached(startArgs, tPushTask);
        } else {
            DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCacheFullRunControl()));
            GELaunchFullCacheRunControlFlow(startArgs, tPushTask);
        }
    }

    void GELaunchPartialCache(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
        DEV_TRACE_DEBUG(CtrlEvent(none(), Workspace(Range(startArgs->contextWorkspaceAddr, startArgs->contextWorkspaceAddr + startArgs->contextWorkspaceSize))));

        if (devProg->controlFlowCache.IsActivatedPartialCache(startArgs)) {
            controlFlowCacheActivated = true;
            DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCachePartRunCache(devProg->controlFlowCache.deviceTaskCount, devProg->controlFlowCache.rootTaskCount)));
            GELaunchRunCached(startArgs, tPushTask);
        }

        DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCacheFullRunControl()));
        RunInit(startArgs, tPushTask);
        RunControlFlow(startArgs);
    }

    void GELaunch(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
        if (devProg->controlFlowCache.IsRecording()) {
            devProg->controlFlowCache.InitInputOutput(startArgs);
        }
        GELaunchPartialCache(startArgs, tPushTask);
    }

    bool AiCoreFree() {
        return false; // extend check point
    }

    static void DumpDeviceTask(uint64_t taskId, DynDeviceTask *deviceTask) {
        DEV_IF_VERBOSE_DEBUG {
        } else {
            return;
        }
        for (uint64_t dupIdx = 0; dupIdx < deviceTask->dynFuncDataCacheListSize; dupIdx++) {
            DevAscendFunctionDuppedData *dupped = deviceTask->dynFuncDataCacheList[dupIdx].duppedData;

            size_t incastSize = dupped->GetSource()->GetIncastSize();
            DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), RActIncastCount(incastSize)));
            for (size_t i = 0; i < incastSize; ++i) {
                DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), RActIncast(i, dupped->SchemaGetIncastRange(i))));
            }

            size_t outcastSize = dupped->GetSource()->GetOutcastSize();
            DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), RActOutcastCount(outcastSize)));
            for (size_t i = 0; i < outcastSize; ++i) {
                DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), RActOutcast(i, dupped->SchemaGetOutcastRange(i))));
            }
        }
    }

    void SubmitToAicoreAndRecycleMemory(bool withoutTail) {
        DEV_VERBOSE_DEBUG("submit stitch task.");
        DEV_TRACE_DEBUG(DEvent(taskId, DActSubmit(stitchContext.Size())));
        AutoScopedPerf asp(PERF_EVT_SUBMIT_AICORE);
        if (stitchContext.Empty()) {
            DEV_INFO("stitch context is empty.");
            return;
        }

        PROF_STAGE_BEGIN(PERF_EVT_DECIDE_SLOT_ADDRESS, "slotaddr.before\n");
        stitchContext.DecideSlotAddress(
            slotContext.GetSlotList(), slotContext.GetSlotSize(), slotContext.GetSlotRefCntPool());
        PROF_STAGE_END(PERF_EVT_DECIDE_SLOT_ADDRESS, "slotaddr.after\n");

        PROF_STAGE_BEGIN(PERF_EVT_DECIDE_INCAST_ADDRESS, "incastaddr.before\n");
        stitchContext.DecideIncastOutcast(taskId);
        PROF_STAGE_END(PERF_EVT_DECIDE_INCAST_ADDRESS, "incastaddr.after\n");

        DEV_IF_VERBOSE_DEBUG {
                stitchContext.DumpStitchInfo();
#if !DEBUG_INFINITE_LIFETIME
                stitchContext.VerifyStitchedListMemory(*args);
#endif
        }

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        workspace.MarkAsNewStitchWindow();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

        PROF_STAGE_BEGIN(PERF_EVT_STAGE_BUILD_TASK, "BuildDeviceTaskData.before\n");
        DynDeviceTask *dynTask = taskContext.BuildDeviceTaskData(stitchContext, taskId, devProg, withoutTail);
        PROF_STAGE_END(PERF_EVT_STAGE_BUILD_TASK, "BuildDeviceTaskData.after\n");

        PROF_STAGE_BEGIN(PERF_EVT_DEALLOCATE_WORKSPACE, "RecycleTensorWorkspace.before\n");
        // Memory recycling
        stitchContext.RecycleTensorWorkspace();

        // Reset stitch context
        stitchContext.Reset();
        slotContext.ClearDirty();
        PROF_STAGE_END(PERF_EVT_DEALLOCATE_WORKSPACE, "RecycleTensorWorkspace.after\n");

        if (devProg->controlFlowCache.IsRecording()) {
            if (!devProg->controlFlowCache.IsRecordingStopped()) {
                devProg->controlFlowCache.PredCountDataBackup(dynTask);
                devProg->controlFlowCache.ReadyQueueDataBackup(dynTask);
                devProg->controlFlowCache.IncastOutcastAddrBackup(dynTask);
                devProg->controlFlowCache.RuntimeAddrBackup(slotContext.GetSlotList(), &slotContext.GetSlotRefCntPool().At(0), devProg->slotSize, workspace.GetTensorAllocator());
            }
            devProg->controlFlowCache.AppendDeviceTask(dynTask);
        }

        PROF_STAGE_BEGIN(PERF_EVT_STAGE_PUSH_TASK, "push.before\n");
        DumpDeviceTask(taskId, dynTask);
        PushTask(dynTask);
        PROF_STAGE_END(PERF_EVT_STAGE_PUSH_TASK, "push.after\n");
        PerfMtTrace(PERF_TRACE_DEV_TASK_BUILD, CTRL_CPU_THREAD_IDX);
    }

    schema::RUid GetRuid(uint64_t rootKey, bool afterAppend = false) {
        int64_t dupIndex = stitchContext.Size();
        if (afterAppend) {
            dupIndex -= 1;
        }
        schema::RUid ruid(taskId, dupIndex, rootKey);
        return ruid;
    }

    void ControlFlowCacheStopCache(uint64_t rootKey) {
        SubmitToAicoreAndRecycleMemory(false);
        devProg->controlFlowCache.StopRecording();
        DEV_INFO("stop recording for %d", (int)rootKey);
    }

    void *CallRootFunctionAlloc(uint64_t rootKey) {
        DevAscendFunction *devRoot = devProg->GetFunction(rootKey);
        DEV_DEBUG("alloc one func %lu %p %s.", rootKey, devRoot, devRoot->GetRawName());
        if (stitchContext.Size() == stitchTaskLoopNumThreshold ||
            stitchContext.stitchedCallOpSize() + devRoot->GetOperationSize() > devProg->singleLoopCallopMaxNum) {
            SubmitToAicoreAndRecycleMemory(false);
            stitchTaskLoopNumThreshold =
                std::min<uint16_t>(stitchTaskLoopNumThreshold + devProg->stitchTaskIncrLoopNum, MAX_CACHED_FUNC_NUM);
            DEV_INFO("next stitch task loop num threshold is %u.", stitchTaskLoopNumThreshold);
        }
        DEV_TRACE_DEBUG(REvent(GetRuid(rootKey), RActDup(devRoot->GetRawName())));

        PROF_STAGE_BEGIN(PERF_EVT_STAGE_DUP_ROOT, "dup.before\n");
        currDevRootDup = workspace.DuplicateRoot(devRoot);
        PROF_STAGE_END(PERF_EVT_STAGE_DUP_ROOT, "dup.after\n");
        return (void *)&currDevRootDup.GetExpression(0);
    }

    void *CallRootFunctionStitch(uint64_t rootKey) {
        DEV_DEBUG("root stitch %lu.", rootKey);
        if (rootKey == RUNTIME_FUNCKEY_CACHESTOP) {
            if (devProg->controlFlowCache.IsRecording()) {
                ControlFlowCacheStopCache(rootKey);
                return RUNTIME_FUNCRET_CACHESTOP_RETURN;
            } else {
                return RUNTIME_FUNCRET_CACHESTOP_CONTINUE;
            }
        }
        if (rootKey == RUNTIME_FUNCKEY_FINISH) {
            DEV_INFO("Finish stitch loop.");
            SubmitToAicoreAndRecycleMemory(false);
            return nullptr;
        }

        DEV_TRACE_DEBUG(REvent(GetRuid(rootKey), currDevRootDup.SchemaGetExpressionTable()));
        // dyn rawshape size depend expresstable calculated
        while (!workspace.TryAllocateFunctionMemory(currDevRootDup, slotContext.GetSlotList())) {
            // Failed to allocate, failed to stitch, submit existing stitched window to aicore and recycle memory
            // If nothing stitched, wait for aicore to finish tasks and release enough memory
            SubmitToAicoreAndRecycleMemory(true);
        }

        if (AiCoreFree()) {
            SubmitToAicoreAndRecycleMemory(false);
        }

        DEV_TRACE_DEBUG(DEvent(taskId, DActStitchStart(GetRuid(rootKey))));
        PROF_STAGE_BEGIN(PERF_EVT_STAGE_STITCH, "stitch.before\n");
        size_t devNextIdx = stitchContext.Size();
        stitchContext.Stitch(slotContext, currDevRootDup, taskId, devNextIdx);

        slotContext.UpdateSlots(currDevRootDup, stitchContext.GetStitchedList(), taskId, devNextIdx);
        PROF_STAGE_END(PERF_EVT_STAGE_STITCH, "stitch.after\n");
        DEV_TRACE_DEBUG(DEvent(taskId, DActStitchFinish(GetRuid(rootKey, true))));
        return nullptr;
    }

private:
    static void *DeviceExecuteCallAlloc(void *ctx_, uint64_t rootKey) {
        DeviceExecuteContext *ctx = (DeviceExecuteContext *)ctx_;
        if (ctx == nullptr) {
            DEV_ERROR("invalid ctx.");
            return nullptr;
        }
        PerfBegin(PERF_EVT_ROOT_FUNC);
        void *result = nullptr;
        if (ctx->DuppedRootCached()) {
            result = nullptr;
        } else if (ctx->devProg->controlFlowCache.IsRecording() && ctx->devProg->controlFlowCache.IsRecordingStopped()) {
            result = nullptr;
        } else {
            result = ctx->CallRootFunctionAlloc(rootKey);
        }
        PerfEnd(PERF_EVT_ROOT_FUNC);
        return result;
    }
    static void *DeviceExecuteCallStitch(void *ctx_, uint64_t rootKey) {
        DeviceExecuteContext *ctx = (DeviceExecuteContext *)ctx_;
        if (ctx == nullptr) {
            DEV_ERROR("invalid ctx.");
            return nullptr;
        }
        PerfBegin(PERF_EVT_ROOT_FUNC);
        void *result = nullptr;
        if (ctx->DuppedRootCached()) {
            result = nullptr;
        } else if (ctx->devProg->controlFlowCache.IsRecording() && ctx->devProg->controlFlowCache.IsRecordingStopped()) {
            result = nullptr;
        } else {
            result = ctx->CallRootFunctionStitch(rootKey);
        }
        PerfEnd(PERF_EVT_ROOT_FUNC);
        if (ctx->DuppedRootUpdateAndCachedAllSubmitted()) {
            DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCachePartRunControlContinue()));
            // forcely break device task
            ctx->devProg->controlFlowCache.RuntimeAddrRestore(ctx->slotContext.GetSlotList(), &ctx->slotContext.GetSlotRefCntPool().At(0), ctx->devProg->slotSize, ctx->workspace.GetTensorAllocator());
            ctx->devProg->controlFlowCache.RuntimeAddrRelocWorkspace(0, ctx->args->contextWorkspaceAddr, ctx->args, ctx->slotContext.GetSlotList());
        }
        return result;
    }
    static void *DeviceExecuteRuntimerLog(void *ctx_, uint64_t value) {
        (void)ctx_;
        DEV_DEBUG("DeviceExecuteRuntimerLog -> Value: %lu", value);
#if DEBUG_PLOG
        (void)value;
#endif
        return nullptr;
    }

    static void *DeviceExecuteShmemAlloctor(void *ctx_, uint64_t value) {
        (void)ctx_;
        uint64_t groupIndex = ((uint64_t*)value)[0];
        uint64_t memType = ((uint64_t*)value)[1];
        uint64_t size = ((uint64_t*)value)[2];
        static uint64_t offset = 0UL;
        constexpr uint64_t OFFSET_BITS = 58UL;
        constexpr uint64_t GROUP_BITS = 2UL;
        constexpr uint64_t MEMTYPE_BITS = 2UL;
        constexpr uint64_t GROUP_SHIFT = OFFSET_BITS;
        constexpr uint64_t MEMTYPE_SHIFT = GROUP_SHIFT + GROUP_BITS;
        constexpr uint64_t FILL_SHIFT = MEMTYPE_SHIFT + MEMTYPE_BITS;
        uint64_t vaddr = offset | (groupIndex << GROUP_SHIFT) | (memType << MEMTYPE_SHIFT) | (1UL << FILL_SHIFT);
        offset += size;
        return (void*)vaddr;
    }
};
} // namespace dynamic
} // namespace npu::tile_fwk
