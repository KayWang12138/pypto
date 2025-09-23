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
#if DEBUG_SWITCH
#include <map>
#endif

#ifndef STR
#define STR_(n)         #n
#define STR(n)          STR_(n)
#endif

#define AOT_CODE_POOL_CODE_SIZE     (4096 * 0x100)
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

    void UpdateSlots(DevAscendFunctionDupped &devRootDup, uint32_t devTaskId, uint32_t devNextIdx) {
        UpdateSlots(workspace_, slotList_.data(), slotList_.size(), slotRefCntPool_, devRootDup, devTaskId, devNextIdx);
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
            int inSlot = devProg->inplaceSlotList[i];
            if (inSlot != -1) {
                slotList[outSlot].desc = slotList[inSlot].desc;
                slotList[outSlot].isOutputSlot = true;
                DEV_INFO("Param %zu Output Slot %d = inSlot %d.", i, outSlot, inSlot);
            }
        }
        for (size_t i = 0; i < devProg->assembleSlotIndexList.size(); ++i) {
            int slotIndex = devProg->assembleSlotIndexList[i];
            slotList[slotIndex].isAssembleSlot = true;
            DEV_INFO("Assemble Slot %d.", slotIndex);
        }
        for (size_t i = 0, ie = devProg->partialUpdateList.size(); i < ie; i++) {
            auto &partialUpdate = devProg->At(devProg->partialUpdateList, i);
            int slotIndex = i;
            if (!partialUpdate.Empty()) {
                slotList[slotIndex].isPartialUpdateStitch = true;
                slotList[slotIndex].partialUpdate = &partialUpdate;
                DEV_INFO("Partial Update Slot %d\n", slotIndex);
            }
        }
        (void)slotSize;
    }

    static void UpdateSlotsForStitch(int slotIdx, DeviceExecuteSlot &slot, DevAscendFunction *devRootSrc, DevAscendFunctionOutcast &outcast,
                                     uint32_t devTaskId, uint32_t devNextIdx, uint32_t outcastIndex, uint64_t *expressionList) {
        (void)slotIdx;
        slot.stitchDupIdx = devNextIdx;
        slot.stitchOutcastIdx = outcastIndex;

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

            DEV_DEBUG("[UpdateSlots]  slot %d CellMatchPartial=%s\n", slotIdx,
                DevAscendFunctionDuppedStitchList::DumpTask<uint64_t>(tableData, slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size()).c_str());
            slot.isPartialUpdateDirty = true;
        } else {
            auto &cellMatchTableDesc = outcast.cellMatchTableDesc;
            auto tableData = &devRootSrc->At(outcast.cellMatchRuntimeFullUpdateTable, 0);
            devRootSrc->CellMatchFillIncastOutcast<false>(
                    producerList, outcast.producerList.size(), expressionList, false, cellMatchTableDesc, tableData);
            DEV_DEBUG("[UpdateSlots] slot %d  CellMatchFull=%s\n", slotIdx,
                DevAscendFunctionDuppedStitchList::DumpTask(tableData, outcast.cellMatchRuntimeFullUpdateTable.size()).c_str());
        }
    }

    template <WsMemCategory category>
    static void UpdateSlots(DeviceWorkspaceAllocator *workspace, DeviceExecuteSlot *slotList, int slotSize,
        ItemPool<uint32_t, category> &slotRefCntPool, DevAscendFunctionDupped &devRootDup, uint32_t devTaskId, uint32_t devNextIdx) {
        UNUSED(slotSize);

        AutoScopedPerf asp(PERF_EVT_UPDATE_SLOT);
        DevAscendFunction *devRootSrc = devRootDup.GetSource();
        uint64_t *expressionList = &devRootDup.GetExpression(0);
        size_t outcastSize = devRootSrc->GetOutcastSize();
        for (size_t i = 0; i < outcastSize; ++i) {
            auto &srcDesc = devRootDup.GetOutcastAddress(i);
            auto &outcast = devRootSrc->GetOutcast(i);
            for (size_t j = 0; j < outcast.toSlotList.size(); ++j) {
                int slotIdx = devRootSrc->At(outcast.toSlotList, j);
                auto &slot = slotList[slotIdx];
                UpdateSlotsForStitch(slotIdx, slot, devRootSrc, outcast, devTaskId, devNextIdx, i, expressionList);
                if (slot.refCnt != nullptr && slot.DerefAndCheckIfZeroRefCnt(slotRefCntPool)) {
                    DEV_DEBUG_ASSERT(!slot.desc.IsNullAddress());
                    workspace->DelayedRecycleSlotMem(slot.desc.addr);
                }

                if (!srcDesc.IsAddress() /* Unroll secondary placeholder */) {
                    slot.desc = srcDesc;
                } else {
                    slot.desc = AddressDescriptor(devNextIdx, i);
                }
                slot.refCnt = nullptr;
                DEV_DEBUG("[UpdateSlots]   Outcast [%3zu] to slot [%3d], address %s.", i, slotIdx, slot.desc.Dump().c_str());
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
        // static constant properties
        uint32_t workspaceRecyclePeriod{0};

        // changing with stitching progress
        uint32_t firstDupIdx{0};
    } stitchReuseContext_;

    void Init(DevAscendProgram *devProg, DeviceWorkspaceAllocator &workspace) {
        workspace.SetupVector(stitchedList_);
        workspace_ = &workspace;

        workspace_->SetupVector(slotInfosInDecidingSlotMem_);
        slotInfosInDecidingSlotMem_.resize(devProg->slotSize); // need pre alloc , left memory for slab allocator

        if (devProg->rootFuncStandardMemReq == 0) {
            // No memory to reuse
            stitchReuseContext_.workspaceRecyclePeriod = UINT32_MAX;
        } else {
            stitchReuseContext_.workspaceRecyclePeriod = devProg->workspaceRecyclePeriod;
        }

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
    }
    void Append(DevAscendFunctionDupped &devRootDup) { stitchedList_.push_back(devRootDup); }

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

    void RecycleAicoreLocalWorkspace() {
        AutoScopedPerf asp(PERF_EVT_DEALLOCATE_WORKSPACE);

        // recycle submitted tasks' workspace memory
        workspace_->RecycleDevFuncWorkspace();
        workspace_->TriggerDelayedRecycle();
    }

    void DumpSlotInfo(const char *label, DeviceExecuteSlot *slotList, size_t slotSize) {
#if DEBUG_SWITCH
        DEV_DEBUG("[DecideSlotAddress] %s.", label);
        for (size_t slotIdx = 0; slotIdx < slotSize; slotIdx++) {
            auto &desc = slotList[slotIdx].desc;
            const char *extraAttr = "";
            if (slotList[slotIdx].isOutputSlot) {
                extraAttr = " <output>";
            } else if (slotList[slotIdx].isAssembleSlot) {
                extraAttr = " <assemble>";
            }
            DEV_DEBUG("[DecideSlotAddress]   Slot [%3lu]: addr %s%s.",
                slotIdx, desc.Dump().c_str(), extraAttr);
        }
#else
        UNUSED(label);
        UNUSED(slotList);
        UNUSED(slotSize);
#endif // DEBUG_SWITCH
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
            if (slot.IsFixedAddress() || outcastRawTensor->linkedIncastId != -1) {
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
                slotInfosInDecidingSlotMem_[slotIdx].refCnt = slotRefCntPool.Make(1);
                outcastDesc = AddressDescriptor(slotIdx ^ NON_ADDR_MASK); // mark as first slot
            } else {
                DEV_DEBUG_ASSERT((outcastDesc.addr & NON_ADDR_MASK) == NON_ADDR_MASK);
                size_t firstSlotIdx = outcastDesc.addr ^ NON_ADDR_MASK;
                ++*slotInfosInDecidingSlotMem_[firstSlotIdx].refCnt;
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
            slot.refCnt = slotInfosInDecidingSlotMem_[slotIdx].refCnt;
        }

        DumpSlotInfo("Update after", slotList, slotSize);
    }

    void DecideIncastOutcast(uint64_t taskId) {
        (void)taskId;
        for (size_t funcIdx = 0; funcIdx < stitchedList_.size(); ++funcIdx) {
            auto &dup = stitchedList_[funcIdx];
            // decide incast address
            size_t incastSize = dup.GetSource()->GetIncastSize();
            DEV_TRACE_DEBUG(REvent(RUid(taskId, funcIdx, dup.GetSource()->GetRootIndex()), RActIncastCount(incastSize)));
            for (size_t i = 0; i < incastSize; ++i) {
                auto &desc = dup.GetIncastAddress(i);
                if (!desc.IsAddress()) {
                    desc = stitchedList_[desc.dupIdx].GetOutcastAddress(desc.outcastIdx);;
                }
                DEV_DEBUG_ASSERT(desc.IsAddress());
                DEV_TRACE_DEBUG(REvent(RUid(taskId, funcIdx, dup.GetSource()->GetRootIndex()), RActIncast(i, dup.SchemaGetIncastRange(i))));
            }

            // decide outcast address
            size_t outcastSize = dup.GetSource()->GetOutcastSize();
            DEV_TRACE_DEBUG(REvent(RUid(taskId, funcIdx, dup.GetSource()->GetRootIndex()), RActOutcastCount(outcastSize)));
            for (size_t i = 0; i < outcastSize; ++i) {
                auto &desc = dup.GetOutcastAddress(i);
                if (!desc.IsAddress()) {
                    desc = stitchedList_[desc.dupIdx].GetOutcastAddress(desc.outcastIdx);
                }
                DEV_DEBUG_ASSERT(desc.IsAddress());
                DEV_TRACE_DEBUG(REvent(RUid(taskId, funcIdx, dup.GetSource()->GetRootIndex()), RActOutcast(i, dup.SchemaGetOutcastRange(i))));
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
            dynTask->cacheList[i] = {funcDup.GetSource(), &funcDup.GetOperationCurrPredCount(0), funcDup.GetSource()->GetCalleeIndexAddr(), funcDup};
        }
    }

    void VerifyStitchedListMemory(DevStartArgs &args) const {
        workspace_->VerifyStitchedListMemory(args, stitchedList_.data(), stitchedList_.size());
    }

    void LogAicpuAlloc() {
        for (auto &&dup : stitchedList_) {
            workspace_->LogAicpuAlloc(dup);
        }
    }

    static void PushBackTask(DevAscendFunctionDuppedStitchList &stitch, uint32_t coreTask,
                             DeviceWorkspaceAllocator *workspace) {
        stitch.PushBack(coreTask, [workspace] { return workspace->AllocateStitch(); });
    }

    uint32_t stitchedCallOpSize() { return stitchedCallOpSize_; }

private:
    struct SlotAdditionalInfo {
        uintdevptr_t slotPtr{0};
        uint32_t *refCnt{nullptr};
    };
    uint32_t stitchedCallOpSize_{0};
    Vector<DevAscendFunctionDupped, WsMemCategory::VECTOR_STITCHED_LIST, DeviceWorkspaceAllocator> stitchedList_;
    Vector<SlotAdditionalInfo, WsMemCategory::VECTOR_TEMPORARY> slotInfosInDecidingSlotMem_;
    DeviceWorkspaceAllocator *workspace_{nullptr};

    uint32_t workspaceRecyclePeriod_{0};
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
            DEV_DEBUG("[Stitch] slot:%d kind:%s dupIdx:%d funcKey:%d,op:%d -> funcKey:%d,op:%d\n",
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
#if DEBUG_SWITCH
                    DEV_DEBUG("nextindex %lu stitch depend slot table cell[%d] = taskid(%u ! %u),", devNextIdx, index, funcId, producerOperationIdx);
#endif
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
#if DEBUG_SWITCH
            for (int j = 0; j < cellMatchTableDesc.GetDimensionSize(); j++) {
                DEV_DEBUG("PartialUpdateStitch cell match, operation[%d] -> dimension[%d] = (offset:%lu ,shape:%lu, cellshape:%d)",
                        consumer.operationIdx, j, consumerOffset[j], consumerShape[j], cellMatchTableDesc.cellShape.dim[j]);
            }
#endif
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
        DEV_DEBUG("outcast %lu is %d, cellMatchStaticOutcastTable is %s\n", (unsigned long)slot.stitchOutcastIdx,
            outcast.stitchByAllFullMatch, IntVecToStr(prevDup, outcast.cellMatchStaticOutcastTable).c_str());
        DEV_DEBUG("=================FullCoverUpdateStitch %zu %zu %zu %zu %d %d===========================\n",
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
        auto nextReuseInfo = nextDup.GetRuntimeReuseInfo();
        if (nextDup.GetSource()->rawTensorWsMemoryRequirement != 0) {
            auto &firstDup = stitchedList_[stitchReuseContext_.firstDupIdx];
            if (firstDup.GetRuntimeReuseInfo().poolResetTimes >= nextReuseInfo.poolResetTimes) {
                return;
            }

            auto isReusedBlock = [&](uint32_t prevIdx) {
                if (prevIdx >= devNextIdx) {
                    return false;
                }
                auto prevReuseInfo = stitchedList_[prevIdx].GetRuntimeReuseInfo();
                return prevReuseInfo.poolResetTimes + 1 == nextReuseInfo.poolResetTimes &&
                    prevReuseInfo.blockIdx == nextReuseInfo.blockIdx;
            };
            if (!isReusedBlock(stitchReuseContext_.firstDupIdx)) {
                uint32_t nextBlockFirstDupIdx = stitchReuseContext_.firstDupIdx;
                while (nextBlockFirstDupIdx < devNextIdx && !isReusedBlock(nextBlockFirstDupIdx)) {
                    nextBlockFirstDupIdx++;
                }
                stitchReuseContext_.firstDupIdx = nextBlockFirstDupIdx;
            }
            for (uint32_t prevIdx = stitchReuseContext_.firstDupIdx; isReusedBlock(prevIdx); prevIdx++) {
                auto &prevDup = stitchedList_[prevIdx];
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
#if DEBUG_SWITCH
                DEV_DEBUG("FastStitch slot %d, incastindex %zu, ispartial %d, stitchDupIdx %u",
                    slotIdx, incastIdx, slot.isPartialUpdateStitch, slot.stitchDupIdx);
#endif
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
                DEV_DEBUG("incast %zu is %d, cellMatchStaticIncastTable is %s\n", incastIdx, incast.stitchByAllFullMatch,
                    IntVecToStr(nextDup, incast.cellMatchStaticIncastTable).c_str());
                matchCount = FullCoverUpdateStitch(nextDup, devNextIdx, slot, slotIdx, incast);
            }
        }
        ReuseStitch(nextDup, devNextIdx);
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
                DEV_INFO("func %d opIndex %zu stitch list: %s.", funcId, opIndex, oss.str().c_str());
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
    void InitAllocator(DeviceWorkspaceAllocator &workspace, npu::tile_fwk::DevStartArgsBase *startArgs) {
        workspace_ = &workspace;
        startArgs_ = startArgs;
    }

    DynDeviceTask *BuildDeviceTaskData(DeviceStitchContext &stitchContext, DevAscendProgram *devProg, bool withoutTail) {
        PerfBegin(PERF_EVT_ALLOCATE_TASK);
        DynDeviceTask *dynTask = workspace_->MakeDynDeviceTask();
        stitchContext.MoveTo(dynTask);
        PerfEnd(PERF_EVT_ALLOCATE_TASK);
        BuildDeviceTaskData(dynTask, devProg);

        // cache allocated memory , when task finish will recycle
        dynTask->taskStageAllocMem = workspace_->SlabGetStageAllocMem(withoutTail, WsAicpuSlabMemType::DUPPED_FUNC_DATA);
        dynTask->isFinish.store(false);
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

private:
    uint64_t stitchedFuncNum{0};
    uint64_t rootFuncNum{0};
    uint64_t leafFuncNum{0};
    uint64_t readyTaskNum {0};
    uint64_t dynFuncDataSize {0};
    uint64_t leafFuncDataSize {0};
private:
    DeviceWorkspaceAllocator *workspace_{nullptr};
    npu::tile_fwk::DevStartArgsBase *startArgs_{nullptr};
private:
    void BuildReadyQueue(DynDeviceTask *dyntask) {
        uint32_t size = sizeof(ReadyCoreFunctionQueue) + dyntask->devTask.coreFunctionCnt * sizeof(taskid_t);
        DEV_ASSERT(dyntask->devTask.coreFunctionCnt <= MAX_READY_QUE_ELM_SIZE);
        ReadyCoreFunctionQueue *queue[READY_QUEUE_SIZE];
        for (size_t index = 0; index < READY_QUEUE_SIZE; ++index) {
            ReadyCoreFunctionQueue *q = workspace_->SlabAlloc(size, WsAicpuSlabMemType::READY_QUE).As<ReadyCoreFunctionQueue>();
            q->head = 0;
            q->tail = 0;
            q->lock = 0;
            q->capacity = dyntask->devTask.coreFunctionCnt;
            q->elem = reinterpret_cast<taskid_t *>(q + 1);
            queue[index] = q;
            dyntask->readyQueue[index] = q;
        }

        ReadyCoreFunctionQueue *aivQueue = queue[dyntask->GetReadyQueueIndexByCoreType(CoreType::AIV)];
        ReadyCoreFunctionQueue *aicQueue = queue[dyntask->GetReadyQueueIndexByCoreType(CoreType::AIC)];
        ReadyCoreFunctionQueue *aicpuQueue = queue[dyntask->GetReadyQueueIndexByCoreType(CoreType::AICPU)];
        int aivQueueTail = 0;
        int aicQueueTail = 0;
        int aicpuQueueTail = 0;

        uint32v8 one = {1, 1, 1, 1, 1, 1, 1, 1};
        uint32v8 base = {0, 1, 2, 3, 4, 5, 6, 7};
        size_t funcSize = dyntask->stitchedList.size();
        for (size_t funcIndex = 0; funcIndex < funcSize; ++funcIndex) {
            auto &dup = dyntask->stitchedList[funcIndex];
            predcount_t *dupPredCountList = &dup.GetOperationCurrPredCount(0);

            auto &predInfo = dup.GetSource()->GetPredInfo();
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
    }

    void BuildDynFuncData(DynDeviceTask *dyntask, DevAscendProgram *devProg) {
        size_t size = sizeof(DynFuncHeader) + dyntask->stitchedList.size() * sizeof(DynFuncData);
        auto header = workspace_->SlabAlloc(size, WsAicpuSlabMemType::DYN_FUNC_DATA).As<DynFuncHeader>();
        dyntask->dynFuncData = header;
        auto dyndata = (DynFuncData *)(header + 1);

        header->funcSize = size * sizeof(int64_t);
        header->seqNo = stitchedFuncNum++;
        header->funcNum = dyntask->stitchedList.size();
        header->cceBinary = (DynFuncBin *) const_cast<DevCceBinary *>(dyntask->cceBinary);
        DEV_ASSERT((uint64_t)header->cceBinary % CCE_BINARY_MOD == 0);

        rootFuncNum += dyntask->stitchedList.size();
        for (size_t funcIndex = 0; funcIndex < dyntask->stitchedList.size(); ++funcIndex) {
            auto &funcDup = dyntask->stitchedList[funcIndex];
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
        dynFuncDataSize += size * sizeof(int64_t);
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
        auto func = dyntask->cacheList[funcIdx].devFunc;
        auto predList = dyntask->cacheList[funcIdx].predCount;
        auto succList = func->GetOperationDepGraphSuccAddr(opIdx, succSize);
        auto callList = dyntask->cacheList[funcIdx].calleList;

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
                predList = dyntask->cacheList[succFuncIdx].predCount;
                callList = dyntask->cacheList[succFuncIdx].calleList;
                doResolve(dyntask, cceBinary[callList[succIdx]].coreType, succFuncIdx, succIdx, predList);
            }
        }
    }

    void ResolveEarlyDepends(DynDeviceTask *dyntask) {
        size_t funcSize = dyntask->stitchedList.size();
        for (size_t funcIdx = 0; funcIdx < funcSize; ++funcIdx) {
            auto func = dyntask->cacheList[funcIdx].devFunc;
            auto predList = dyntask->cacheList[funcIdx].predCount;
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

    void BuildDeviceTaskData(DynDeviceTask *dyntask, DevAscendProgram *devProg) {
        dyntask->cceBinary = devProg->GetCceBinary(0);
        dyntask->aicpuLeafBinary = devProg->GetAicpuLeafBinary(0);
        DeviceStitchContext::CheckStitch(dyntask);

        DEV_DEBUG("build ready queue.");
        PerfBegin(PERF_EVT_READY_QUEUE);
        BuildReadyQueue(dyntask);
        PerfEnd(PERF_EVT_READY_QUEUE);

        PerfBegin(PERF_EVT_RESOLVE_EARLY);
        ResolveEarlyDepends(dyntask);
        PerfEnd(PERF_EVT_RESOLVE_EARLY);

        DEV_DEBUG("build func data.");
        PerfBegin(PERF_EVT_CORE_FUNCDATA);
        BuildDynFuncData(dyntask, devProg);
        PerfEnd(PERF_EVT_CORE_FUNCDATA);
        DEV_INFO("start a new static func.");

        DEV_IF_NONDEVICE {
            dyntask->DumpTopo();
        }

#if DEBUG_INFINITE_LIFETIME
        if constexpr (IsDeviceMode()) {
            dyntask->DumpTensorAddrInfo(workspace_->DumpTensorWsBaseAddr(), workspace_->DumpTensorWsSize());
        }
#endif
#if DEBUG_SWITCH
        dyntask->DumpLeafs();
#endif
    }
};
const uint64_t SLEEP_TIME_US = 10000;
struct DeviceExecuteContext {
    std::function<void(uint64_t, DeviceTask *, DeviceExecuteContext *)> pushTask;
    DevStartArgs *args{nullptr};
    uint64_t taskId{0};

    DevAscendProgram *devProg{nullptr};
    DeviceExecuteProgram execProg;

    DeviceWorkspaceAllocator workspace;

    DeviceSlotContext slotContext;

    DeviceStitchContext stitchContext;

    DeviceTaskContext taskContext;

    Vector<uint64_t, WsMemCategory::VECTOR_SYMBOL_TABLE> symbolTable;

    DevAscendFunctionDupped currDevRootDup;

    CostModel::ModelData *costModelData{nullptr};

    void *aicoreModel{nullptr};

    SPSCQueue<DynDeviceTask *, SUBMMIT_TASK_QUE_SIZE> submmitTaskQueue_;

    static uint64_t GetInputShapeDimSize(DeviceExecuteContext *ctx, uint64_t inputIndex) {
        DevTensorData *input = &ctx->args->inputTensorList[inputIndex];
        return input->shape.dimSize;
    }
    static uint64_t GetInputShapeDim(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t n) {
        DevTensorData *input = &ctx->args->inputTensorList[inputIndex];
        return input->shape.dim[n];
    }
    static int64_t GetInputDataInt32Dim1(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0) {
        DevTensorData *input = &ctx->args->inputTensorList[inputIndex];
        return ((int32_t *)input->address)[off0];
    }
    static int64_t GetInputDataInt32Dim2(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1) {
        DevTensorData *input = &ctx->args->inputTensorList[inputIndex];
        return ((int32_t *)input->address)[off0 * input->shape.dim[1] + off1];
    }
    static int64_t GetInputDataInt32Dim3(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1, uint64_t off2) {
        DevTensorData *input = &ctx->args->inputTensorList[inputIndex];
        return ((int32_t *)input->address)[off0 * input->shape.dim[1] * input->shape.dim[2] + off1 * input->shape.dim[2] + off2]; // 2: dim 2
    }
    static int64_t GetInputDataInt32Dim4(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1,
        uint64_t off2, uint64_t off3) {
        DevTensorData *input = &ctx->args->inputTensorList[inputIndex];
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
#if DEBUG_SWITCH
        std::string dump = devProg->Dump(0, true);
        DEV_INFO("[DEVICE] %s.", dump.c_str());
#endif
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

    void GELaunch(DevStartArgs *startArgs, std::function<void(uint64_t, DeviceTask *, DeviceExecuteContext *)> tPushTask) {
        PerfBegin(PERF_EVT_CONTROL_FLOW_INIT);
        this->pushTask = tPushTask;
        this->args = startArgs;
        this->devProg = startArgs->devProg;

        workspace.Init(startArgs);

        slotContext.InitAllocator(workspace, devProg->slotSize);
        slotContext.FillInputOutputSlot(devProg, startArgs);

        stitchContext.Init(devProg, workspace);

        taskContext.InitAllocator(workspace, startArgs);

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
        workspace.InitAicpuMetaSlabAllocator();

        PerfEnd(PERF_EVT_CONTROL_FLOW_INIT);
        DEV_INFO("Image size = %lu.", devProg->GetSize());

        PerfBegin(PERF_EVT_CONTROL_FLOW);
        CallRootEntryType callRootList[static_cast<uint32_t>(CallRootStage::T_CALLROOT_MAX)] = {
            DeviceExecuteCallAlloc,
            DeviceExecuteCallStitch,
            DeviceExecuteRuntimerLog
        };
        execProg.controlFlowBinary.CallControlFlow(this, symbolTable.data(), callRootList, startArgs);
        PerfEnd(PERF_EVT_CONTROL_FLOW);
    }

    bool AiCoreFree() {
        return false; // extend check point
    }

    void SubmitToAicoreAndRecycleMemory(bool withoutTail) {
        DEV_TRACE_DEBUG(DEvent(taskId, DActSubmit(stitchContext.Size())));
        AutoScopedPerf asp(PERF_EVT_SUBMIT_AICORE);
        PROF_STAGE_BEGIN(PERF_EVT_STAGE_BUILD_TASK, "task.before\n");

        taskContext.ReleaseFinishedTasks(PERF_EVT_RELEASE_FINISH_TASK, PERF_EVT_DEALLOCATE_TASK);

        if (stitchContext.Empty()) {
            PROF_STAGE_END(PERF_EVT_STAGE_BUILD_TASK, "task.after\n");
            return;
        }

        PROF_STAGE_BEGIN(PERF_EVT_DECIDE_SLOT_ADDRESS, "slotaddr.before\n");
        stitchContext.DecideSlotAddress(
            slotContext.GetSlotList(), slotContext.GetSlotSize(), slotContext.GetSlotRefCntPool());
        PROF_STAGE_END(PERF_EVT_DECIDE_SLOT_ADDRESS, "slotaddr.after\n");

        PROF_STAGE_BEGIN(PERF_EVT_DECIDE_INCAST_ADDRESS, "incastaddr.before\n");
        stitchContext.DecideIncastOutcast(taskId);
        PROF_STAGE_END(PERF_EVT_DECIDE_INCAST_ADDRESS, "incastaddr.after\n");

#if DEBUG_SWITCH
        stitchContext.DumpStitchInfo();
#if !DEBUG_INFINITE_LIFETIME
        stitchContext.VerifyStitchedListMemory(*args);
#endif
#endif // DEBUG_SWITCH

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        workspace.MarkAsNewStitchWindow();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

        // Memory recycling
        stitchContext.RecycleAicoreLocalWorkspace();

        DynDeviceTask *dynTask = taskContext.BuildDeviceTaskData(stitchContext, devProg, withoutTail);
        PROF_STAGE_END(PERF_EVT_STAGE_BUILD_TASK, "task.after\n");

        PROF_STAGE_BEGIN(PERF_EVT_STAGE_PUSH_TASK, "push.before\n");
        pushTask(taskId++, &dynTask->devTask, this);
        PROF_STAGE_END(PERF_EVT_STAGE_PUSH_TASK, "push.after\n");

        // Reset stitch context
        stitchContext.Reset();
        slotContext.ClearDirty();
        auto FreeTaskfunc = [this] (DynDeviceTask* task) -> bool {
            if (task->isFinish.load(std::memory_order_relaxed)) {
                workspace.SlabFreeStageAllocMem(task->taskStageAllocMem); // recycle slab alloc memory
                return true;
            }
            return false;
        };

        // try free finished task and recycle aicpu meta memory
        submmitTaskQueue_.FreeUntil(FreeTaskfunc);

        while (!submmitTaskQueue_.TryEnqueue(dynTask)) {
            // maybe que is full, need wait task finish and recycle aicpu meta memory
            submmitTaskQueue_.FreeUntil(FreeTaskfunc);
        }
    }

    schema::RUid GetRuid(uint64_t rootKey, bool afterAppend = false) {
        int64_t dupIndex = stitchContext.Size();
        if (afterAppend) {
            dupIndex -= 1;
        }
        schema::RUid ruid(taskId, dupIndex, rootKey);
        return ruid;
    }

    void *CallRootFunctionAlloc(uint64_t rootKey) {
        DevAscendFunction *devRoot = devProg->GetFunction(rootKey);
        DEV_INFO("prepare one func %p %s.", devRoot, devRoot->GetRawName());
        if (stitchContext.Size() == MAX_CACHED_FUNC_NUM ||
            stitchContext.stitchedCallOpSize() + devRoot->GetOperationSize() > MAX_READY_QUE_ELM_SIZE) {
            SubmitToAicoreAndRecycleMemory(false);
        }
        DEV_TRACE_DEBUG(REvent(GetRuid(rootKey), RActDup(devRoot->GetRawName())));

        PROF_STAGE_BEGIN(PERF_EVT_STAGE_DUP_ROOT, "dup.before\n");
        currDevRootDup = workspace.DuplicateRoot(devRoot);
        PROF_STAGE_END(PERF_EVT_STAGE_DUP_ROOT, "dup.after\n");
        return (void *)&currDevRootDup.GetExpression(0);
    }

    void *CallRootFunctionStitch(uint64_t rootKey) {
        if (rootKey == RUNTIME_FINISH_FUNCKEY) {
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

        slotContext.UpdateSlots(currDevRootDup, taskId, devNextIdx);
        PROF_STAGE_END(PERF_EVT_STAGE_STITCH, "stitch.after\n");
        DEV_TRACE_DEBUG(DEvent(taskId, DActStitchFinish(GetRuid(rootKey, true))));
        return nullptr;
    }

    void TaskFinish(DynDeviceTask *dynTask) {
        dynTask->isFinish.store(true);
    }

    static void TaskFinish(DeviceTask *task, void *ctx_) {
        DeviceExecuteContext *ctx = (DeviceExecuteContext *)ctx_;
        auto *dynTask = (DynDeviceTask *)task;
        ctx->TaskFinish(dynTask);
    }

private:
    static void *DeviceExecuteCallAlloc(void *ctx_, uint64_t rootKey) {
        DeviceExecuteContext *ctx = (DeviceExecuteContext *)ctx_;
        if (ctx == nullptr) {
            DEV_ERROR("invalid ctx.");
            return nullptr;
        }
        PerfBegin(PERF_EVT_ROOT_FUNC);
        void *result = ctx->CallRootFunctionAlloc(rootKey);
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
        void *result = ctx->CallRootFunctionStitch(rootKey);
        PerfEnd(PERF_EVT_ROOT_FUNC);
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
};
} // namespace dynamic
} // namespace npu::tile_fwk
