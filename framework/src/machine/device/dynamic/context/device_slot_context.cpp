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
 * \file device_slot_context.cpp
 * \brief
 */

#include "machine/device/dynamic/context/device_slot_context.h"

namespace npu::tile_fwk::dynamic {

void DeviceSlotContext::InitAllocator(DeviceWorkspaceAllocator &workspace, uint64_t slotSize) {
    workspace.SetupVector(slotList_);
    workspace.SetupItemPool(slotRefCntPool_, slotSize);
    workspace_ = &workspace;
    slotList_.resize(slotSize);
}

void DeviceSlotContext::FillInputOutputSlot(DevAscendProgram *devProg, DevStartArgs *args) {
    FillInputOutputSlot(slotList_.data(), slotList_.size(), devProg, args);
}

static void UpdateSlotsForStitch(int slotIdx, DeviceExecuteSlot &slot, DevAscendFunction *devRootSrc,
                                 DevAscendFunctionOutcast &outcast, uint32_t devTaskId, uint32_t devNextIdx,
                                 uint32_t outcastIndex, uint64_t *expressionList) {
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
static void UpdateSlotsImpl(DeviceWorkspaceAllocator *workspace, DeviceExecuteSlot *slotList,
    const StitchedList &stitchedList, ItemPool<uint32_t, category> &slotRefCntPool,
    DevAscendFunctionDupped &devRootDup, uint32_t devTaskId, uint32_t devNextIdx) {
    AutoScopedPerf asp(PERF_EVT_UPDATE_SLOT);
    DevAscendFunction *devRootSrc = devRootDup.GetSource();
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
    uint64_t *expressionList = &devRootDup.GetExpression(0);
    for (size_t i = 0; i < outcastSize; ++i) {
        auto &srcDesc = devRootDup.GetOutcastAddress(i);
        auto &outcast = devRootSrc->GetOutcast(i);
        for (size_t j = 0; j < outcast.toSlotList.size(); ++j) {
            int slotIdx = devRootSrc->At(outcast.toSlotList, j);
            auto &slot = slotList[slotIdx];
            UpdateSlotsForStitch(slotIdx, slot, devRootSrc, outcast, devTaskId, devNextIdx, i, expressionList);
            if (!slot.RefCntIsNull() && slot.RefCntDec(slotRefCntPool)) {
                // At this moment only old addresses have available refCnt
                DEV_DEBUG_ASSERT(!slot.desc.IsNullAddress());
                uintdevptr_t freeAddr = slot.desc.IsAddress() ? slot.desc.addr :
                    stitchedList[slot.desc.dupIdx].GetOutcastAddress(slot.desc.outcastIdx).GetAddress();
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

void DeviceSlotContext::UpdateSlots(DevAscendFunctionDupped &devRootDup, const StitchedList &stitchedList,
                                    uint32_t devTaskId, uint32_t devNextIdx) {
    UpdateSlotsImpl(workspace_, slotList_.data(), stitchedList, slotRefCntPool_,
        devRootDup, devTaskId, devNextIdx);
}

void DeviceSlotContext::FillInputOutputSlot(DeviceExecuteSlot *slotList, size_t slotSize, DevAscendProgram *devProg,
    DevStartArgs *args) {
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

}