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
#include "device_perf.h"
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
        memcpy_s(reinterpret_cast<void *>(pool.base) , size, data, size);
        __builtin___clear_cache(reinterpret_cast<void *>(pool.base), 
                                reinterpret_cast<uint8_t *>(pool.base) + size
                                );
        PerfEnd(PERF_EVT_CONTROL_FLOW_MAPEXE_MEMCPY);
        code_ = reinterpret_cast<unsigned char *>(pool.base);
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
    void InitAllocator(DeviceWorkspaceAllocator &workspace, uint64_t slotSize);

    void FillInputOutputSlot(DevAscendProgram *devProg, DevStartArgs *args);

    void UpdateSlots(DevAscendFunctionDupped &devRootDup, const StitchedList &stitchedList, uint32_t devTaskId,
                     uint32_t devNextIdx);

    DeviceExecuteSlot *GetSlotList() { return slotList_.data(); }
    size_t GetSlotSize() { return slotList_.size(); }

    auto &GetSlotRefCntPool() { return slotRefCntPool_; }

    void ClearDirty() {
        for (size_t i = 0; i < slotList_.size(); i++) {
            slotList_[i].stitchDupIdx = INVALID_STITCH_IDX;
        }
    }

public:
    void FillInputOutputSlot(DeviceExecuteSlot *slotList, size_t slotSize, DevAscendProgram *devProg,
                             DevStartArgs *args);
    static void UpdateSlotsForStitch(int slotIdx, DeviceExecuteSlot &slot, DevAscendFunction *devRootSrc,
                                     DevAscendFunctionOutcast &outcast, uint32_t devTaskId, uint32_t devNextIdx,
                                     uint32_t outcastIndex, uint64_t *expressionList);
    template <WsMemCategory category>
    static void UpdateSlots(DeviceWorkspaceAllocator *workspace, DeviceExecuteSlot *slotList,
        const StitchedList &stitchedList, int slotSize, ItemPool<uint32_t, category> &slotRefCntPool,
        DevAscendFunctionDupped &devRootDup, uint32_t devTaskId, uint32_t devNextIdx);
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

    void Init(DevAscendProgram *devProg, DeviceWorkspaceAllocator &workspace);
    void Reset();

    void DumpStitchInfo();

    size_t Size() const { return stitchedList_.size(); }
    bool Empty() const { return stitchedList_.empty(); }
    
    void Append(DevAscendFunctionDupped &devRootDup) { stitchedList_.push_back(devRootDup); }

    const auto &GetStitchedList() const { return stitchedList_; }

    static void CheckStitch(DevAscendFunctionDupped *stitchedList, int size, DevAscendFunctionDupped *nextDup);

    static void CheckStitch(DynDeviceTask *dyntask);

    uint64_t Stitch(DeviceSlotContext &slotContext, DevAscendFunctionDupped &nextDup, size_t devTaskId,
                    size_t devNextIdx);

    void RecycleTensorWorkspace();

    void DumpSlotInfo(const char *label, DeviceExecuteSlot *slotList, size_t slotSize);

    void DecideSlotAddress(DeviceExecuteSlot *slotList, size_t slotSize,
                           ItemPool<uint32_t, WsMemCategory::ITEMPOOL_SLOT_REF_CNT> &slotRefCntPool);

    void DecideIncastOutcast(uint64_t taskId);

    void MoveTo(DynDeviceTask *dynTask);

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

    static void HandleOneStitch(
            DevAscendFunctionDupped &producerDup, DevAscendFunctionDupped &consumerDup,
            DevAscendFunctionDuppedStitchList &producerStitchList, size_t producerOperationIdx,
            size_t consumerIdx, size_t consumerOperationIdx, DeviceWorkspaceAllocator *workspace,
            StitchKind debugStitchKind, int debugSlotIdx);

    static void HandleOneStitch(
            DevAscendFunctionDupped &producerDup, DevAscendFunctionDupped &consumerDup,
            size_t producerOperationIdx, size_t consumerIdx, size_t consumerOperationIdx,
            DeviceWorkspaceAllocator *workspace, StitchKind debugStitchKind, int debugSlotIdx);

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
        DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionIncast& incast);

    uint64_t FullCoverDefaultUpdateStitch(DevAscendFunctionDupped &nextDup, size_t devNextIdx, DeviceExecuteSlot& slot,
        int slotIdx, DevAscendFunctionIncast& incast);

    uint64_t FullCoverUpdateStitch(DevAscendFunctionDupped &nextDup, size_t devNextIdx, DeviceExecuteSlot& slot,
        int slotIdx, DevAscendFunctionIncast& incast);

    void ReuseStitch(DevAscendFunctionDupped &nextDup, size_t devNextIdx);

    uint64_t FastStitch(DeviceExecuteSlot *slotList, size_t slotSize, DevAscendFunctionDupped &nextDup,
        size_t devTaskId, size_t devNextIdx);

    static void DumpStitchInfo(DevAscendFunctionDupped *stitchedList, int stitchedSize);

private:
    static
    bool MemOverlap(uint64_t ahead, uint64_t alength, uint64_t bhead, uint64_t blength) {
        return !(ahead + alength <= bhead || bhead + blength <= ahead);
    }

    static void StitchForWorkspaceReuse(DevAscendFunctionDupped *stitchingList, int stitchingSize,
        DevAscendFunctionDupped &prevDup, DevAscendFunctionDupped &currDup, size_t devCurrIdx,
        DeviceWorkspaceAllocator *workspace);
};

const uint32_t OP_ATTRS_PRE_NUM = 8;
const uint32_t OP_ATTRS_OFFSET_PRE_NUM = 4;
const uint32_t EXPR_TABLE_PRE_NUM = 8;
const uint32_t RAW_TENSOR_ADDR_MASK = 8;
const uint32_t CCE_BINARY_MOD = 8;
const size_t DUP_PRED_COUNT_LOOP_MAX = 8;
const size_t DUP_PRED_COUNT_PRE_LOOP_CNT = 4;
struct DeviceTaskContext {
    void InitAllocator(DevAscendProgram *devProg, DeviceWorkspaceAllocator &workspace,
                       npu::tile_fwk::DevStartArgsBase *startArgs);

    DynDeviceTask *BuildDeviceTaskData(DeviceStitchContext &stitchContext, uint32_t taskId, DevAscendProgram *devProg,
                                       bool withoutTail);

    void ReleaseFinishedTasks(int perfEvtReleaseFinishTask, int perfEvtDeallocateTask);

    void AppendFinishTask(DynDeviceTask *dynTask);

    void ShowStats();

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
    void BuildReadyQueue(DynDeviceTask *dyntask, DevAscendProgram *devProg);

    void BuildDynFuncData(DynDeviceTask *dyntask, uint32_t taskId, DevAscendProgram *devProg,
        DevAscendFunctionDupped *stitchedList, uint64_t stitchedSize);

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

    void ResolveEarlyDepends(DynDeviceTask *dyntask, size_t funcIdx, size_t opIdx);

    void ResolveEarlyDepends(DynDeviceTask *dyntask);

public:
    static void DumpReadyQueue(DynDeviceTask *dynTask, const char *prefix);

    static void DumpDepend(DynDeviceTask *dyntask, DevAscendProgram *devProg, DevStartArgs *startArgs, const char *prefix);

    void BuildDeviceTaskDataAndReadyQueue(DynDeviceTask *dyntask, uint32_t taskId, DevAscendProgram *devProg);
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

    bool DuppedRootCached();

    bool DuppedRootUpdateAndCachedAllSubmitted();

    static uint64_t GetInputShapeDimSize(DeviceExecuteContext *ctx, uint64_t inputIndex);
    static uint64_t GetInputShapeDim(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t n);
    static int64_t GetInputDataInt32Dim1(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0);
    static int64_t GetInputDataInt32Dim2(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1);
    static int64_t GetInputDataInt32Dim3(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1,
        uint64_t off2);
    static int64_t GetInputDataInt32Dim4(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1,
        uint64_t off2, uint64_t off3);

    static void *SymbolHandlerIdToHandler(SymbolHandlerId id);

    DeviceExecuteContext(DevStartArgs *startArgs);

    void ShowStats();

    void RunInit(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    void PushTask(DynDeviceTask *dynTask);

    void GELaunchRunCached(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    void RunControlFlow(DevStartArgs *startArgs);

    void GELaunchFullCacheRunControlFlow(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    void GELaunchFullCache(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    void GELaunchPartialCache(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    void GELaunch(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    bool AiCoreFree();

    static void DumpDeviceTask(uint64_t taskId, DynDeviceTask *deviceTask);

    void SubmitToAicoreAndRecycleMemory(bool withoutTail, bool isLastTask = false);

    schema::RUid GetRuid(uint64_t rootKey, bool afterAppend = false);

    void ControlFlowCacheStopCache(uint64_t rootKey);

    void *CallRootFunctionAlloc(uint64_t rootKey);

    void *CallRootFunctionStitch(uint64_t rootKey);

private:
    static void *DeviceExecuteCallAlloc(void *ctx_, uint64_t rootKey);

    static void *DeviceExecuteCallStitch(void *ctx_, uint64_t rootKey);

    static void *DeviceExecuteRuntimerLog(void *ctx_, uint64_t value);

    static void *DeviceExecuteShmemAlloctor(void *ctx_, uint64_t value);
};
} // namespace dynamic
} // namespace npu::tile_fwk
