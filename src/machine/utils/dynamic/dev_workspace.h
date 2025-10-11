
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
 * \file dev_workspace.h
 * \brief
 */

#ifndef DEV_WORKSPACE_H
#define DEV_WORKSPACE_H

#include "dev_encode.h"
#include "item_pool.h"
#include "spsc_queue.h"
#include "../machine_ws_intf.h"
#include "allocator/allocators.h"

#ifndef __DEVICE__
#include "interface/configs/config_manager.h"
#endif

namespace npu::tile_fwk::dynamic {

const size_t MAX_CACHED_FUNC_NUM = 128;
const size_t MAX_READY_QUE_ELM_SIZE = 20000;

struct DynFuncCacheItem {
    DevAscendFunction *devFunc;
    predcount_t *predCount;
    int *calleList;
    DevAscendFunctionDupped dup;
};
struct WsSlabStageAllocMem {
    std::atomic_bool canFree{false};
    StageAllocInfo aicpuCoherentStageMem;
    StageAllocInfo aicpuStitchStageMem;

    WsSlabStageAllocMem() = default;
    WsSlabStageAllocMem(const WsSlabStageAllocMem& other)
        : canFree(other.canFree.load(std::memory_order_relaxed)),
          aicpuCoherentStageMem(other.aicpuCoherentStageMem),
          aicpuStitchStageMem(other.aicpuStitchStageMem) {}

    WsSlabStageAllocMem& operator=(const WsSlabStageAllocMem& other) {
        if (this != &other) {
            canFree.store(other.canFree.load(std::memory_order_relaxed), 
                         std::memory_order_relaxed);
            aicpuCoherentStageMem = other.aicpuCoherentStageMem;
            aicpuStitchStageMem = other.aicpuStitchStageMem;
        }
        return *this;
    }
};

class DeviceWorkspaceAllocator;
constexpr size_t READY_QUEUE_SIZE = 3UL;
struct DynDeviceTask {
    DeviceTask devTask;
    DynFuncHeader* dynFuncData{nullptr};

    ReadyCoreFunctionQueue *readyQueue[READY_QUEUE_SIZE];
    DynFuncCacheItem cacheList[MAX_CACHED_FUNC_NUM];
    Vector<DevAscendFunctionDupped, WsMemCategory::VECTOR_STITCHED_LIST, DeviceWorkspaceAllocator> stitchedList;
    const DevCceBinary *cceBinary;
    const DevAicpuLeafBinary *aicpuLeafBinary;
    WsAllocation selfAlloc;
    WsSlabStageAllocMem taskStageAllocMem;

    uint32_t GetReadyQueueIndexByCoreType(CoreType coreType) {
        if (coreType == CoreType::AICPU) {
            return static_cast<uint32_t>(READY_QUEUE_SIZE) - 1;
        }
        return static_cast<uint32_t>(coreType);
    }

    DynDeviceTask(DeviceWorkspaceAllocator &allocator) {
        memset_s(&devTask, sizeof(devTask), 0, sizeof(devTask));
        stitchedList.InitAllocator(allocator);
    }

    predcount_t &GetOperationCurrPredCount(uint32_t id) {
        return stitchedList[FuncID(id)].GetOperationCurrPredCount(TaskID(id));
    }

    int GetOperationCoreType(uint32_t id) {
        auto callee = stitchedList[FuncID(id)].GetSource()->GetOperationAttrCalleeIndex(TaskID(id));
        return cceBinary[callee].coreType;
    }

    std::string DumpTaskData(uint32_t id) {
        auto &funcDup = stitchedList[FuncID(id)];
        return funcDup.DumpDyn(FuncID(id), TaskID(id), cceBinary);
    }

    void DumpTopo() {
        auto header = dynFuncData;
#ifdef __DEVICE__
        std::string path = "./output/dyn_topo.txt";
#else
        std::string path = config::LogTopFolder() + "/dyn_topo.txt";
#endif
        static std::ofstream of(path);
        if (of.tellp() == 0) {
            of << "seqNo,taskId,rootIndex,rootHash,opmagic,leafIndex,leafHash,coreType,psgId,successors\n";
        }
        for (size_t funcIdx = 0; funcIdx < stitchedList.size(); funcIdx++) {
            stitchedList[funcIdx].DumpTopo(of, header->seqNo, funcIdx, cceBinary);
        }
        of.flush();
    }

    void DumpLeafs() {
        for (size_t funcIdx = 0; funcIdx < stitchedList.size(); funcIdx++) {
            auto lines = stitchedList[funcIdx].DumpLeafs(dynFuncData->seqNo, funcIdx);
            for (auto &&line : lines) {
                DEV_ERROR("[DumpLeafs] %s", line.c_str());
            }
        }
    }

#if DEBUG_INFINITE_LIFETIME
    void DumpTensorAddrInfo(uintdevptr_t dumpTensorWsAddr, uint64_t dumpTensorWsSize) {
        UNUSED(dumpTensorWsAddr);
        UNUSED(dumpTensorWsSize);
        std::stringstream oss;
        std::vector<std::string> infos;
        for (uint32_t funcIdx = 0; funcIdx < stitchedList.size(); funcIdx++) {
            stitchedList[funcIdx].DumpTensorAddrInfo(infos, dynFuncData->seqNo, funcIdx);
        }
        auto str = std::move(oss).str();
        DEV_ERROR("[DumpTensor] seqNo,taskId,rawMagic,address,dtype,bytesOfDtype,(shapes,)");
        DEV_ERROR("[DumpTensor] >>>");
        for (auto &info : infos) {
            DEV_ERROR("[DumpTensor] %s", info.c_str());
        }
        DEV_ERROR("[DumpTensor] <<<");
    }
#endif
};

#define INVALID_STITCH_IDX      ((uint32_t)-1)

struct DeviceExecuteSlot {
    AddressDescriptor desc;
    bool isOutputSlot{false};
    bool isAssembleSlot{false};
    bool isPartialUpdateStitch{false};
    bool isPartialUpdateDirty{false};
    uint32_t *refCnt{nullptr}; // refCnt to stored tensor
    uint32_t stitchDupIdx{INVALID_STITCH_IDX};
    uint32_t stitchOutcastIdx;

    DevAscendProgramPartialUpdate *partialUpdate{nullptr};
    bool IsFixedAddress() const {
        return isOutputSlot || isAssembleSlot;
    }

    template <WsMemCategory category>
    bool DerefAndCheckIfZeroRefCnt(ItemPool<uint32_t, category> &pool) {
        DEV_DEBUG_ASSERT(refCnt != nullptr);
        --*refCnt;
        if (*refCnt == 0) {
            pool.Destroy(refCnt);
            refCnt = nullptr;
            return true;
        }
        return false;
    }
};

const uint32_t SUBMMIT_TASK_QUE_SIZE = 32;
class DeviceWorkspaceAllocator {
public:
    DeviceWorkspaceAllocator() = default;
    ~DeviceWorkspaceAllocator() = default;

    void Init(DevStartArgs *args) {
        uintdevptr_t baseAddr = args->workspaceAddr;

        // Host coherent allocators MUST be initialized EARLIEST since some other allocators might depend on them
        InitHostCoherentAllocators(baseAddr, args->aicpuCoherentWorkspaceSize, args->devProg);
        baseAddr += args->aicpuCoherentWorkspaceSize;

        InitCoreLocalAllocators(baseAddr, args->aicoreLocalWorkspaceSize, args->devProg);
        baseAddr += args->aicoreLocalWorkspaceSize;

#if DEBUG_INFINITE_LIFETIME
        dumpTensorWsAllocator_.InitAicoreLocal(baseAddr, args->devProg->debugDumpTensorMemReq);
        DEV_DEBUG("[DumpTensor] dumpTensorWsAllocator_: ptr=0x%lx, size=%lu",
                  baseAddr, args->devProg->debugDumpTensorMemReq);
        baseAddr += args->devProg->debugDumpTensorMemReq;
        dumpTensorWsAllocatorCounter_ = dumpTensorWsAllocator_.Allocate<uint64_t>(1).As<uint64_t>();
        *dumpTensorWsAllocatorCounter_ = dumpTensorWsAllocator_.AllocatedSize();
#endif
        SetupVector(slotMemToBeFree_);
        slotMemToBeFree_.reserve(args->devProg->slotPoolSize);

        standardRootWorkspace_ = args->devProg->rootFuncStandardMemReq;
        devProg_ = args->devProg;
    }

    uintdevptr_t StackWorkspaceAddr() const { return stackWorkspaceBase_; }
    uint64_t StandardStackWorkspacePerCore() const { return standardStackWorkspacePerCore_; }

#if DEBUG_INFINITE_LIFETIME
    uintdevptr_t DumpTensorWsBaseAddr() const { return dumpTensorWsAllocator_.MemBaseAddr(); }
    uint64_t DumpTensorWsSize() const { return dumpTensorWsAllocator_.Capacity(); }
#endif
    template <typename T, WsMemCategory category, typename WsAllocator_T>
    void SetupVector(Vector<T, category, WsAllocator_T> &vector) {
        if constexpr (std::is_same_v<WsAllocator_T, npu::tile_fwk::dynamic::DeviceWorkspaceAllocator>) {
            vector.InitAllocator((*this));
        } else {
            vector.InitAllocator(aicpuCoherentAllocator_);
        }
    }

    template <typename T, WsMemCategory category>
    void SetupItemPool(ItemPool<T, category> &pool, size_t count) {
        pool.Init(aicpuCoherentAllocator_, count);
    }

    WsMemoryState VerifyAicoreLocalMemoryState(uintdevptr_t ptr, size_t size) const {
        return aicoreLocalWsVerifier_.Verify(ptr, size);
    }

    void VerifyStitchedListMemory(DevStartArgs &args, const DevAscendFunctionDupped *stitchedList, size_t size) {
        struct MemoryInfo {
            uintdevptr_t ptr;
            size_t size;
            DevAscendFunctionDupped dup;
            size_t stitchedListIndex;
            size_t rawIndex;

            void DumpError() const {
                std::string ioPropertyDump;
                switch (dup.GetSource()->GetRawTensor(rawIndex)->ioProperty) {
                    case DevIOProperty::ROOT_INCAST:
                        ioPropertyDump = " (Root Incast)";
                        break;
                    case DevIOProperty::ROOT_OUTCAST:
                        ioPropertyDump = " (Root Outcast)";
                        break;
                    default:
                        break;
                }
                DEV_ERROR("  Func (%2zu) %16s rawTensor[%2zu], @%" PRIx64 " [%zu bytes]%s.",
                    stitchedListIndex, dup.GetSource()->GetRawName(), rawIndex, ptr, size,
                    ioPropertyDump.c_str());
            }
        };

        bool verificationSuccess = true;

        std::set<uintdevptr_t> inoutAddr;
        for (int i = 0; i < args.GetInputTensorSize(); i++) {
            inoutAddr.insert(args.GetInputTensor(i).address);
        }
        for (int i = 0; i < args.GetOutputTensorSize(); i++) {
            inoutAddr.insert(args.GetOutputTensor(i).address);
        }

        for (size_t i = 0; i < size; i++) {
            const auto &dup = stitchedList[i];

            auto isValidWsTensor = [&](uintdevptr_t ptr, size_t memSize) {
                return slotVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
                    globalTensorVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
                    funcWsVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE ||
                    outcastWsVerifier_.Verify(ptr, memSize) == WsMemoryState::INSIDE;
            };

            size_t rawTensorCount = dup.GetSource()->GetRawTensorSize();
            for (size_t j = 0; j < rawTensorCount; j++) {
                auto *rawTensor = dup.GetSource()->GetRawTensor(j);
                auto memReq = rawTensor->GetMemoryRequirement(dup.GetExpressionAddr());
                MemoryInfo memInfo {
                    dup.GetRawTensorAddr(j),
                    // For workspace tensors, the memoryRequirement property is deprecated
                    rawTensor->ioProperty == DevIOProperty::NONE ? 0 : memReq,
                    dup,
                    i,
                    j,
                };
                switch (VerifyAicoreLocalMemoryState(memInfo.ptr, memInfo.size)) {
                    case WsMemoryState::INSIDE:
                        if (!isValidWsTensor(memInfo.ptr, memInfo.size)) {
                            DEV_ERROR("Invalid workspace tensor (not completely inside any workspace segment):");
                            memInfo.DumpError();
                            verificationSuccess = false;
                        }
                        break;
                    case WsMemoryState::CROSS_BOUNDARY:
                        DEV_ERROR("Memory crossing workspace boundary:");
                        memInfo.DumpError();
                        verificationSuccess = false;
                        break;
                    default:
                        if (!inoutAddr.count(memInfo.ptr)) {
                            DEV_ERROR("Non input/output tensor outside of workspace:");
                            memInfo.DumpError();
                            verificationSuccess = false;
                        }
                        break;
                }
            }
        }

        DEV_ASSERT(verificationSuccess);
    }

private:
    void ResetFuncWsAllocator() {
        aicoreLocalFuncWsAllocator_.ResetPool();
        funcWsAllocationInfo_.ptr = 0;
        funcWsAllocationInfo_.blockIdx = 0;
    }

    bool TryAllocateFuncWs(DevAscendFunctionDupped dup, uint64_t size, WsAllocatorCounter *dfxCounter = nullptr) {
        if (funcWsAllocationInfo_.ptr == 0 || funcWsAllocationInfo_.allocated + size > standardRootWorkspace_) {
            if (!aicoreLocalFuncWsAllocator_.CanAllocate(standardRootWorkspace_)) {
                ResetFuncWsAllocator();
                DEV_DEBUG_ASSERT(aicoreLocalFuncWsAllocator_.CanAllocate(standardRootWorkspace_));
            } else if (funcWsAllocationInfo_.ptr != 0) {
                funcWsAllocationInfo_.blockIdx++;
            }
            WsAllocation allocation = aicoreLocalFuncWsAllocator_.Malloc(
                standardRootWorkspace_, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
            if (dfxCounter) {
                dfxCounter->LogMalloc(allocation);
            }
            funcWsAllocationInfo_.ptr = allocation.ptr;
            funcWsAllocationInfo_.allocated = 0;
        }
        dup.RuntimeWorkspace() = funcWsAllocationInfo_.ptr + funcWsAllocationInfo_.allocated;
        auto &reuseInfo = dup.GetRuntimeReuseInfo();
        reuseInfo.poolResetTimes = aicoreLocalFuncWsAllocator_.ResetTimes();
        reuseInfo.blockIdx = funcWsAllocationInfo_.blockIdx;
        funcWsAllocationInfo_.allocated += size;
        return true;
    }

    struct {
        uintdevptr_t ptr = 0;
        uint64_t allocated = 0;
        uint32_t blockIdx = 0;
    } funcWsAllocationInfo_;

public:
#if DEBUG_INFINITE_LIFETIME
    WsAllocation DebugDumpTensorAllocate(size_t memReq,
        WsMemCategory category = WsMemCategory::UNCLASSIFIED) {
        DEV_ASSERT(dumpTensorWsAllocator_.CanAllocate(memReq));
        WsAllocation allocation = dumpTensorWsAllocator_.Malloc(memReq, category);
        *dumpTensorWsAllocatorCounter_ = dumpTensorWsAllocator_.AllocatedSize();
        return allocation;
    }
#endif
    bool TryAllocateFunctionMemory(DevAscendFunctionDupped devRootDup, DeviceExecuteSlot *slotList) {
        AutoScopedPerf asp(PERF_EVT_ALLOCATE_WORKSPACE);

        WsAllocatorCounter *pDfxCounter = nullptr;
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        WsAllocatorCounter funcAllocDfx;
        pDfxCounter = &funcAllocDfx;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

        DevAscendFunction *devRootSrc = devRootDup.GetSource();

        // alloc outcast workspace
        size_t outcastSize = devRootSrc->outcastWsMemoryRequirement;
        if (outcastSize != 0) {
            if (!aicoreLocalFuncOutcastAllocator_.CanAllocate(outcastSize)) {
                return false;
            }

            WsAllocation allocation = aicoreLocalFuncOutcastAllocator_.Malloc(
                outcastSize, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
            funcAllocDfx.LogMalloc(allocation);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
#if DEBUG_INFINITE_LIFETIME
        allocation = DebugDumpTensorAllocate(outcastSize, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
#endif
            devRootDup.RuntimeOutcastBase() = allocation.ptr;
        } else {
            devRootDup.RuntimeOutcastBase() = 0;
        }

        // alloc inner workspace
        size_t wsSize = devRootSrc->rawTensorWsMemoryRequirement;
        if (wsSize != 0) {
            if (!TryAllocateFuncWs(devRootDup, wsSize, pDfxCounter)) {
                return false;
            }
#if DEBUG_INFINITE_LIFETIME
        WsAllocation allocation = DebugDumpTensorAllocate(wsSize, WsMemCategory::TENSOR_ROOTFUNC_INTERNAL);
        devRootDup.RuntimeWorkspace() = allocation.ptr;
#endif
        } else {
            devRootDup.RuntimeWorkspace() = 0;
        }

        // assign incast address descriptor
        for (size_t i = 0; i < devRootSrc->GetIncastSize(); ++i) {
            DEV_DEBUG_ASSERT(devRootSrc->GetIncast(i).fromSlotList.size() > 0);

            int slotIndex = devRootSrc->At(devRootSrc->GetIncast(i).fromSlotList, 0);
            devRootDup.GetIncastAddress(i) = slotList[slotIndex].desc;
            DEV_DEBUG("get incast %zu, from slot %d address %s.", i, slotIndex, devRootDup.GetIncastAddress(i).Dump().c_str());
        }

        // assign outcast address separately first, will be reassigned when corresponding slot was replaced
        uintdevptr_t outcastBaseAddr = devRootDup.RuntimeOutcastBase();
        for (size_t i = 0; i < devRootSrc->GetOutcastSize(); ++i) {
            int slotIndex = -1;
            auto &toSlotList = devRootSrc->GetOutcast(i).toSlotList;
            for (size_t k = 0; k < toSlotList.size(); ++k) {
                auto idx = devRootSrc->At(toSlotList, k);
                if (slotList[idx].IsFixedAddress() || slotList[idx].isPartialUpdateStitch) { // true表示固定地址，用户输出/Assemble的结果
                    slotIndex = idx;
                    break;
                }
            }
            AddressDescriptor desc;
            if (slotIndex != -1) {
                desc = slotList[slotIndex].desc;
                if (desc.IsNullAddress()) {
                    auto rawTensor = devRootSrc->GetOutcastRawTensor(i);
                    if (rawTensor->linkedIncastId == -1) {
                        auto memReq = rawTensor->GetMemoryRequirement(devRootDup.GetExpressionAddr());
                        auto allocation = aicoreGlobalAllocator_.Allocate<uint8_t>(memReq);
#if DEBUG_INFINITE_LIFETIME
                        allocation = DebugDumpTensorAllocate(memReq);
#endif
                        desc = AddressDescriptor(allocation.ptr);
                        slotList[slotIndex].desc = desc;
                    }
                }
            } else if (devRootSrc->GetOutcast(i).exprListIndex != -1) {
                uint64_t *exprTbl = devRootDup.GetExpressionAddr();
                uint64_t addr = exprTbl[devRootSrc->GetOutcast(i).exprListIndex];
                desc = AddressDescriptor(addr);
            } else {
                desc = AddressDescriptor(outcastBaseAddr + devRootSrc->GetOutcastRawTensor(i)->addrOffset);
            }

            //判断是否与incast 共地址
            auto rawTensor = devRootSrc->GetOutcastRawTensor(i);
            if (rawTensor->linkedIncastId != -1) {
                desc = devRootDup.GetIncastAddress(rawTensor->linkedIncastId);
            }

            devRootDup.GetOutcastAddress(i) = desc;
            DEV_DEBUG("get outcast %zu slot %d address %s.", i, slotIndex, desc.Dump().c_str());
        }

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        funcAllocDfx.DelayedDumpAsRootFuncAndReset(wsMemDelayedDumper_, devRootDup.GetSource()->GetRawName());
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        return true;
    }

    bool IsValidSlotMemRequirement(uint64_t memReq) const {
        return aicoreLocalSlotAllocator_.IsValidSlotMemRequirement(memReq);
    }

    uintdevptr_t AllocateSlot(const char *rootFuncName) {
        WsAllocation allocation = aicoreLocalSlotAllocator_.Allocate();
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        wsMemDelayedDumper_.LogTensorMalloc(rootFuncName, allocation);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        (void)rootFuncName;
        return allocation.ptr;
    }

    void DelayedRecycleSlotMem(uintdevptr_t ptr) {
        slotMemToBeFree_.push_back(ptr);
    }

    void TriggerDelayedRecycle() {
        for (uintdevptr_t ptr : slotMemToBeFree_) {
            aicoreLocalSlotAllocator_.Deallocate(ptr);
        }
        slotMemToBeFree_.clear();
    }

    void RecycleDevFuncWorkspace() {
        aicoreLocalFuncOutcastAllocator_.ResetPool();
        ResetFuncWsAllocator();
    }

    DevAscendFunctionDupped DuplicateRoot(DevAscendFunction *func) {
        WsAllocation tinyAlloc = SlabAlloc(func->GetDuppedDataAllocSize(), WsAicpuSlabMemType::DUPPED_FUNC_DATA);
        return DevAscendFunctionDupped::DuplicateRoot(func, tinyAlloc);
    }

    void DestroyDuppedFunc(DevAscendFunctionDupped &dup) {
        dup.ReleaseDuppedMemory(aicpuCoherentAllocator_);
    }

    DynDeviceTask *MakeDynDeviceTask() {
        WsAllocation alloc = SlabAlloc(sizeof(DynDeviceTask), WsAicpuSlabMemType::DEV_DYN_TASK);
        DynDeviceTask *dynTask = new((void *)alloc.ptr) DynDeviceTask(*this);
        dynTask->selfAlloc = alloc;
        return dynTask;
    }

    DevAscendFunctionDuppedStitch *AllocateStitch() {
        WsAllocation allocation = SlabAlloc(sizeof(DevAscendFunctionDuppedStitch), WsAicpuSlabMemType::DUPPED_STITCH);
        DevAscendFunctionDuppedStitch *stitch = allocation.As<DevAscendFunctionDuppedStitch>();
        uint64_t *clear = PtrToPtr<DevAscendFunctionDuppedStitch, uint64_t>(stitch);
        clear[0] = 0;
        clear[1] = 0;
        return stitch;
    }

    void ResetAicpuMemCounter() {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        aicpuCoherentAllocator_.ResetCounter();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    }

    void LogAicpuAlloc(DevAscendFunctionDupped dup) {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        dup.LogAicpuAlloc(aicpuCoherentAllocator_);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        (void)dup;
    }

    void RewindMemoryDumper() {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        wsMemDelayedDumper_.Rewind();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    }

    void MarkAsNewStitchWindow() {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        aicpuCoherentAllocator_.DelayedDumpAndResetCounter(wsMemDelayedDumper_);
        aicpuStitchAllocator_.DelayedDumpAndResetCounter(wsMemDelayedDumper_);
        wsMemDelayedDumper_.MarkAsNewStitchWindow();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    }

    void DumpMemoryUsage(const char *hint) const {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
        wsMemDelayedDumper_.DumpStitchWindowMemoryUsage();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        aicpuCoherentAllocator_.DumpMemoryUsage(hint, "Metadata");
        aicpuMetaSlabAllocator_.DumpMemoryUsage(hint, "Metadata slab allocator");
        aicpuStitchSlabAllocator_.DumpMemoryUsage(hint, "Metadata Stitch slab allocator");
        aicoreLocalFuncWsAllocator_.DumpMemoryUsage(hint, "Tensor (inner) workspace");
        aicoreLocalFuncOutcastAllocator_.DumpMemoryUsage(hint, "Tensor (outcast) workspace");
        aicoreGlobalAllocator_.DumpMemoryUsage(hint, "Tensor (global) workspace");
        aicoreLocalSlotAllocator_.DumpMemoryUsage(hint);

        // Dump stack memory
        DEV_MEM_DUMP("Stack workspace memory usage (%s)\n", hint);
        DEV_MEM_DUMP("            Memory pool size: %10lu bytes\n", stackWorkspaceSize_);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
    }

    void InitAicpuMetaSlabAllocator() {
        DEV_ASSERT(aicpuCoherentAllocator_.FreeMemorySize() > 0);
        uint64_t memBase = aicpuCoherentAllocator_.MemBaseAddr() + aicpuCoherentAllocator_.AllocatedSize();
        uint64_t realMemBase = AlignUp(memBase, sizeof(uint64_t));
        uint32_t metaSlabMemSize = aicpuCoherentAllocator_.FreeMemorySize() - (realMemBase - memBase);
        uint32_t slabSize = CalcAicpuMetaSlabAlloctorSlabPageSize(metaSlabMemSize);
        aicpuMetaSlabAllocator_.Init(reinterpret_cast<void*>(realMemBase), metaSlabMemSize, slabSize);
        for (size_t i = 0; i < ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT); i++) {
            if (slabMemObjSizeFunc[i] != nullptr) {
                DEV_ASSERT(aicpuMetaSlabAllocator_.RegistCache(i, (this->*slabMemObjSizeFunc[i])()));
            }
        }
    }

    WsAllocation SlabAlloc(uint32_t objSize, WsAicpuSlabMemType type) {
        void* ptr = nullptr;
        DEV_DEBUG("SlabAlloc type = %u, size = %u.", ToUnderlying(type), objSize);
        SlabTryDynAddCache(type, objSize); // ready que need dyn add cache
        do {
            if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
                ptr = aicpuMetaSlabAllocator_.Alloc(ToUnderlying(type));
            } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
                ptr = aicpuStitchSlabAllocator_.Alloc(ToUnderlying(type));
            }
            if (ptr != nullptr) {
                break;
            }

            if (submmitTaskSlabMemQueue_.IsEmpty()) {
                // should not happen, first task alloc failed
                aicpuMetaSlabAllocator_.DumpMemoryStatusWhenAbnormal("SlabAlloc null");
                aicpuStitchSlabAllocator_.DumpMemoryStatusWhenAbnormal("SlabAlloc null");
                DEV_ASSERT_MSG(false, "Slab alloc null,type=%u,objsize=%u.", ToUnderlying(type), objSize);
            }
            uint32_t ttl = 0;
            uint32_t ttlTimout = 100000;
            while (!SlabStageAllocMemTryRecycle()) {  // wait sch aicpu finish task
                ttl++;
                if (ttl > ttlTimout) {
                    DEV_WARN("Waiting for device task memory reclamation for too long.");
                }
            };
        } while (true);

        WsAllocation allocation;
        allocation.ptr = reinterpret_cast<uintdevptr_t>(ptr);
        allocation.node_ = reinterpret_cast<void *>(0xDEADBEEFDEADBEFF);
        return allocation;
    }

    WsSlabStageAllocMem SlabGetStageAllocMem(bool keepTail, WsAicpuSlabMemType keepType) {
        WsSlabStageAllocMem stageMem;
        stageMem.aicpuCoherentStageMem = aicpuMetaSlabAllocator_.PopStageAllocMem(keepTail, ToUnderlying(keepType));
        stageMem.aicpuStitchStageMem = aicpuStitchSlabAllocator_.PopStageAllocMem(false, 0); // not support keep alloc memory
        return stageMem;
    }

    void SlabStageAllocMemSubmmit(WsSlabStageAllocMem* submmitSlabMem) {
        while (!submmitTaskSlabMemQueue_.TryEnqueue(submmitSlabMem)) {
            // maybe que is full, need wait task finish and recycle aicpu meta memory
            SlabStageAllocMemTryRecycle();
        }
        return;
    }

    /* support vector allocator,so need have this fucntion member */
    template <typename T>
    WsAllocation Allocate(uint64_t count, WsMemCategory category) {
        DEV_ASSERT(category == WsMemCategory::VECTOR_STITCHED_LIST);
        return SlabAlloc(count * sizeof(T), WsAicpuSlabMemType::VEC_STITCHED_LIST);
    }

    void Deallocate(WsAllocation) {} // just for support vector allocator,so need have this fucntion member
private:
    void InitHostCoherentAllocators(uintdevptr_t workspaceAddr,
                                    uint64_t aicpuCoherentWorkspaceSize,
                                    DevAscendProgram *devProg) {
        // Stitch pool is part of aicpu coherent workspace and cannot extend the total size
        DEV_ASSERT(devProg->stitchPoolSize <= aicpuCoherentWorkspaceSize);

        // Initialize aicpu memory
        uint64_t baseAddr = workspaceAddr;

        aicpuCoherentAllocator_.InitAicpuCoherent(baseAddr, aicpuCoherentWorkspaceSize - devProg->stitchPoolSize);
        baseAddr += aicpuCoherentWorkspaceSize - devProg->stitchPoolSize;
        InitAicpuStitchSlabAllocator(reinterpret_cast<void*>(baseAddr), devProg->stitchPoolSize);
        baseAddr += devProg->stitchPoolSize;

        DEV_ASSERT(workspaceAddr <= baseAddr && baseAddr <= workspaceAddr + aicpuCoherentWorkspaceSize);
    }

    void InitCoreLocalAllocators(uintdevptr_t workspaceAddr,
                                 uint64_t aicoreLocalWorkspaceSize,
                                 DevAscendProgram *devProg) {
        uint64_t baseAddr = workspaceAddr;

        // Initialize in-core stack memory
        stackWorkspaceBase_ = baseAddr;
        standardStackWorkspacePerCore_ = devProg->standardStackWorkspacePerCore;
        stackWorkspaceSize_ = devProg->standardStackWorkspacePerCore * devProg->devArgs.GetBlockNum();
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceSpill(
            mem(devProg->standardStackWorkspacePerCore), devProg->devArgs.GetBlockNum(),
            range(stackWorkspaceBase_, stackWorkspaceBase_ + stackWorkspaceSize_))));

        baseAddr += stackWorkspaceSize_;

        // Initialize aicore workspace memory verifier
        uint64_t aicoreLocalWorkspaceWithoutStack = aicoreLocalWorkspaceSize - stackWorkspaceSize_;
        aicoreLocalWsVerifier_.Init(
            baseAddr,
            aicoreLocalWorkspaceWithoutStack);

        // Initialize aicore slot tensor memory
        auto aicoreLocalSlotAllocatorSize = devProg->slotPoolSize * devProg->slotStandardMemReq;
        slotVerifier_.Init(baseAddr, aicoreLocalSlotAllocatorSize);
        aicoreLocalSlotAllocator_.InitAicoreLocal(
            baseAddr,
            devProg->slotPoolSize,
            devProg->slotStandardMemReq,
            aicpuCoherentAllocator_);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceCrossDeviceTaskOutcast(range(baseAddr, baseAddr + aicoreLocalSlotAllocatorSize))));
        baseAddr += aicoreLocalSlotAllocatorSize;

        // Initialize gloabl tensor memory
        auto dynWsMem = aicoreLocalWorkspaceSize - devProg->aicoreLocalWorkspaceSize;
        auto globalTensorMem = dynWsMem + devProg->globalTensorMem;
        globalTensorVerifier_.Init(baseAddr, globalTensorMem);
        aicoreGlobalAllocator_.InitAicoreLocal(baseAddr, globalTensorMem);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspacePartialOutcast(range(baseAddr, baseAddr + globalTensorMem))));
        baseAddr += globalTensorMem;

        // Initialize aicore function internal workspace tensor memory
        auto aicoreLocalFuncWsAllocatorSize = devProg->rootFuncStandardMemReq * devProg->workspaceRecyclePeriod;
        funcWsVerifier_.Init(baseAddr, aicoreLocalFuncWsAllocatorSize);
        aicoreLocalFuncWsAllocator_.InitAicoreLocal(baseAddr, aicoreLocalFuncWsAllocatorSize);
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceInnerTensor(range(baseAddr, baseAddr + aicoreLocalFuncWsAllocatorSize))));
        baseAddr += aicoreLocalFuncWsAllocatorSize;

        // Initialize aicore function internal outcast tensor memory
        uint64_t remaining = workspaceAddr + aicoreLocalWorkspaceSize - baseAddr;
        outcastWsVerifier_.Init(baseAddr, remaining);
        aicoreLocalFuncOutcastAllocator_.InitAicoreLocal(baseAddr, remaining); // Remaining all
        DEV_TRACE_DEBUG(CtrlEvent(none(), WorkspaceInDeviceTaskOutcast(range(baseAddr, baseAddr + remaining))));
        baseAddr += remaining;

        DEV_ASSERT(workspaceAddr <= baseAddr && baseAddr <= workspaceAddr + aicoreLocalWorkspaceSize);
    }

    uint32_t DevFunctionDuppedSlabMemObjSize() {
        if (maxDevFuncDuppedSize_ == 0) {
            for (uint32_t i = 0; i < devProg_->GetFunctionSize(); i++) {
                uint64_t curSize = devProg_->GetFunction(i)->GetDuppedDataAllocSize();
                if (curSize > maxDevFuncDuppedSize_) {
                    maxDevFuncDuppedSize_ = curSize;
                }
            }
        }

        return maxDevFuncDuppedSize_;
    }

    /* 按照devicetask最大支持stitch阈值分配对象 */
    uint32_t DynFuncDataSlabMemObjSize() {
        return (sizeof(DynFuncHeader) + MAX_CACHED_FUNC_NUM * sizeof(DynFuncData));
    }

    /* 按照devicetask最大支持stitch阈值分配对象 */
    uint32_t VecStitchListSLabMemObjSize() {
        return MAX_CACHED_FUNC_NUM * sizeof(DevAscendFunctionDupped);
    }

    uint32_t DynDevTaskSlabMemObjSize() {
        return sizeof(struct DynDeviceTask);
    }

    uint32_t DuppedStitchSlabMemObjSize() {
        return sizeof(struct DevAscendFunctionDuppedStitch);
    }

    uint32_t ReadyQueSlabMemObjSize() {
        return sizeof(ReadyCoreFunctionQueue) + MAX_READY_QUE_ELM_SIZE * sizeof(uint32_t);
    }

    uint32_t (DeviceWorkspaceAllocator::*slabMemObjSizeFunc[ToUnderlying(WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT)])() = {
        &DeviceWorkspaceAllocator::DevFunctionDuppedSlabMemObjSize,
        &DeviceWorkspaceAllocator::DynFuncDataSlabMemObjSize,
        &DeviceWorkspaceAllocator::VecStitchListSLabMemObjSize,
        &DeviceWorkspaceAllocator::DynDevTaskSlabMemObjSize,
        &DeviceWorkspaceAllocator::ReadyQueSlabMemObjSize,
        nullptr, // invalid type
        &DeviceWorkspaceAllocator::DuppedStitchSlabMemObjSize,
    };

    /* 根据当前算子的业务模型分析计算出slab 管理内存页大小, 基于当前可评估的所有内存类型的最大值评估 */
    uint32_t CalcAicpuMetaSlabAlloctorSlabPageSize(uint32_t totalMemSize) {
        uint32_t slabSize = 0;
        constexpr uint32_t extendBuf = 1024;
        uint32_t allocNumOneSlab = 4; // default
        for (size_t i = 0; i < ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT); ++i) {
            if (slabMemObjSizeFunc[i] != nullptr) {
                uint32_t currentSize = (this->*slabMemObjSizeFunc[i])();
                if (currentSize > slabSize) {
                    slabSize = currentSize;
                }
            }
        }
        slabSize += extendBuf;
        uint32_t leastSlabReqMem = (ToUnderlying(WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT)) * slabSize;
        DEV_ASSERT(leastSlabReqMem < totalMemSize);
        uint32_t realMaxAllocNum = totalMemSize / leastSlabReqMem;
        if (realMaxAllocNum < allocNumOneSlab) {
            allocNumOneSlab = realMaxAllocNum;
        }
        slabSize *= allocNumOneSlab;
        return ALIGN_UP(slabSize, sizeof(uint64_t));
    }

    void InitAicpuStitchSlabAllocator(void* memBase, uint32_t totalSize) {
        DEV_ASSERT(memBase != nullptr && totalSize > 0);
        constexpr uint32_t slabSize = 4 * 1024; // fix size
        aicpuStitchSlabAllocator_.Init(memBase, totalSize, slabSize);
        for (size_t i = ToUnderlying(WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) + 1;
            i < ToUnderlying(WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT); ++i) {
            if (slabMemObjSizeFunc[i] != nullptr) {
                uint32_t objSize = (this->*slabMemObjSizeFunc[i])();
                DEV_ASSERT(slabSize > objSize);
                DEV_ASSERT(aicpuStitchSlabAllocator_.RegistCache(i, (this->*slabMemObjSizeFunc[i])()));
            }
        }
    }

    void SlabTryDynAddCache(WsAicpuSlabMemType type, uint32_t objSize) {
        if (type < WsAicpuSlabMemType::COHERENT_SLAB_MEM_TYPE_BUTT) {
            if (!aicpuMetaSlabAllocator_.ExistCache(ToUnderlying(type), objSize)) {
                DEV_ASSERT(aicpuMetaSlabAllocator_.RegistCache(ToUnderlying(type), objSize));
            }
        } else if (type < WsAicpuSlabMemType::SLAB_MEM_TYPE_BUTT) {
            if (!aicpuStitchSlabAllocator_.ExistCache(ToUnderlying(type), objSize)) {
                DEV_ASSERT(aicpuStitchSlabAllocator_.RegistCache(ToUnderlying(type), objSize));
            }
        } else {
            DEV_ASSERT(false);
        }
    }
private:
    bool SlabStageAllocMemTryRecycle() {
        auto FreeTaskSlabMemfunc = [this] (WsSlabStageAllocMem* slabStageMem) -> bool {
            if (slabStageMem->canFree.load(std::memory_order_relaxed)) {
                // recycle slab alloc memory
                aicpuMetaSlabAllocator_.FreeStageAllocMem(slabStageMem->aicpuCoherentStageMem);
                aicpuStitchSlabAllocator_.FreeStageAllocMem(slabStageMem->aicpuStitchStageMem);
                return true;
            }
            return false;
        };

        // try free finished task and recycle aicpu meta memory
        return submmitTaskSlabMemQueue_.FreeUntil(FreeTaskSlabMemfunc);
    }

private:
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    DelayedDumper wsMemDelayedDumper_;
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

    WsAicpuCoherentAllocator aicpuCoherentAllocator_;  // aicpu coherent for small suballocation, not support recycle
    SlabWsAllocator aicpuMetaSlabAllocator_; // aicpu meta memory, support reclamation
    SlabWsAllocator aicpuStitchSlabAllocator_; // aicpu stitched data support reclamation
    SeqWsAllocator aicoreGlobalAllocator_;     // only aicore accesses it
    SeqWsAllocator aicoreLocalFuncWsAllocator_;     // only aicore accesses it
    SeqWsAllocator aicoreLocalFuncOutcastAllocator_;
    WsAicoreLocalSlotAllocator aicoreLocalSlotAllocator_;

#if DEBUG_INFINITE_LIFETIME
    SeqWsAllocator dumpTensorWsAllocator_;
    uint64_t *dumpTensorWsAllocatorCounter_;
#endif

    uintdevptr_t stackWorkspaceBase_{0};
    uint64_t standardStackWorkspacePerCore_{0};
    uint64_t stackWorkspaceSize_{0};

    uint64_t standardRootWorkspace_{0};
    uint32_t maxDevFuncDuppedSize_{0};
    DevAscendProgram *devProg_{nullptr};

    WsMemoryVerifier aicoreLocalWsVerifier_;
    WsMemoryVerifier slotVerifier_;
    WsMemoryVerifier funcWsVerifier_;
    WsMemoryVerifier outcastWsVerifier_;
    WsMemoryVerifier globalTensorVerifier_;

    Vector<uintdevptr_t, WsMemCategory::VECTOR_AICORE_RECYCLE_LIST> slotMemToBeFree_;
    SPSCQueue<WsSlabStageAllocMem *, SUBMMIT_TASK_QUE_SIZE> submmitTaskSlabMemQueue_;
};
} // namespace npu::tile_fwk::dynamic
#endif
