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
 * \file machine_agent.h
 * \brief
 */

#ifndef MACHINE_AGENT_H
#define MACHINE_AGENT_H
#include <iostream>
#include "interface/machine/host/machine_task.h"
#include "machine_compiler.h"
#include "interface/cache/function_cache.h"
#include "runtime/runtime.h"

namespace npu::tile_fwk {
constexpr int64_t MACHINE_DEBUG = 1;
constexpr int64_t MACHINE_ERROR = -1;
constexpr int64_t MACHINE_OK = 0;

#if defined(MACHINE_DEBUG) && MACHINE_DEBUG == 1
#define MACHINE_ASSERT(exp) ASSERT(exp)
#else
#define MACHINE_ASSERT(exp)
#endif

class HostMachine;

/* 每次device agent 处理后的信息, 如所有在workspace申请的内存, 每个AscendFunction一个 */
struct MachineDeviceAgentInfo {
    //uint32_t aicoreCnt{AICORE_NUM}; // 从全局配置获取
    uint8_t* workspaceGmAddr{nullptr};
    uint8_t* invokeEntryOffsetsGmAddr{nullptr};
    uint8_t* topoGmAddr{nullptr};
    uint8_t* functionBinGmAddr{nullptr};
    uint8_t* readyStateGmAddr{nullptr};
    uint8_t* coreFuncWsAddrGmAddr{nullptr};
    uint8_t* deviceTaskGmAddr{nullptr};
    uint8_t* readyAicQueElmGmAddr{nullptr};
    uint8_t* readyAivQueElmGmAddr{nullptr};
    uint8_t* readyAicpuQueElmGmAddr{nullptr};
    uint8_t* readyAicQueGmAddr{nullptr};
    uint8_t* readyAivQueGmAddr{nullptr};
    uint8_t* readyAicpuQueGmAddr{nullptr};

    std::vector<uint64_t> coreFunctionInvokeEntryAddr;
    std::vector<uint64_t> coreFunctionInvokeEntryInfo;
    std::vector<uint64_t> coreFunctionTopoAddr;
    std::vector<uint64_t> coreFuncBinAddr;
    std::vector<npu::tile_fwk::CoreFunctionWsAddr> coreFunctionWsAddr;
    npu::tile_fwk::DeviceTask devceTask;
    std::map<int, uint8_t *> stubOutRawTensorAddr;
    std::vector<uint64_t> coreFunctionInvokeEntryOriAddr;

    void FreeDevMemory() const {
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
        //runtime::GetRA()->FreeTensor(workspaceGmAddr); //  stub out raw tensor addr 依赖此，暂时不能释放
        npu::tile_fwk::runtime::GetRA()->FreeTensor(invokeEntryOffsetsGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(topoGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(functionBinGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(readyStateGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(coreFuncWsAddrGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(deviceTaskGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(readyAicQueElmGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(readyAivQueElmGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(readyAicpuQueElmGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(readyAicQueGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(readyAivQueGmAddr);
        npu::tile_fwk::runtime::GetRA()->FreeTensor(readyAicpuQueGmAddr);
#endif
    }
};
class DeviceAgentTask {
public:
    explicit DeviceAgentTask(npu::tile_fwk::MachineTask *task) : compileTask(task) {}
    ~DeviceAgentTask() {}

    uint64_t GetTaskId() const { return this->compileTask->GetTaskId(); }
    npu::tile_fwk::Function *GetFunction() const { return this->compileTask->GetFunction(); }
    std::optional<npu::tile_fwk::CacheValue> GetFuncCacheValue() const { return this->cacheValue_;}
    void SetFunctionCache(std::optional<npu::tile_fwk::CacheValue> value) { this->cacheValue_ = value; }
    uint64_t GetWorkSpaceSize() const { return compileInfo.aicoreCnt * compileInfo.workSpaceStackSize + compileInfo.invokeParaWorkSpaceSize; }
    uint8_t* GetDeviceTaskGmAddr() const { return this->deviceInfo.deviceTaskGmAddr; }
    void SetDeviceWorkSpaceAddr(uint8_t* addr) { this->deviceInfo.workspaceGmAddr = addr; }
    void SetAicpuStream(void* stream) { this->aicpuStream_ = stream; }
    void SetOpOriginArgsInfo(const std::vector<OriArgInfo>& originArgs) {
        this->opOriginArgs_ = originArgs;
    }

    uint8_t* GetOpOriginArgsRawTensorAddr(size_t seq) {
        ASSERT(seq < this->opOriginArgs_.size());
        return reinterpret_cast<uint8_t*>(this->opOriginArgs_[seq].addr);
    }

    void SetAsync(bool isAsync) { this->isAsync_ = isAsync; }

    bool IsAsync() const { return this->isAsync_; }
    void ProcessReadyCoreFunctions(const CacheValue &cacheValue) {
        ReadyCoreFunctionCache *readyFunction = cacheValue.readyListCache;
        for (uint64_t i = 0; i < cacheValue.header.readyCoreFunctionNum; i++) {
            if (readyFunction->readyCoreFunction[i].coreType == static_cast<uint64_t>(CoreType::AIC)) {
                this->compileInfo.readyAicIdVec.emplace_back(readyFunction->readyCoreFunction[i].id);
                ALOG_DEBUG_F("ready aic function: %lu", readyFunction->readyCoreFunction[i].id);
            } else if (readyFunction->readyCoreFunction[i].coreType == static_cast<uint64_t>(CoreType::AICPU)) {
                this->compileInfo.readyAicpuIdVec.emplace_back(readyFunction->readyCoreFunction[i].id);
                ALOG_DEBUG_F("ready aicpu function: %lu", readyFunction->readyCoreFunction[i].id);
            } else {
                this->compileInfo.readyAivIdVec.emplace_back(readyFunction->readyCoreFunction[i].id);
                ALOG_DEBUG_F("ready aiv function: %lu", readyFunction->readyCoreFunction[i].id);
            }
        }
    }

    void UpdateCoreFunction(const CacheValue &cacheValue) {
        CoreFunctionTopoCache *cacheTopo = cacheValue.topoCache;
        uint64_t coreFuncNum = cacheValue.header.coreFunctionNum;
        uint64_t *topoOffset = cacheTopo->coreFunctionTopoOffsets;
        uint64_t *binOffset = cacheValue.binCache->coreFunctionBinOffsets;
        for (uint64_t i = 0; i < coreFuncNum; i++) {
            CoreFunctionTopo *oneTopo = reinterpret_cast<CoreFunctionTopo *>(
                    reinterpret_cast<uint8_t *>(cacheTopo) + topoOffset[i]);
            this->compileInfo.coreFunctionIdToProgramId.insert({i, oneTopo->psgId}); // 缓存下来后面functionbin偏移会用
            this->compileInfo.coreFunctionReadyState.emplace_back(
                CoreFunctionReadyState(oneTopo->readyCount, oneTopo->coreType));
            ALOG_DEBUG_F("core function : topoAddr %lx readyCount %ld coreType %lu", i,
                oneTopo->readyCount, oneTopo->coreType);
            ASSERT((oneTopo->coreType == static_cast<uint64_t>(MachineType::AIC)) ||
                   (oneTopo->coreType == static_cast<uint64_t>(MachineType::AIV)) ||
                   (oneTopo->coreType == static_cast<uint64_t>(MachineType::HUB)) ||
                   (oneTopo->coreType == static_cast<uint64_t>(MachineType::AICPU)));
            uint64_t offset = binOffset[oneTopo->psgId] + sizeof(uint64_t);
            this->compileInfo.coreFuncBinOffset.emplace_back(offset);
        }
        for (uint64_t i = coreFuncNum; i < cacheValue.header.virtualFunctionNum + coreFuncNum; i++) {
            CoreFunctionTopo *oneTopo = reinterpret_cast<CoreFunctionTopo *>(
                    reinterpret_cast<uint8_t *>(cacheTopo) + topoOffset[i]);
            ASSERT((oneTopo->coreType == static_cast<uint64_t>(MachineType::VIRTUAL_PURE)) ||
                (oneTopo->coreType == static_cast<uint64_t>(MachineType::VIRTUAL_MIX)));
            this->compileInfo.coreFunctionReadyState.emplace_back(
                CoreFunctionReadyState(oneTopo->readyCount, oneTopo->coreType));
            ALOG_DEBUG_F("virtual core function : topoAddr %lx readyCount %ld coreType %lu", i,
                oneTopo->readyCount, oneTopo->coreType);
        }
    }

    void UpdateCompileInfo() {
        CacheValue cacheValue = this->GetFuncCacheValue().value();
        UpdateCoreFunction(cacheValue);
        ProcessReadyCoreFunctions(cacheValue);
        auto &coreTensorInfoVec = this->compileInfo.coreTensorInfoVec;
        size_t invokeOffsetSize = 0;
        for (auto &mapEntry : this->compileInfo.invokeParaOffset) {
            std::vector<uint64_t> argsOffset;
            std::vector<int64_t> tensorsIdx;
            std::list<InvokeParaOffset> &invokeParaOffsetList = mapEntry.second;
            ALOG_DEBUG_F("Tensornum[%zu].", invokeParaOffsetList.size());
            this->compileInfo.coreTensorNum.emplace_back(invokeParaOffsetList.size());
            this->compileInfo.coreFunctionTensorInfoOffset.emplace_back(coreTensorInfoVec.size() * sizeof(TensorInfo));
            this->compileInfo.coreFunctionInvokeEntryOffset.emplace_back((invokeOffsetSize * sizeof(uint64_t)));
            for (auto &elm : invokeParaOffsetList) {
                invokeOffsetSize++;
                TensorInfo tensorInfo;
                SetDumpTensorInfo(elm, tensorInfo, this);
                ALOG_DEBUG_F("Current tensor info paramType is %d, dims is %u, tensorInforpid is %d\n",
                    tensorInfo.paramType, tensorInfo.dims, tensorInfo.hostpid);
                argsOffset.emplace_back(elm.offset);
                if (elm.opOriginArgsSeq == INVALID_IN_OUT_INDEX) {
                    tensorsIdx.emplace_back(-1);
                } else {
                    tensorsIdx.emplace_back(elm.opOriginArgsSeq);
                }
                ALOG_DEBUG_F("offset %lu  opOriginArgsSeq %zu.\n", elm.offset, elm.opOriginArgsSeq);
                coreTensorInfoVec.emplace_back(tensorInfo);
            }
            this->compileInfo.invokeArgsOffset.emplace_back(argsOffset);
            this->compileInfo.invokeTensorsIdx.emplace_back(tensorsIdx);
        }
        this->compileInfo.invokeOffsetSize = invokeOffsetSize;
        return;
    }
    void Validate() {
        ASSERT(this->GetFuncCacheValue() != std::nullopt);
        ASSERT(this->compileInfo.coreFunctionCnt != 0);
        ASSERT(this->GetFuncCacheValue().value().header.coreFunctionNum == this->compileInfo.coreFunctionCnt);
        ASSERT(this->compileInfo.coreFunctionCnt == this->compileInfo.invokeParaOffset.size());
    }
private:
    void SetDumpTensorInfo(const InvokeParaOffset &elm, TensorInfo &tensorInfo, const DeviceAgentTask *task) const {
        tensorInfo.functionMagic = elm.funcitonMagic;
        tensorInfo.dataType = static_cast<uint16_t>(elm.datatype);
        tensorInfo.dims = elm.tensorShape.size();
        tensorInfo.paramType = elm.paramType;
        tensorInfo.idx = elm.ioIndex;
        if (IsAstDataDumpEnabled()) {
            tensorInfo.hostpid = getpid();
        }
        tensorInfo.subgraphId = task->GetFunction()->Operations()[0].GetSubgraphID();
        tensorInfo.deviceId = npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId();
        tensorInfo.rawMagic = elm.rawMagic;
        tensorInfo.opMagic = elm.opMagic;
        tensorInfo.dataByte = BytesOf(elm.datatype);
        ALOG_DEBUG_F("Current tile tensor rawMagic %u, opMagic %u", tensorInfo.rawMagic, tensorInfo.opMagic);
        for (size_t i = 0; i < elm.tensorShape.size(); i++) {
            tensorInfo.shape[i] = elm.tensorShape[i];
            tensorInfo.stride[i] = elm.rawTensorShape[i];
            ALOG_DEBUG_F("tensor shape[%zu] = %d, stride[%zu] = %d", i, tensorInfo.shape[i], i, tensorInfo.stride[i]);
        }
    }
public:
    std::vector<OriArgInfo> opOriginArgs_;
    void* aicpuStream_{nullptr};
    npu::tile_fwk::MachineTask *compileTask;
    MachineCompileInfo compileInfo;
    MachineDeviceAgentInfo deviceInfo;
private:
    bool isAsync_{false};
    std::optional<CacheValue> cacheValue_;
};

class MachineAgent {
public:
    MachineAgent() = default;
    void AgentProc(DeviceAgentTask* task);
private:
    void DumpData(const std::string &fileName, const char* data,size_t len);
    void Validate(DeviceAgentTask* task);
    int PrepareWorkSpace(DeviceAgentTask* task);
    int PrepareInvokeEntry(DeviceAgentTask* task);
    int PrepareTopo(DeviceAgentTask* task);
    int PrepareCoreFunctionBin(DeviceAgentTask* task);
    int PrepareReadyCoreFunction(DeviceAgentTask* task);
    int PrepareHcclContext(DeviceAgentTask *task);
    int PrepareReadyState(DeviceAgentTask* task);
    void FillL2PrefetchInfo(DeviceAgentTask *task, DeviceTask &devTask) const;
    void FillDeviceTask(DeviceAgentTask *task, DeviceTask &devTask, MachineDeviceAgentInfo &devInfo) const;
    void DumpDeviceTaskInfo(const DeviceAgentTask *task, uint8_t *deviceTaskGmAddr, const DeviceTask &devTask) const;
    int ConstructDeviceTask(DeviceAgentTask* task);
    void SetDumpTensorInfo(InvokeParaOffset &elm, TensorInfo &tensorInfo, MachineTask *task);
    void FillVirtualFunction(DeviceAgentTask *task);
};

class MachinePipe {
public:
    MachinePipe() = default;

    void PipeProc(DeviceAgentTask* task);
};
int Run(const void *stream, const void *workSpaceGmAddr, DeviceAgentTask *deviceAgentTask,
    const std::vector<void *> &opOriginArgs, const std::vector<size_t> &argsSize, bool isAsync);
}
#endif // MACHINE_AGENT_H
