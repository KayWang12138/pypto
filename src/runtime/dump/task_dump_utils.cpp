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
 * \file task_dump_utils.cpp
 * \brief dump and recover for DeviceAgentTask
 */

#include "task_dump_utils.h"
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include "securec.h"
#include "runtime/runtime.h"

namespace npu::tile_fwk {
namespace {
inline size_t MemSizeAlign(const size_t bytes, const uint32_t aligns = 32U) {
    const size_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

void CalcReadyStateSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = task->compileInfo.coreFunctionReadyState.size() * sizeof(CoreFunctionReadyState);
}

void DumpReadyStateData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    memcpy_s(dataPtr, dataSize, task->compileInfo.coreFunctionReadyState.data(), dataSize);
}

void RecoverReadyStateData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    task->compileInfo.coreFunctionReadyState.resize(dataSize / sizeof(CoreFunctionReadyState));
    memcpy_s(task->compileInfo.coreFunctionReadyState.data(), dataSize, dataPtr, dataSize);
}

void CalcAicReadyFuncSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = task->compileInfo.readyAicIdVec.size() * sizeof(uint64_t);
}

void DumpAicReadyFuncData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    memcpy_s(dataPtr, dataSize, task->compileInfo.readyAicIdVec.data(), dataSize);
}

void RecoverAicReadyFuncData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    task->compileInfo.readyAicIdVec.resize(dataSize / sizeof(uint64_t));
    memcpy_s(task->compileInfo.readyAicIdVec.data(), dataSize, dataPtr, dataSize);
}

void CalcAivReadyFuncSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = task->compileInfo.readyAivIdVec.size() * sizeof(uint64_t);
}

void DumpAivReadyFuncData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    memcpy_s(dataPtr, dataSize, task->compileInfo.readyAivIdVec.data(), dataSize);
}

void RecoverAivReadyFuncData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    task->compileInfo.readyAivIdVec.resize(dataSize / sizeof(uint64_t));
    memcpy_s(task->compileInfo.readyAivIdVec.data(), dataSize, dataPtr, dataSize);
}

void CalcCpuReadyFuncSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = task->compileInfo.readyAicpuIdVec.size() * sizeof(uint64_t);
}

void DumpCpuReadyFuncData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    memcpy_s(dataPtr, dataSize, task->compileInfo.readyAicpuIdVec.data(), dataSize);
}

void RecoverCpuReadyFuncData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    task->compileInfo.readyAicpuIdVec.resize(dataSize / sizeof(uint64_t));
    memcpy_s(task->compileInfo.readyAicpuIdVec.data(), dataSize, dataPtr, dataSize);
}

void CalcCacheHeaderSize(const DeviceAgentTask *task, size_t &dataSize) {
    (void) task;
    dataSize = sizeof(CacheHeader);
}

void DumpCacheHeaderData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    memcpy_s(dataPtr, dataSize, &cacheValue.header, dataSize);
}

void RecoverCacheHeaderData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    memcpy_s(&cacheValue.header, dataSize, dataPtr, dataSize);
    task->SetFunctionCache(cacheValue);
}

void CalcCacheBinSize(const DeviceAgentTask *task, size_t &dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    CoreFunctionBinCache *cacheBin = cacheValue.binCache;
    dataSize = cacheBin->dataSize + sizeof(uint64_t); // datasize字段头也一起加上
}

void DumpCacheBinData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    CoreFunctionBinCache *cacheBin = cacheValue.binCache;
    memcpy_s(dataPtr, dataSize, cacheBin, dataSize);
}

void RecoverCacheBinData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    cacheValue.binCache = reinterpret_cast<CoreFunctionBinCache *>(new(std::nothrow) uint8_t[dataSize]);
    memcpy_s(cacheValue.binCache, dataSize, dataPtr, dataSize);
    task->SetFunctionCache(cacheValue);
}

void CalcCacheTopoSize(const DeviceAgentTask *task, size_t &dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    CoreFunctionTopoCache *cacheTopo = cacheValue.topoCache;
    dataSize = cacheTopo->dataSize + sizeof(uint64_t); // datasize字段头也一起加上
}

void DumpCacheTopoData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    CoreFunctionTopoCache *cacheTopo = cacheValue.topoCache;
    memcpy_s(dataPtr, dataSize, cacheTopo, dataSize);
}

void RecoverCacheTopoData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    (void) dataSize;
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    cacheValue.topoCache = reinterpret_cast<CoreFunctionTopoCache *>(new(std::nothrow) uint8_t[dataSize]);
    memcpy_s(cacheValue.topoCache, dataSize, dataPtr, dataSize);
    task->SetFunctionCache(cacheValue);
}

void CalcDistTilingSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = 0;
    const std::unordered_map<std::string, Distributed::TilingStorage>& tilingData =
        task->compileTask->GetFunction()->GetDistTilingManager()->GetAllTilingTensorData();
    for (const auto &item : tilingData) {
        dataSize += item.first.size();
        dataSize++;
        dataSize += sizeof(size_t);
        dataSize += item.second.GetValidsize();
    }
}

void DumpDistTilingData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    (void)dataSize;
    const std::unordered_map<std::string, Distributed::TilingStorage>& tilingData =
        task->compileTask->GetFunction()->GetDistTilingManager()->GetAllTilingTensorData();
    size_t offset = 0;
    for (const auto &item : tilingData) {
        std::vector<uint8_t> symbolData;
        for (char ch : item.first) {
            symbolData.push_back(ch);
        }
        symbolData.push_back('\0');
        (void)memcpy_s(dataPtr + offset, symbolData.size(), symbolData.data(), symbolData.size());
        offset += symbolData.size();
        size_t validSize = item.second.GetValidsize();
        (void)memcpy_s(dataPtr + offset, sizeof(size_t), &validSize, sizeof(size_t));
        offset += sizeof(validSize);
        (void)memcpy_s(dataPtr + offset, validSize, item.second.GetConstStoragePtr(), validSize);
        offset += item.second.GetValidsize();
    }
}

void RecoverDistTilingData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    std::shared_ptr<Distributed::TilingManager> tilingManager =
        task->compileTask->GetFunction()->GetDistTilingManager();
    size_t offset = 0;
    while (offset < dataSize) {
        std::string symbol(reinterpret_cast<const char *>(dataPtr + offset));
        offset = offset + symbol.size() + 1;
        size_t validSize = 0;
        (void)memcpy_s(&validSize, sizeof(size_t), dataPtr + offset, sizeof(size_t));
        offset += sizeof(size_t);
        std::vector<int> tilingData(validSize/sizeof(int), 0);
        (void)memcpy_s(tilingData.data(), validSize, dataPtr + offset, validSize);
        offset += validSize;
        tilingManager->Save(symbol, tilingData);
    }
}

void CalcTensorInfoSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = task->compileInfo.coreTensorInfoVec.size() * sizeof(TensorInfo);
}

void DumpTensorInfoData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    memcpy_s(dataPtr, dataSize, task->compileInfo.coreTensorInfoVec.data(), dataSize);
}

void RecoverTensorInfoData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    task->compileInfo.coreTensorInfoVec.resize(dataSize / sizeof(TensorInfo));
    memcpy_s(task->compileInfo.coreTensorInfoVec.data(), dataSize, dataPtr, dataSize);
}

struct DumpParaOffsetInfo {
    uint64_t coreFuncId;
    size_t opOriginArgsSeq;
    uint64_t offset;
    int rawMagic;
    uint64_t rawTensorOffset;
    bool isTensorParam;

    DumpParaOffsetInfo() : coreFuncId(0), opOriginArgsSeq(INVALID_IN_OUT_INDEX), offset(0),
                           rawMagic(0), rawTensorOffset(0), isTensorParam(false) {}

    DumpParaOffsetInfo(const uint64_t funcId, const size_t argsSeq, const uint64_t paraOffset, const int magic,
                       const uint64_t tensorOffset, const bool isTensor)
        : coreFuncId(funcId), opOriginArgsSeq(argsSeq), offset(paraOffset),
          rawMagic(magic), rawTensorOffset(tensorOffset), isTensorParam(isTensor) {}
};

void CalcParaOffsetSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = 0;
    for (const auto &paraOffset : task->compileInfo.invokeParaOffset) {
        dataSize += paraOffset.second.size() * sizeof(DumpParaOffsetInfo);
    }
}

void DumpParaOffsetData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    std::vector<DumpParaOffsetInfo> paraOffsetVec;
    for (const auto &paraOffsetEntry  : task->compileInfo.invokeParaOffset) {
        for (const InvokeParaOffset &paraOffset : paraOffsetEntry.second) {
            DumpParaOffsetInfo paraOffsetInfo(paraOffsetEntry.first, paraOffset.opOriginArgsSeq, paraOffset.offset,
                                              paraOffset.rawMagic, paraOffset.rawTensorOffset, paraOffset.isTensorParam);
            paraOffsetVec.push_back(paraOffsetInfo);
        }
    }
    memcpy_s(dataPtr, dataSize, paraOffsetVec.data(), dataSize);
}

void RecoverParaOffsetData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    std::vector<DumpParaOffsetInfo> paraOffsetVec;
    paraOffsetVec.resize(dataSize / sizeof(DumpParaOffsetInfo));
    memcpy_s(paraOffsetVec.data(), dataSize, dataPtr, dataSize);
    for (const DumpParaOffsetInfo &paraOffsetInfo : paraOffsetVec) {
        InvokeParaOffset invokeParaOffset;
        invokeParaOffset.isTensorParam = paraOffsetInfo.isTensorParam;
        invokeParaOffset.opOriginArgsSeq = paraOffsetInfo.opOriginArgsSeq;
        invokeParaOffset.offset = paraOffsetInfo.offset;
        invokeParaOffset.rawMagic = paraOffsetInfo.rawMagic;
        auto rawTensor =
            task->compileTask->GetFunction()->GetTensorMap().GetRawTensorByRawMagic(paraOffsetInfo.rawMagic);
        if (rawTensor != nullptr) {
            invokeParaOffset.rawSymbol = rawTensor->symbol;
        }
        invokeParaOffset.rawTensorOffset = paraOffsetInfo.rawTensorOffset;
        task->compileInfo.invokeParaOffset[paraOffsetInfo.coreFuncId].push_back(invokeParaOffset);
    }
}

void CalcFuncWSSize(const DeviceAgentTask *task, size_t &dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    dataSize = (cacheValue.header.coreFunctionNum) * sizeof(CoreFunctionWsAddr);
}

void DumpFuncWSSizeData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    auto &compileInfo = task->compileInfo;
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    std::vector<CoreFunctionWsAddr> coreFunctionWsOffset;
    CoreFunctionTopoCache *cacheTopo = cacheValue.topoCache;
    uint64_t *topoOffset = cacheTopo->coreFunctionTopoOffsets;
    for (uint64_t i = 0; i < cacheValue.header.coreFunctionNum; i++) {
        coreFunctionWsOffset.push_back(CoreFunctionWsAddr(compileInfo.coreFuncBinOffset.at(i),
                                                          compileInfo.coreFunctionInvokeEntryOffset.at(i),
                                                          compileInfo.coreFunctionIdToProgramId.at(i), topoOffset[i],
                                                          compileInfo.coreFunctionTensorInfoOffset.at(i),
                                                          compileInfo.coreTensorNum.at(i)));
    }
    memcpy_s(dataPtr, dataSize, coreFunctionWsOffset.data(), dataSize);
}

void RecoverFuncWSSizeData(const uint8_t *dataPtr, const size_t dataSize, DeviceAgentTask *task) {
    std::vector<CoreFunctionWsAddr> coreFunctionWsOffset;
    coreFunctionWsOffset.resize(dataSize / sizeof(CoreFunctionWsAddr));
    memcpy_s(coreFunctionWsOffset.data(), dataSize, dataPtr, dataSize);
    task->compileInfo.coreFuncBinOffset.resize(coreFunctionWsOffset.size());
    task->compileInfo.coreFunctionInvokeEntryOffset.resize(coreFunctionWsOffset.size());
    task->compileInfo.coreFunctionTensorInfoOffset.resize(coreFunctionWsOffset.size());
    task->compileInfo.coreTensorNum.resize(coreFunctionWsOffset.size());
    for (size_t i = 0; i < coreFunctionWsOffset.size(); ++i) {
        task->compileInfo.coreFuncBinOffset[i] = coreFunctionWsOffset[i].functionBinAddr;
        task->compileInfo.coreFunctionInvokeEntryOffset[i] = coreFunctionWsOffset[i].invokeEntryAddr;
        task->compileInfo.coreFunctionIdToProgramId[i] = coreFunctionWsOffset[i].psgId;
        task->compileInfo.coreFunctionTensorInfoOffset[i] = coreFunctionWsOffset[i].invokeEntryInfo;
        task->compileInfo.coreTensorNum[i] = coreFunctionWsOffset[i].invokeEntryNum;
    }
}

enum class DumpDataType {
    READY_STATUS = 0,      // CoreFunction ready_status(no need update)
    READY_AIC_CORE_FUNC,   // ready aic CoreFunction id list(no need update)
    READY_AIV_CORE_FUNC,   // ready aiv CoreFunction id list(no need update)
    READY_CPU_CORE_FUNC,   // ready aicpu CoreFunction id list(no need update)
    CACHE_HEADER,          // cache header
    CACHE_BIN,             // all cce bin
    CACHE_TOPO,            // cache topo
    DIST_TILING,           // distributed tiling data
    INVODE_TENSOR_INFO,    // invoke tensor
    INVOKE_PARA_OFFSET,    // invoke para offset
    CORE_FUNC_WS_ADDR,     // corefunc args
    BOTTOM
};

#pragma pack (8)
struct DeviceTaskBinData {
    BaseArgs baseArgs;
    DeviceTask deviceTask;        // initial task data
    uint64_t dataSize[static_cast<size_t>(DumpDataType::BOTTOM)];
    uint64_t dataOffset[static_cast<size_t>(DumpDataType::BOTTOM)];
    uint8_t data[0];
};
#pragma pack ()

using CalcBinDataSize = void (*)(const DeviceAgentTask *, size_t &);
using DumpBinData = void (*)(const DeviceAgentTask *, uint8_t *, size_t);
using RecoverBinData = void (*)(const uint8_t *, const size_t, DeviceAgentTask *);
struct BinDataFuncSet {
    DumpDataType type;
    CalcBinDataSize calcFunc;
    DumpBinData dumpFunc;
    RecoverBinData recoverFunc;
};

BinDataFuncSet kBinDataProcVec[static_cast<size_t>(DumpDataType::BOTTOM)] = {
    {DumpDataType::READY_STATUS,        CalcReadyStateSize,   DumpReadyStateData,   RecoverReadyStateData},
    {DumpDataType::READY_AIC_CORE_FUNC, CalcAicReadyFuncSize, DumpAicReadyFuncData, RecoverAicReadyFuncData},
    {DumpDataType::READY_AIV_CORE_FUNC, CalcAivReadyFuncSize, DumpAivReadyFuncData, RecoverAivReadyFuncData},
    {DumpDataType::READY_CPU_CORE_FUNC, CalcCpuReadyFuncSize, DumpCpuReadyFuncData, RecoverCpuReadyFuncData},
    {DumpDataType::CACHE_HEADER,        CalcCacheHeaderSize,  DumpCacheHeaderData,  RecoverCacheHeaderData},
    {DumpDataType::CACHE_BIN,           CalcCacheBinSize,     DumpCacheBinData,     RecoverCacheBinData},
    {DumpDataType::CACHE_TOPO,          CalcCacheTopoSize,    DumpCacheTopoData,    RecoverCacheTopoData},
    {DumpDataType::DIST_TILING,         CalcDistTilingSize,   DumpDistTilingData,   RecoverDistTilingData},
    {DumpDataType::INVODE_TENSOR_INFO,  CalcTensorInfoSize,   DumpTensorInfoData,   RecoverTensorInfoData},
    {DumpDataType::INVOKE_PARA_OFFSET,  CalcParaOffsetSize,   DumpParaOffsetData,   RecoverParaOffsetData},
    {DumpDataType::CORE_FUNC_WS_ADDR,   CalcFuncWSSize,       DumpFuncWSSizeData,   RecoverFuncWSSizeData}
};
}

bool TaskDumpUtils::DumpTaskToBinFile(const DeviceAgentTask *deviceAgentTask, const std::string &dumpFileName) {
    ALOG_DEBUG_F("Begin to dump bin file[%s].", dumpFileName.c_str());
    std::vector<uint64_t> dataSizes;
    std::vector<uint64_t> alignSizes;
    size_t dataSize = 0;
    size_t alignSize = 0;
    size_t allSize = sizeof(DeviceTaskBinData);
    for (size_t i = 0; i < static_cast<size_t>(DumpDataType::BOTTOM); ++i) {
        const auto &dataProc = kBinDataProcVec[i];
        dataProc.calcFunc(deviceAgentTask, dataSize);
        dataSizes.emplace_back(dataSize);
        alignSize = MemSizeAlign(dataSize); // device地址需要考虑对齐，这里按照32对齐下
        alignSizes.emplace_back(alignSize);
        allSize += alignSize;
        ALOG_DEBUG_F("Calc data size[%zu] align size[%zu].", dataSize, alignSize);
    }

    DeviceTaskBinData *taskBin = reinterpret_cast<DeviceTaskBinData *>(new (std::nothrow) uint8_t[allSize]);
    if (taskBin == nullptr) {
        return false;
    }
    auto binData = &taskBin->baseArgs;
    binData->opaque = 0;
    binData->taskId = deviceAgentTask->compileTask->GetTaskId();
    taskBin->deviceTask.coreFunctionCnt = deviceAgentTask->compileInfo.coreFunctionCnt;
    taskBin->deviceTask.coreFuncData.stackWorkSpaceSize = deviceAgentTask->compileInfo.workSpaceStackSize;
    taskBin->deviceTask.coreFuncData.stackWorkSpaceAddr = deviceAgentTask->compileInfo.invokeParaWorkSpaceSize;
    ALOG_DEBUG_F("Dump FunctionCnt[%lu] taskId[%lu] WorkSpaceSize[%lu].", taskBin->deviceTask.coreFunctionCnt,
                 binData->taskId, taskBin->deviceTask.coreFuncData.stackWorkSpaceSize);

    // 1.base data
    size_t offset = sizeof(DeviceTaskBinData);
    uint8_t *basePtr = reinterpret_cast<uint8_t *>(taskBin);
    for (size_t i = 0; i < static_cast<size_t>(DumpDataType::BOTTOM); ++i) {
        taskBin->dataOffset[i] = offset;
        taskBin->dataSize[i] = dataSizes[i];
        if (dataSizes[i] > 0) {
            const auto &dataProc = kBinDataProcVec[i];
            dataProc.dumpFunc(deviceAgentTask, basePtr + offset, dataSizes[i]);
        }
        offset += alignSizes[i];
        ALOG_DEBUG_F("Dump data offset[%zu].\n", offset);
    }

    bool ret = DumpFile(reinterpret_cast<const char *>(taskBin), allSize, dumpFileName);
    delete[] taskBin;
    return ret;
}

bool TaskDumpUtils::RecoverTaskFromBinFile(const std::string &binFilePath, DeviceAgentTask *deviceAgentTask) {
    std::vector<char> binBuffer;
    if (!ReadBytesFromFile(binFilePath, binBuffer)) {
        return false;
    }
    DeviceTaskBinData *taskBin = reinterpret_cast<DeviceTaskBinData *>(binBuffer.data());
    deviceAgentTask->compileInfo.coreFunctionCnt = taskBin->deviceTask.coreFunctionCnt;
    deviceAgentTask->compileInfo.workSpaceStackSize = taskBin->deviceTask.coreFuncData.stackWorkSpaceSize;
    deviceAgentTask->compileInfo.invokeParaWorkSpaceSize = taskBin->deviceTask.coreFuncData.stackWorkSpaceAddr;
    if (deviceAgentTask->GetFuncCacheValue() == std::nullopt) {
        CacheValue cacheValue;
        deviceAgentTask->SetFunctionCache(cacheValue);
    }

    uint8_t* baseAddr = reinterpret_cast<uint8_t*>(taskBin);
    for (size_t i = 0; i < static_cast<size_t>(DumpDataType::BOTTOM); ++i) {
        if (taskBin->dataSize[i] > 0) {
            const auto &dataProc = kBinDataProcVec[i];
            dataProc.recoverFunc(baseAddr + taskBin->dataOffset[i], taskBin->dataSize[i], deviceAgentTask);
        }
        ALOG_INFO_F("Recover bin data[%zu], data size[%]", i, taskBin->dataSize[i]);
    }
    return true;
}
} // namespace npu::tile_fwk
