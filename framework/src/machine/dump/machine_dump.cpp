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
 * \file machine_dump.cpp
 * \brief
 */

#include "machine_dump.h"
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include "securec.h"
#include "machine/platform/platform_manager.h"

namespace npu::tile_fwk {
namespace {
constexpr int64_t NUM_FOUR = 4;
using Json = nlohmann::json;
using CalcBinDataSize = void (*)(const DeviceAgentTask *, size_t &);
using DumpBinData = void (*)(const DeviceAgentTask *, uint8_t *, size_t);

struct DumpParaOffsetInfo {
    uint64_t coreFuncId;
    size_t opOriginArgsSeq;
    uint64_t offset;
    bool isTensorParam;

    DumpParaOffsetInfo() : coreFuncId(0), opOriginArgsSeq(INVALID_IN_OUT_INDEX), offset(0), isTensorParam(false) {}

    DumpParaOffsetInfo(const uint64_t funcId, const size_t argsSeq, const uint64_t paraOffset, const bool isTensor)
        : coreFuncId(funcId), opOriginArgsSeq(argsSeq), offset(paraOffset), isTensorParam(isTensor) {}
};

inline size_t MemSizeAlign(const size_t bytes, const uint32_t aligns = 32U) {
    const size_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

void CalcReadyStateSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = task->compileInfo.coreFunctionReadyState.size() * sizeof(CoreFunctionReadyState);
    return;
}

void DumpReadyStateData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    memcpy_s(dataPtr, dataSize, task->compileInfo.coreFunctionReadyState.data(), dataSize);
}

void CalcAicReadyFuncSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = sizeof(ReadyCoreFunctionQueue) + task->compileInfo.readyAicIdVec.size() * sizeof(uint64_t);
    return;
}

void DumpAicReadyFuncData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    (void) dataSize;
    ReadyCoreFunctionQueue rq;
    rq.head = 0;
    rq.tail = task->compileInfo.readyAicIdVec.size();
    rq.elem = nullptr; // need update on aicpu
    rq.lock = 0;
    ALOG_INFO_F("AIC ready size: %zu", rq.tail);
    memcpy_s(dataPtr, sizeof(StaticReadyCoreFunctionQueue ), &rq, sizeof(StaticReadyCoreFunctionQueue ));
    auto readyPtr = dataPtr + sizeof(StaticReadyCoreFunctionQueue );
    auto readySize = task->compileInfo.readyAicIdVec.size() * sizeof(uint64_t);
    memcpy_s(readyPtr, readySize, task->compileInfo.readyAicIdVec.data(), readySize);
}

void CalcAivReadyFuncSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = sizeof(StaticReadyCoreFunctionQueue) + task->compileInfo.readyAivIdVec.size() * sizeof(uint64_t);
    return;
}

void DumpAivReadyFuncData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    (void)dataSize;
    StaticReadyCoreFunctionQueue rq;
    rq.head = 0;
    rq.tail = task->compileInfo.readyAivIdVec.size();
    rq.elem = nullptr; // need update on aicpu
    rq.lock = 0;
    ALOG_INFO_F("AIV ready size: %zu", rq.tail);
    memcpy_s(dataPtr, sizeof(StaticReadyCoreFunctionQueue), &rq, sizeof(StaticReadyCoreFunctionQueue));
    auto readyPtr = dataPtr + sizeof(StaticReadyCoreFunctionQueue);
    auto readySize = task->compileInfo.readyAivIdVec.size() * sizeof(uint64_t);
    memcpy_s(readyPtr, readySize, task->compileInfo.readyAivIdVec.data(), readySize);
}

void CalcCacheHeaderSize(const DeviceAgentTask *task, size_t &dataSize) {
    (void) task;
    dataSize = sizeof(CacheHeader);
}

void DumpCacheHeaderData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    memcpy_s(dataPtr, dataSize, &cacheValue.header, dataSize);
}

void CalcCCEBinSize(const DeviceAgentTask *task, size_t &dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    CoreFunctionBinCache *cacheBin = cacheValue.binCache;
    dataSize = cacheBin->dataSize + sizeof(uint64_t); // datasize字段头也一起加上
    return;
}

void DumpCCEBinData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    CoreFunctionBinCache *cacheBin = cacheValue.binCache;
    memcpy_s(dataPtr, dataSize, cacheBin, dataSize);
}

void CalcTopoSize(const DeviceAgentTask *task, size_t &dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    CoreFunctionTopoCache *cacheTopo = cacheValue.topoCache;
    dataSize = cacheTopo->dataSize + sizeof(uint64_t); // datasize字段头也一起加上
    return;
}

void DumpTopoData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    CoreFunctionTopoCache *cacheTopo = cacheValue.topoCache;
    memcpy_s(dataPtr, dataSize, cacheTopo, dataSize);
}

void CalcInvokeOffsetSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = 0;
    if (!task->compileInfo.invokeArgsOffset.empty()) {
        dataSize = task->compileInfo.invokeArgsOffset.size() *
                   task->compileInfo.invokeArgsOffset.at(0).size() * sizeof(uint64_t);
    }
}

void DumpInvokeOffsetData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    (void) dataSize;
    size_t tmpOffset = 0;
    auto &invokeArgsOffsetVec = task->compileInfo.invokeArgsOffset;
    for (size_t i = 0; i < invokeArgsOffsetVec.size(); i++) {
        size_t copySize = invokeArgsOffsetVec[i].size() * sizeof(uint64_t);
        memcpy_s(dataPtr + tmpOffset, copySize, invokeArgsOffsetVec[i].data(), copySize);
        tmpOffset += copySize;
    }
}

void CalcInvokeIndexSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = 0;
    if (!task->compileInfo.invokeTensorsIdx.empty()) {
        dataSize = task->compileInfo.invokeTensorsIdx.size() *
                   task->compileInfo.invokeTensorsIdx.at(0).size() * sizeof(uint64_t);
    }
}

void DumpInvokeIndexData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    (void) dataSize;
    size_t tmpOffset = 0;
    auto &invokeTensorIdxVec = task->compileInfo.invokeTensorsIdx;
    for (size_t i = 0; i < invokeTensorIdxVec.size(); i++) {
        size_t copySize = invokeTensorIdxVec[i].size() * sizeof(int64_t);
        memcpy_s(dataPtr + tmpOffset, copySize, invokeTensorIdxVec[i].data(), copySize);
        tmpOffset += copySize;
    }
}

void CalcTensorInfoSize(const DeviceAgentTask *task, size_t &dataSize) {
    dataSize = task->compileInfo.coreTensorInfoVec.size() * sizeof(TensorInfo);
    return;
}

void DumpTensorInfoData(const DeviceAgentTask *task, uint8_t *dataPtr, size_t dataSize) {
    memcpy_s(dataPtr, dataSize, task->compileInfo.coreTensorInfoVec.data(), dataSize);
}

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
                                              paraOffset.isTensorParam);
            paraOffsetVec.push_back(paraOffsetInfo);
        }
    }
    memcpy_s(dataPtr, dataSize, paraOffsetVec.data(), dataSize);
}

void CalcFuncWSSize(const DeviceAgentTask *task, size_t &dataSize) {
    CacheValue cacheValue = task->GetFuncCacheValue().value();
    dataSize = (cacheValue.header.coreFunctionNum + cacheValue.header.virtualFunctionNum) * sizeof(CoreFunctionWsAddr);
    return;
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
    for (uint64_t i = cacheValue.header.coreFunctionNum;
         i < cacheValue.header.coreFunctionNum + cacheValue.header.virtualFunctionNum; i++) {
        coreFunctionWsOffset.push_back(CoreFunctionWsAddr(0, 0, 0xFFFFFFF, topoOffset[i], 0, 0));
    }
    memcpy_s(dataPtr, dataSize, coreFunctionWsOffset.data(), dataSize);
}

struct BinDataProc {
    BinDataType type;
    CalcBinDataSize calcFunc;
    DumpBinData dumpFunc;
};

BinDataProc g_binDataProcVec[static_cast<uint32_t>(BinDataType::END)] = {
    {BinDataType::READY_STATUS,        CalcReadyStateSize,   DumpReadyStateData},
    {BinDataType::READY_AIC_CORE_FUNC, CalcAicReadyFuncSize, DumpAicReadyFuncData},
    {BinDataType::READY_AIV_CORE_FUNC, CalcAivReadyFuncSize, DumpAivReadyFuncData},
    {BinDataType::CACHE_HEADER,        CalcCacheHeaderSize,  DumpCacheHeaderData},
    {BinDataType::CCE_BIN,             CalcCCEBinSize,       DumpCCEBinData},
    {BinDataType::TOPO,                CalcTopoSize,         DumpTopoData},
    {BinDataType::INVOKE_OFFSET_TABLE, CalcInvokeOffsetSize, DumpInvokeOffsetData},
    {BinDataType::INVODE_TENSOR_INDEX, CalcInvokeIndexSize,  DumpInvokeIndexData},
    {BinDataType::INVODE_TENSOR_INFO,  CalcTensorInfoSize,   DumpTensorInfoData},
    {BinDataType::INVOKE_PARA_OFFSET,  CalcParaOffsetSize,   DumpParaOffsetData},
    {BinDataType::CORE_FUNC_WS_ADDR,   CalcFuncWSSize,       DumpFuncWSSizeData}
};
}

std::string MachineDump::PrepareBinPath() {
    constexpr int size = 1024;
    char dumpBuf[size] = {};
    char *cwd = getcwd(dumpBuf, size);
    if (cwd == nullptr) {
        ALOG_ERROR_F("failed to call getcwd()");
        return "";
    }
    std::string outputPath = std::string(cwd) + "/DumpBin";
    struct stat st = {};
    bool outPathExist = stat(outputPath.c_str(), &st) == 0;
    if (outPathExist) {
        if (!S_ISDIR(st.st_mode)) {
            ALOG_ERROR_F("%s is not a directory!", outputPath.c_str());
            return "";
        }
    } else {
        if (mkdir(outputPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
            ALOG_ERROR_F("failed to call mkdir to create %s", outputPath.c_str());
            return "";
        }
    }
    return outputPath;
}

void MachineDump::InitTaskBinData(const DeviceAgentTask *deviceAgentTask, DeviceTaskBin &taskBin) {
    auto binData = &taskBin.baseArgs;
    binData->opaque = 0;
    binData->taskId = deviceAgentTask->compileTask->GetTaskId();
    taskBin.deviceTask.coreFunctionCnt = deviceAgentTask->compileInfo.coreFunctionCnt;
    taskBin.deviceTask.coreFuncData.stackWorkSpaceSize = deviceAgentTask->compileInfo.workSpaceStackSize;
    taskBin.deviceTask.coreFuncData.stackWorkSpaceAddr = deviceAgentTask->compileInfo.invokeParaWorkSpaceSize;
    ALOG_DEBUG_F("Dump FunctionCnt[%lu] taskId[%lu] WorkSpaceSize[%lu].", taskBin.deviceTask.coreFunctionCnt,
        binData->taskId, taskBin.deviceTask.coreFuncData.stackWorkSpaceSize);
}

void MachineDump::DumpJsonFile(const DeviceAgentTask *deviceAgentTask, const std::string &dumpFileName,
                               const std::string &dumpPath) {
    ALOG_DEBUG_F("Workspace size of task is [%lu].", deviceAgentTask->GetWorkSpaceSize());
    std::string jsonFile = dumpPath + "/" + dumpFileName + ".json";
    std::ofstream file(jsonFile);
    Json binJson;
    binJson["binFileName"] = "kernel";
    binJson["binFileSuffix"] = ".o";
    binJson["subKernelBin"] = dumpFileName + ".o";
    binJson["kernelName"] = "ast_main_0";
    binJson["coreType"] = "MIX";
    binJson["magic"] = "RT_DEV_BINARY_MAGIC_ELF";
    binJson["dynamicParamMode"] = "folded_with_desc";
    binJson["workspace"] = {
        {"num", 1},
        {"size", {deviceAgentTask->GetWorkSpaceSize() == 0 ? 1 : deviceAgentTask->GetWorkSpaceSize()}},
        {"type", {0}}
    };

    file << binJson.dump(NUM_FOUR) << std::endl;
    file.close();
    ALOG_INFO_F("Json file[%s] has been dumped.", jsonFile.c_str());
}

bool MachineDump::DumpBinAndJsonToFile(const DeviceAgentTask *deviceAgentTask,
    const DeviceTaskBin* binData, const size_t allSize, const std::string &dumpFileName, const std::string &dumpPath) {
    ALOG_INFO_F("Try to dump bin and json file[%s] in path[%s].", dumpFileName.c_str(), dumpPath.c_str());
    // dump bin file
    std::string binFile = dumpPath + "/" + dumpFileName + ".o";
    if (!DumpFile(reinterpret_cast<const char *>(binData), allSize, binFile)) {
        ALOG_ERROR_F("Failed to dump file %s.", binFile.c_str());
        return false;
    }
    ALOG_INFO_F("Bin file[%s] has been dumped.", binFile.c_str());
    DumpJsonFile(deviceAgentTask, dumpFileName, dumpPath);
    ALOG_INFO_F("Finish dump bin and json file.");
    return true;
}

void MachineDump::GetDumpBinData(const DeviceAgentTask *deviceAgentTask, std::vector<uint8_t> &binData) {
    if (deviceAgentTask == nullptr) {
        return;
    }
    DeviceTaskBin deviceTaskBin;
    InitTaskBinData(deviceAgentTask, deviceTaskBin);

    size_t totalSize = sizeof(DeviceTaskBin);
    for (size_t i = 0; i < static_cast<size_t>(BinDataType::END); ++i) {
        const auto &dataProc = g_binDataProcVec[i];
        size_t dataSize = 0;
        dataProc.calcFunc(deviceAgentTask, dataSize);
        deviceTaskBin.dataSize[i] = dataSize;
        deviceTaskBin.dataOffset[i] = totalSize;
        totalSize += MemSizeAlign(dataSize);
        ALOG_DEBUG_F("Calculate data size[%zu], offset[%zu].", deviceTaskBin.dataSize[i], deviceTaskBin.dataOffset[i]);
    }
    ALOG_DEBUG_F("Total size[%zu].", totalSize);
    binData.resize(totalSize);
    // copy DeviceTaskBin
    memcpy_s(binData.data(), sizeof(DeviceTaskBin), &deviceTaskBin, sizeof(DeviceTaskBin));
    for (size_t i = 0; i < static_cast<size_t>(BinDataType::END); ++i) {
        const auto &dataProc = g_binDataProcVec[i];
        dataProc.dumpFunc(deviceAgentTask, binData.data() + deviceTaskBin.dataOffset[i], deviceTaskBin.dataSize[i]);
    }
}

bool MachineDump::DumpASTBinData(const DeviceAgentTask *deviceAgentTask,
                                 const std::string &dumpFileName, const std::string &dumpPath) {
    ALOG_DEBUG_F("Try to dump bin file[%s] at path[%s].", dumpFileName.c_str(), dumpPath.c_str());
    std::string realDumpPath = RealPath(dumpPath);
    if (realDumpPath.empty()) {
        ALOG_DEBUG_F("Dump path[%s] is not existed, try to create dir.", dumpPath.c_str());
        if (!CreateMultiLevelDir(dumpPath)) {
            return false;
        }
        realDumpPath = RealPath(dumpPath);
    }
    std::vector<uint64_t> dataSizes;
    std::vector<uint64_t> alignSizes;
    size_t dataSize = 0;
    size_t alignSize = 0;
    size_t allSize = sizeof(DeviceTaskBin);
    for (size_t i = 0; i < static_cast<size_t>(BinDataType::END); ++i) {
        const auto &dataProc = g_binDataProcVec[i];
        dataProc.calcFunc(deviceAgentTask, dataSize);
        dataSizes.emplace_back(dataSize);
        alignSize = MemSizeAlign(dataSize); // device地址需要考虑对齐，这里按照32对齐下
        alignSizes.emplace_back(alignSize);
        allSize += alignSize;
        ALOG_DEBUG_F("Clac data size[%zu] align size[%zu].", dataSize, alignSize);
    }

    DeviceTaskBin *binData = reinterpret_cast<DeviceTaskBin *>(new (std::nothrow) uint8_t[allSize]);
    if (binData == nullptr) {
        return false;
    }
    InitTaskBinData(deviceAgentTask, *binData);

    // 1.base data
    size_t offset = sizeof(DeviceTaskBin);
    uint8_t *basePtr = reinterpret_cast<uint8_t *>(binData);
    for (size_t i = 0; i < static_cast<size_t>(BinDataType::END); ++i) {
        binData->dataOffset[i] = offset;
        binData->dataSize[i] = dataSizes[i];
        const auto &dataProc = g_binDataProcVec[i];
        dataProc.dumpFunc(deviceAgentTask, basePtr + offset, dataSizes[i]);
        offset += alignSizes[i];
        ALOG_DEBUG_F("Dump data offset[%zu].\n", offset);
    }

    bool ret = DumpBinAndJsonToFile(deviceAgentTask, binData, allSize, dumpFileName, realDumpPath);
    delete[] binData;
    return ret;
}
} // namespace npu::tile_fwk
