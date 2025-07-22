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
 * \file tilefwk_api.cpp
 * \brief
 */

#include "interface/inner/tilefwk/tilefwk_api.h"
#include "backend.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "machine/host/machine_agent.h"
#include "machine/dump/kernel_dump_utils.h"
#include "machine/host/machine_compiler.h"
#include "machine/cache_manager/cache_manager.h"
#include "interface/platform/platform_manager.h"
#include "interface/registry/tile_fwk_op_registry.h"

namespace npu::tile_fwk {
int32_t TileFwkInit(const std::string &socVersion) {
    (void)PlatformManager::Instance().Initialize(socVersion);
    Program::GetInstance().HostMachineInit(HostMachineMode::API);
    return 0;
}

int32_t TileFwkBeginFunction(const std::string &funcName, const std::vector<std::reference_wrapper<Tensor>> &opArgs) {
    return Program::GetInstance().BeginFunction(funcName, FunctionType::STATIC, GraphType::TENSOR_GRAPH, opArgs);
}

int32_t TileFwkEndFunction(const bool isWaitTaskFinished) {
    return Program::GetInstance().EndFunction(isWaitTaskFinished);
}

void *TileFwkCompile() {
    return Program::GetInstance().Compile();
}

int32_t TileFwkGetWorkspaceSize(const void *handle, uint64_t *workspaceSize) {
    *workspaceSize = Program::GetInstance().GetWorkSpaceSize(handle);
    return 0;
}

int32_t TileFwkRunAsync(void *handle, const void *workspace, const void *stream, const std::vector<void *> &opArgs,
    const std::vector<size_t> &prefetchSizes) {
    return Program::GetInstance().RunAsync(stream, workspace, handle, opArgs, prefetchSizes);
}

void TileFwkFreeHandle(const void *handle) {
    /* handle 当前在AstRun结束后释放， 此接口暂时后续保留扩展使用 */
    (void)handle;
}

void TileFwkFinalize() {
    Program::GetInstance().Reset();
}

void TileFwkSetVecTileShapes(const std::vector<int> &tileShape) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
}

void TileFwkSetCubeTileShapes(const std::array<int, 2> &m, const std::vector<int> &k, const std::array<int, 2> &n) {
    Program::GetInstance().GetTileShape().SetCubeTileShapes(m, k, n);
}

void TileFwkAssign(Tensor &dst, const Tensor &src) {
    dst = src;
}

/* 返回compile handle */
void *Program::Compile() {
    (void)CacheManager::Instance().Initialize();
    MachineTask *task = hostMachine_.Compile();
    auto deviceAgentTask = new DeviceAgentTask(task); // need free somewhere
    auto function = deviceAgentTask->compileTask->GetFunction();
    deviceAgentTask->compileInfo.distTilingManager = function->GetDistTilingManager();
    deviceAgentTask->compileInfo.commGroups = Program::GetInstance().GetCommGroupRecorder().Output();
    // if disk cache is matched, try to recover task info and bin
    if (task->GetCacheReuseType() == CacheReuseType::Bin) {
        if (!CacheManager::Instance().RecoverTask(task->GetCacheKey(), deviceAgentTask)) {
            ALOG_ERROR_F("Fail to recover task from cache[%s].", task->GetCacheKey().c_str());
            return nullptr;
        }
    } else {
        /* calc workspace size and every sub function invoke entry para offset */
        CalcFunctionInvokeWorkespace(nullptr, function, deviceAgentTask->compileInfo);

        deviceAgentTask->compileInfo.distTilingManager = function->GetDistTilingManager();
        deviceAgentTask->compileInfo.PrintDistributed();

        deviceAgentTask->compileInfo.workSpaceStackSize = function->GetStackWorkespaceSize();
        auto &cache = hostMachine_.GetFunctionCache();
        (void)GenCode(deviceAgentTask->compileTask, deviceAgentTask->compileInfo.invokeParaOffset, cache);
        /* finish compile add function cache */
        cache.Insert(function->GetFunctionHash(), *function);
        deviceAgentTask->SetFunctionCache(cache.Get(function->GetFunctionHash()));
        if (function->IsFunctionType(FunctionType::STATIC)) {
            deviceAgentTask->Validate();
            deviceAgentTask->UpdateCompileInfo();
        }
        // save compile result on disk
        CacheManager::Instance().SaveTaskFile(deviceAgentTask);
    }

    if (config::GetHostConfig(KEY_DUMP_BIN_AND_JSON, false)) {
        if (!KernelDumpUtils::DumpKernelFile(deviceAgentTask,
                    config::GetHostConfig(KEY_DUMP_KERNEL_NAME, ""),
                    config::GetHostConfig(KEY_DUMP_BIN_AND_JSON_PATH, ""))) {
            ALOG_ERROR_F("dump ast bin failed");
            return nullptr;
        }
    }

    ALOG_INFO("End compile: func name = ", function->GetRawName());
    return reinterpret_cast<void *>(deviceAgentTask);
}

uint64_t Program::GetWorkSpaceSize(const void *handle) {
    return (reinterpret_cast<const DeviceAgentTask *>(handle))->GetWorkSpaceSize();
}

int Program::RunAsync(const void *stream, const void *workSpaceGmAddr, void *handle,
    const std::vector<void *> &opOriginArgs, const std::vector<size_t> &argsSize) {
    ALOG_INFO("Program::Run stream = ", stream);
    DeviceAgentTask *deviceAgentTask = reinterpret_cast<DeviceAgentTask *>(handle);
    int ret = Run(stream, workSpaceGmAddr, deviceAgentTask, opOriginArgs, argsSize, true);
    ALOG_INFO("End run: func name = ", deviceAgentTask->compileTask->GetFunction()->GetRawName(), " ret = ", ret);
    currentFunctionPtr_ = &(currentFunctionPtr_->Parent()); // reset init function
    return 0;
}

extern "C" bool TileFwkCompileFatbin(const char *opType, const char *socVersion,
        const char *dumpPath, const char *kernelName) {
    ALOG_INFO_F("Start to compile fatbin for op type[%s], dump path and file name is [%s] and [%s].",
                opType, dumpPath, kernelName);
    (void)PlatformManager::Instance().Initialize(socVersion);
    // load op impl so
    void *opLibHandle = KernelDumpUtils::LoadTileFwkImplOpLib();
    std::vector<uint64_t> configKeys = OpImplRegistry::GetInstance().GetAllConfigKeys(opType);
    if (configKeys.empty()) {
        ALOG_INFO_F("Cannot find registered configKeys of op type[%s].", opType);
        KernelDumpUtils::FreeOpHandle(opLibHandle);
        return false;
    }

    FatbinHeadInfo fatbinHeadInfo;
    std::vector<char> fatbinBuffer;
    std::vector<JsonInfo> allBinJsonInfo;
    for (auto &configKey : configKeys) {
        std::string subKernelName = std::string(kernelName) + "_" + std::to_string(configKey);
        ALOG_INFO_F("Start to compile sub kernel[%s] for op type[%s] and configKey[%lu].",
                    subKernelName.c_str(), opType, configKey);
        if (!TileOpCompile(opType, configKey, subKernelName, dumpPath)) {
            KernelDumpUtils::FreeOpHandle(opLibHandle);
            return false;
        }
        ++fatbinHeadInfo.configKeyNum;
        fatbinHeadInfo.binOffsets.emplace_back(fatbinBuffer.size());
        fatbinHeadInfo.configKeyList.emplace_back(configKey);

        // parse sub kernel bin file
        std::string subKernelBinPath = std::string(dumpPath) + "/" + subKernelName + ".o";
        if (!KernelDumpUtils::GetBufferFromBinFile(subKernelBinPath, fatbinBuffer)) {
            KernelDumpUtils::FreeOpHandle(opLibHandle);
            return false;
        }
        // parse sub kernel json file
        std::string subKernelJsonPath = std::string(dumpPath) + "/" + subKernelName + ".json";
        JsonInfo subJsonInfo;
        subJsonInfo.configKey = configKey;
        if (!KernelDumpUtils::GetSubJsonInfo(subKernelJsonPath, subJsonInfo)) {
            KernelDumpUtils::FreeOpHandle(opLibHandle);
            return false;
        }
        allBinJsonInfo.emplace_back(subJsonInfo);
    }
    KernelDumpUtils::FreeOpHandle(opLibHandle);
    std::string fatbinBinFilePath = std::string(dumpPath) + "/" + std::string(kernelName) + ".o";
    if (!KernelDumpUtils::WriteBufferToFatbin(fatbinHeadInfo, fatbinBinFilePath, fatbinBuffer)) {
        return false;
    }

    std::string fatbinJsonFilePath = std::string(dumpPath) + "/" + std::string(kernelName) + ".json";
    KernelDumpUtils::WriteFatbinJson(allBinJsonInfo, fatbinJsonFilePath, kernelName);
    ALOG_INFO_F("Finish to compile fatbin, bin file[%s] and json file[%s].",
                fatbinBinFilePath.c_str(), fatbinJsonFilePath.c_str());
    return true;
}

/**
 * compile op for AscendCppBackend binary
 */
bool TileOpCompile(const std::string &opType, const uint64_t configKey, const std::string &kernelName,
    const std::string &dumpPath) {
    if (!Program::GetInstance().GetHostMachine().ForceEnableBackend()) {
        ALOG_WARN("Fail to init host machine backend.");
        return false;
    }

    Program::GetInstance().GetConfig().Reset();
    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON, true);
    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON_PATH, dumpPath);
    config::SetHostConfig(KEY_DUMP_KERNEL_NAME, kernelName);
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);

    OpImplFunc opFunc = OpImplRegistry::GetInstance().GetOpImplFunc(opType, configKey);
    if (opFunc == nullptr) {
        ALOG_WARN("Op impl func is not found.");
        return false;
    }
    opFunc(configKey);
    Program::GetInstance().Reset();
    return true;
}

namespace Distributed {
void TileFwkSetDistTileShapes(std::array<int, MAX_DIST_DIM_SIZE> row, std::array<int, MAX_DIST_DIM_SIZE> col,
    std::array<int, MAX_DIST_DIM_SIZE> rank) {
    Program::GetInstance().GetTileShape().SetDistTileShapes(row, col, rank);
}

void TileFwkSpecifyStaticRankId(int rankId) {
    Program::GetInstance().GetTileShape().SpecifyStaticRankId(rankId);
}
} // namespace Distributed
} // namespace npu::tile_fwk