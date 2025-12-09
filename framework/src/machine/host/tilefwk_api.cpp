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
 * \file tilefwk_api.cpp
 * \brief
 */

#include "interface/inner/tilefwk/tilefwk_api.h"

#include "tilefwk/tilefwk.h"
#include "tilefwk/op_registry.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/machine/host/host_machine.h"
#include "machine/host/backend.h"
#include "machine/host/device_agent_task.h"
#include "machine/dump/kernel_dump_utils.h"
#include "machine/host/machine_compiler.h"
#include "machine/cache_manager/cache_manager.h"
#include "machine/platform/platform_manager.h"
#include "tilefwk/comm_group_recorder.h"

namespace npu::tile_fwk {
int32_t TileFwkInit(const std::string &socVersion) {
    (void)PlatformManager::Instance().Initialize(socVersion);
    Program::GetInstance().Reset();
    HostMachine::GetInstance().Init(HostMachineMode::API);
    return 0;
}

int32_t TileFwkBeginFunction(const std::string &funcName, const std::vector<std::reference_wrapper<const Tensor>> &opArgs) {
    return Program::GetInstance().BeginFunction(funcName, FunctionType::STATIC, GraphType::TENSOR_GRAPH, opArgs);
}

int32_t TileFwkEndFunction(const bool isWaitTaskFinished) {
    return Program::GetInstance().EndFunction(isWaitTaskFinished);
}

/* 返回compile handle */
void *TileFwkCompile() {
    (void)CacheManager::Instance().Initialize();
    MachineTask *task = HostMachine::GetInstance().Compile();
    auto deviceAgentTask = new DeviceAgentTask(task); // need free somewhere
    auto function = deviceAgentTask->compileTask->GetFunction();
    deviceAgentTask->compileInfo.distTilingManager = function->GetDistTilingManager();
    deviceAgentTask->compileInfo.commGroups = Distributed::CommGroupRecorder::GetInstance().Output();
    std::string kernelPath;
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
        auto &cache = Program::GetInstance().GetFunctionCache();
        (void)GenCode(deviceAgentTask->compileTask, deviceAgentTask->compileInfo.invokeParaOffset, cache, kernelPath);
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
                    config::GetHostConfig(KEY_DUMP_BIN_AND_JSON_PATH, ""), kernelPath)) {
            ALOG_ERROR_F("dump ast bin failed");
            return nullptr;
        }
    }

    ALOG_INFO("End compile: func name = ", function->GetRawName());
    return reinterpret_cast<void *>(deviceAgentTask);
}

int32_t TileFwkGetWorkspaceSize(const void *handle, uint64_t *workspaceSize) {
    *workspaceSize = (reinterpret_cast<const DeviceAgentTask *>(handle))->GetWorkSpaceSize();
    return 0;
}

void TileFwkFreeHandle(const void *handle) {
    /* handle 当前在AstRun结束后释放， 此接口暂时后续保留扩展使用 */
    (void)handle;
}

void TileFwkFinalize() {
    Program::GetInstance().Reset();
}

void TileFwkSetVecTileShapes(const std::vector<int64_t> &tileShape) {
    TileShape::Current().SetVecTile(tileShape);
}

void TileFwkSetCubeTileShapes(
    const std::array<int64_t, 2> &m, const std::array<int64_t, 0x3> &k, const std::array<int64_t, 2> &n) {
    TileShape::Current().SetCubeTile(m, k, n);
}

void TileFwkAssign(Tensor &dst, const Tensor &src) {
    dst = src;
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
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON, true);
    config::SetHostConfig(KEY_DUMP_BIN_AND_JSON_PATH, dumpPath);
    config::SetHostConfig(KEY_DUMP_KERNEL_NAME, kernelName);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetHostOption(ONLY_CODEGEN, true);

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
    TileShape::Current().SetDistTile(row, col, rank);
}

void TileFwkSpecifyStaticRankId(int rankId) {
    TileShape::Current().SetDistRankId(rankId);
}
} // namespace Distributed
} // namespace npu::tile_fwk
