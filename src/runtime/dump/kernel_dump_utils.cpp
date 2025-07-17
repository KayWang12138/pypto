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
 * \file kernel_dump_utils.cpp
 * \brief dump binary and kernel.o into one kernel
 */

#include "runtime/dump/kernel_dump_utils.h"
#include <climits>
#include "interface/utils/file_utils.h"
#include "tilefwk/function.h"
#include "interface/machine/host/host_machine.h"
#include "interface/operation/distributed/comm_barrier_manager.h"
#include "interface/program/program.h"
#include "runtime/dump/machine_dump.h"

namespace npu::tile_fwk {
namespace {
constexpr int64_t LEVEL_FOUR = 4;
constexpr int64_t MAX_BLOCK_NUM = 24;
const std::string KERNEL_FILE_PREFIX = "ast_op_";
const std::string KERNEL_BIN_FILE_SUFFIX = ".o";
const std::string KERNEL_JSON_FILE_SUFFIX = ".json";
const std::string AICORE_KERNEL_FILE_PATH = "../../src/runtime/kernel/kernel.o";

inline size_t DataSizeAlign(const size_t bytes, const uint32_t aligns = 32U) {
    const size_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}
}

bool KernelDumpUtils::DumpKernelFile(const DeviceAgentTask *deviceAgentTask, const std::string &kernelName, const std::string &dumpDirPath) {
    if (deviceAgentTask == nullptr || deviceAgentTask->GetFunction() == nullptr) {
        return false;
    }
    std::string realDumpDirPath = RealPath(dumpDirPath);
    if (realDumpDirPath.empty()) {
        if (!CreateMultiLevelDir(dumpDirPath)) {
            return false;
        }
        realDumpDirPath = RealPath(dumpDirPath);
    }
    std::string finalKernelName = kernelName.empty() ? deviceAgentTask->GetFunction()->GetMagicName() : kernelName;
    if (!DumpBinFile(deviceAgentTask, finalKernelName, realDumpDirPath)) {
        return false;
    }

    DumpJsonFile(deviceAgentTask, finalKernelName, realDumpDirPath);
    return true;
}

bool KernelDumpUtils::DumpBinFile(const DeviceAgentTask *deviceAgentTask, const std::string &kernelName, const std::string &dumpDirPath) {
    KernelHeader kernelHeader;
    size_t offset = sizeof(kernelHeader);
    // op binary
    kernelHeader.dataOffset[static_cast<size_t>(KernelContextType::OpBinary)] = offset;
    Function *function = deviceAgentTask->GetFunction();
    std::vector<uint8_t> opBinData;
    if (function->IsFunctionType(FunctionType::DYNAMIC) && function->GetDyndevAttribute()) {
        opBinData = function->GetDyndevAttribute()->devProgBinary;
    }

    if ((function->IsFunctionTypeAndGraphType({FunctionType::STATIC}, {GraphType::TENSOR_GRAPH, GraphType::TILE_GRAPH})) &&
        (function->BelongTo().GetLastFunction() == nullptr ||
         !function->BelongTo().GetLastFunction()->IsFunctionType(FunctionType::DYNAMIC))) {
        MachineDump::GetDumpBinData(deviceAgentTask, opBinData);
    }

    if (opBinData.empty()) {
        return false;
    }
    kernelHeader.dataSize[static_cast<size_t>(KernelContextType::OpBinary)] = opBinData.size();
    offset += DataSizeAlign(opBinData.size());

    // kernel.o
    std::vector<uint8_t> kernelBinData = LoadFile(AICORE_KERNEL_FILE_PATH);
    kernelHeader.dataOffset[static_cast<size_t>(KernelContextType::Kernel)] = offset;
    kernelHeader.dataSize[static_cast<size_t>(KernelContextType::Kernel)] = kernelBinData.size();
    offset += DataSizeAlign(kernelBinData.size());

    std::vector<uint8_t> binData(offset, 0);
    memcpy_s(binData.data(), sizeof(kernelHeader), &kernelHeader, sizeof(kernelHeader));
    memcpy_s(binData.data() + kernelHeader.dataOffset[static_cast<size_t>(KernelContextType::OpBinary)],
             kernelHeader.dataSize[static_cast<size_t>(KernelContextType::OpBinary)], opBinData.data(),
             kernelHeader.dataSize[static_cast<size_t>(KernelContextType::OpBinary)]);
    memcpy_s(binData.data() + kernelHeader.dataOffset[static_cast<size_t>(KernelContextType::Kernel)],
             kernelHeader.dataSize[static_cast<size_t>(KernelContextType::Kernel)], kernelBinData.data(),
             kernelHeader.dataSize[static_cast<size_t>(KernelContextType::Kernel)]);
    std::string binFilePath = dumpDirPath + "/" + kernelName + KERNEL_BIN_FILE_SUFFIX;
    return DumpFile(binData, binFilePath);
}

void KernelDumpUtils::DumpJsonFile(const DeviceAgentTask *deviceAgentTask, const std::string &kernelName, const std::string &dumpDirPath) {
    std::string jsonFilePath = dumpDirPath + "/" + kernelName + KERNEL_JSON_FILE_SUFFIX;
    std::ofstream file(jsonFilePath);
    Json binJson;
    binJson["binFileName"] = kernelName;
    binJson["binFileSuffix"] = KERNEL_BIN_FILE_SUFFIX;
    binJson["kernelName"] = "ast_main_0";
    binJson["coreType"] = "MIX";
    binJson["blockDim"] = MAX_BLOCK_NUM;
    binJson["magic"] = "RT_DEV_BINARY_MAGIC_ELF";
    binJson["dynamicParamMode"] = "floded_with_desc";
    binJson["workspace"] = {
        {"num", 1},
        {"size", {deviceAgentTask->GetWorkSpaceSize() == 0 ? 1 : deviceAgentTask->GetWorkSpaceSize()}},
        {"type", {0}}
    };
    file << binJson.dump(LEVEL_FOUR) << std::endl;
    file.close();
}

bool KernelDumpUtils::GetBufferFromBinFile(const std::string &binFilePath, std::vector<char> &buffer) {
    char resolvedPath[PATH_MAX] = {0x00};
    if (realpath(binFilePath.c_str(), resolvedPath) == nullptr) {
        ALOG_INFO("path is invalid, val: ", binFilePath);
        return false;
    }
    std::ifstream ifStream(resolvedPath, std::ios::binary | std::ios::ate);
    if (!ifStream.is_open()) {
        ALOG_INFO("Open file failed, path is: ", binFilePath);
        return false;
    }
    try {
        std::streamsize bufferSize = ifStream.tellg();
        if (bufferSize <= 0) {
            ifStream.close();
            ALOG_INFO("Get stream failed");
            return false;
        }
        if (bufferSize > INT_MAX) {
            ifStream.close();
            ALOG_INFO("bufferSize is invalid.");
            return false;
        }
        ifStream.seekg(0, std::ios::beg);
        size_t curSize = buffer.size();
        size_t increSize = static_cast<size_t>(bufferSize);
        buffer.resize(curSize + increSize);
        ifStream.read(&buffer[curSize], bufferSize);
        ifStream.close();
    } catch (const std::ifstream::failure &e) {
        ifStream.close();
        std::cerr << "Error attr: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool KernelDumpUtils::WriteBufferToFatbin(FatbinHeadInfo &fatbinHeadInfo, const std::string &path,
        const std::vector<char> &fatbinBuffer) {
    std::ofstream fatbinFile(path, std::ios::binary);
    if (!fatbinFile.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }
    size_t headSize = sizeof(fatbinHeadInfo.configKeyNum);
    size_t binOffsetListSize = fatbinHeadInfo.binOffsets.size() * sizeof(size_t);
    size_t configKeyListSize = fatbinHeadInfo.configKeyList.size() * sizeof(uint64_t);
    size_t offset = headSize + binOffsetListSize + configKeyListSize;
    for (auto &i : fatbinHeadInfo.binOffsets) {
        i += offset;
    }
    std::vector<uint8_t> fatbinData(offset, 0);
    memcpy_s(fatbinData.data(), headSize, &fatbinHeadInfo.configKeyNum, headSize);
    memcpy_s(fatbinData.data() + headSize, configKeyListSize, fatbinHeadInfo.configKeyList.data(), configKeyListSize);
    memcpy_s(fatbinData.data() + headSize + configKeyListSize, binOffsetListSize,
             fatbinHeadInfo.binOffsets.data(), binOffsetListSize);
    fatbinFile.write(reinterpret_cast<const char*>(fatbinData.data()), fatbinData.size());
    fatbinFile.write(reinterpret_cast<const char*>(fatbinBuffer.data()), fatbinBuffer.size());
    if (!fatbinFile.good()) {
        std::cerr << "Error occurred during writing!" << std::endl;
    }
    ALOG_INFO("headInfoSize is: ", offset, ", fatbin size is: ", fatbinFile.tellp());
    fatbinFile.close();
    return true;
}

void KernelDumpUtils::WriteFatbinJson(const std::vector<JsonInfo> &allBinJsonInfo, const std::string &fatbinJsonPath,
        const std::string &binFileName) {
    std::ofstream jsonFile(fatbinJsonPath);
    Json fatbinJson;
    fatbinJson["binFileName"] = binFileName;
    fatbinJson["binFileSuffix"] = ".o";
    fatbinJson["coreType"] = "MIX";
    fatbinJson["kernelName"] = "ast_main_0";
    fatbinJson["magic"] = "RT_DEV_BINARY_MAGIC_ELF";
    fatbinJson["dynamicParamMode"] = "folded_with_desc";
    fatbinJson["blockDim"] = -1;
    int64_t workspaceSize = -1;
    Json kernelListJson;
    for (auto &binInfo : allBinJsonInfo) {
        Json kernelInfo;
        kernelInfo["configKey"] = binInfo.configKey;
        kernelInfo["blockDim"] = binInfo.blockDim;
        kernelInfo["kernelName"] = binInfo.kernelName;
        kernelInfo["workspaceSize"] = binInfo.workspaceSize;
        workspaceSize = binInfo.workspaceSize > workspaceSize ? binInfo.workspaceSize : workspaceSize;
        kernelListJson.emplace_back(kernelInfo);
    }
    fatbinJson["workspace"] = {{"num", 1}, {"size", workspaceSize}, {"type", {0}}};
    fatbinJson["kernelList"] = kernelListJson;
    fatbinJson["compileInfo"] = "";
    jsonFile << fatbinJson.dump(LEVEL_FOUR) << std::endl;
    jsonFile.close();
}

bool KernelDumpUtils::GetSubJsonInfo(const std::string &jsonPath, JsonInfo &kernelJsonInfo) {
    char resolvedPath[PATH_MAX] = {0x00};
    if (realpath(jsonPath.c_str(), resolvedPath) == nullptr) {
        ALOG_INFO("realpath failed, val is: ", jsonPath);
        return false;
    }
    Json jsonValue;
    std::ifstream ifs(resolvedPath);
    try {
        if (!ifs.is_open()) {
            return false;
        }
        ifs >> jsonValue;
        ifs.close();
    } catch (const std::exception &e) {
        ifs.close();
        return false;
    }
    try {
        kernelJsonInfo.blockDim = jsonValue.at("blockDim").get<int64_t>();
        kernelJsonInfo.kernelName = jsonValue.at("kernelName").get<std::string>();
        if (jsonValue.find("workspace") != jsonValue.end()) {
            Json workspaceValue = jsonValue["workspace"];
            kernelJsonInfo.workspaceSize = workspaceValue.at("size").get<std::vector<int64_t>>().at(0);
        }
    } catch (const std::exception &e) {
        std::cerr << "Json parse error: " << e.what() << ", json val is: " << jsonValue.dump() << std::endl;
        return false;
    }
    return true;
}

void GetEnv(const char *envName, std::string &envValue) {
    const size_t envValueMaxLen = 1024UL * 1024UL;
    const char *envTemp = std::getenv(envName);
    if ((envTemp == nullptr) || (strnlen(envTemp, envValueMaxLen) >= envValueMaxLen)) {
        ALOG_INFO("Env[%s] not found. \n", envName);
        return;
    }
    envValue = envTemp;
}

void KernelDumpUtils::LoadTileFwkOpLib(void *opLibHandle) {
    std::string tileFwkLibPath;
    GetEnv("TILE_FWK_OP_IMPL_PATH", tileFwkLibPath);
    if (tileFwkLibPath.empty()) {
        tileFwkLibPath = ".";
    }
    tileFwkLibPath += "/libtile_fwk_impl.so";
    opLibHandle = dlopen(tileFwkLibPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (opLibHandle == nullptr) {
        ALOG_INFO("Cannot dlopen libast_impl.so, ", dlerror());
    }
}

void KernelDumpUtils::FreeOpHandle(void *opLibHandle) {
    if (opLibHandle != nullptr) {
        (void)dlclose(opLibHandle);
    }
}
}
