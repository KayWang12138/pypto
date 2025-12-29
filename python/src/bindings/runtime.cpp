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
 * \file runtime.cpp
 * \brief
 */

#include "pybind_common.h"

#include <utility>
#include <vector>
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/runtime/device_launcher_binding.h"
#include "machine/runtime/emulation_launcher.h"

#include <cstring>
#include <mutex>
#include <map>

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
#pragma pack(push, 8)
struct Mc2ServerCfg {
    uint32_t version = 0;
    uint8_t debugMode = 0;
    uint8_t sendArgIndex = 0;
    uint8_t recvArgIndex = 0;
    uint8_t commOutArgIndex = 0;
    uint8_t reserved[8] = {};
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2HcommCfg {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    char reserved[13] = {};
    char groupName[128] = {};
    char algConfig[128] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};
#pragma pack(pop)

struct Mc2CommConfig {
    uint32_t version;
    uint32_t hcommCnt;
    struct Mc2ServerCfg serverCfg;
    struct Mc2HcommCfg hcommCfg;
};

extern "C" int HcclAllocComResourceByTiling(void* comm, void *stream, void *mc2Tiling, void **commContext);

int32_t MakeMc2TilingStruct(struct Mc2CommConfig &commConfig, const std::string &groupName)
{
    constexpr uint32_t version = 2;
    constexpr uint32_t hcommCnt = 1;
    constexpr uint32_t opTypeAllToAll = 6; // numeric representation of AlltoAll
    const char *algConfig = "AllGather=level0:ring";
    constexpr uint32_t arraySize = 128;

    commConfig.version = version;
    commConfig.hcommCnt = hcommCnt;
    commConfig.hcommCfg.skipLocalRankCopy = 0;
    commConfig.hcommCfg.skipBufferWindowCopy = 0;
    commConfig.hcommCfg.stepSize = 0;
    commConfig.hcommCfg.opType = opTypeAllToAll;
    
    std::strncpy(commConfig.hcommCfg.groupName, groupName.c_str(), arraySize - 1);
    commConfig.hcommCfg.groupName[arraySize - 1] = '\0';
    
    std::strncpy(commConfig.hcommCfg.algConfig, algConfig, arraySize - 1);
    commConfig.hcommCfg.algConfig[arraySize - 1] = '\0';

    return 0;
}

std::mutex g_ctxMutex;
std::map<uint64_t, uint64_t> g_hcclContextCache;

} // namespace

namespace pypto {

DeviceTensorData CopyToHost(const DeviceTensorData &tensorData) {
    return CopyDevToHost(tensorData);
}

void SetVerifyData(const std::vector<DeviceTensorData> &inputs,
                   const std::vector<DeviceTensorData> &outputs,
                   const std::vector<DeviceTensorData> &goldens) {
    ProgramData::GetInstance().Reset();
    for (size_t i = 0; i < inputs.size(); i++) {
        auto rawData = RawTensorData::CreateTensor(
            inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t *)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(
            outputs[i].GetDataType(), outputs[i].GetShape());
        ProgramData::GetInstance().AppendOutput(rawData);
    }
    for (size_t i = 0; i < goldens.size(); i++) {
        if (goldens[i].GetAddr() == 0) {
            ProgramData::GetInstance().AppendGolden(nullptr);
        } else {
            auto rawData = RawTensorData::CreateTensor(
            goldens[i].GetDataType(), goldens[i].GetShape(), (uint8_t *)goldens[i].GetAddr());
            ProgramData::GetInstance().AppendGolden(rawData);
        }
    }
}

std::string DeviceRunOnceDataFromHost(
    const std::vector<DeviceTensorData> &inputs, const std::vector<DeviceTensorData> &outputs) {
    ProgramData::GetInstance().Reset();
    Function *func = Program::GetInstance().GetLastFunction();
    if (!func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC, GraphType::TENSOR_GRAPH)) {
        return "Invalid function format";
    }

    auto attr = func->GetDyndevAttribute();
    if (attr == nullptr) {
        return "Invalid function format";
    }

    auto inputSize = attr->startArgsInputLogicalTensorList.size();
    auto outputSize = attr->startArgsOutputLogicalTensorList.size();
    if (inputSize != inputs.size() || outputSize != outputs.size()) {
        return "mismatch input/output";
    }

    for (size_t i = 0; i < inputs.size(); i++) {
        auto rawData = RawTensorData::CreateTensor(inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t *)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(outputs[i].GetDataType(), outputs[i].GetShape());
        ProgramData::GetInstance().AppendOutput(rawData);
    }

    if (config::GetOption<bool>(PROFILE_ENABLE) && EmulationLauncher::EmulationRunOnce(func) != 0) {
        return "emulation run failed";
    }

    if (DeviceRunOnce(func) != 0) {
        return "device run failed";
    }

    for (size_t i = 0; i < outputs.size(); i++) {
        auto output = ProgramData::GetInstance().GetOutputData(i);
        StringUtils::DataCopy(outputs[i].GetAddr(), output->GetDataSize(), output->data(), output->GetDataSize());
    }

    if (HasInplaceArgs(Program::GetInstance().GetLastFunction()) || outputs.size() == 0) {
        for (size_t i = 0; i < inputs.size(); i++) {
            auto input = ProgramData::GetInstance().GetInputData(i);
            StringUtils::DataCopy(inputs[i].GetAddr(), input->GetDataSize(), input->data(), input->GetDataSize());
        }
    }
    return "";
}

std::string OperatorDeviceRunOnceDataFromDevice([[maybe_unused]] py::int_ pythonOperatorPython,
    [[maybe_unused]] const std::vector<DeviceTensorData> &inputs, [[maybe_unused]] const std::vector<DeviceTensorData> &outputs,
    [[maybe_unused]] py::int_ incomingStreamPython, [[maybe_unused]] py::int_ workspaceData) {
#ifdef BUILD_WITH_CANN
    auto opAddr = static_cast<uintptr_t>(pythonOperatorPython);
    if (opAddr == 0) {
        return "invalid operator";
    }

    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    Function *func = op->GetFunction();
    if (!func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC, GraphType::TENSOR_GRAPH)) {
        return "Invalid function format";
    }

    auto attr = func->GetDyndevAttribute();
    if (attr == nullptr) {
        return "Invalid function format";
    }

    auto inputSize = attr->startArgsInputLogicalTensorList.size();
    auto outputSize = attr->startArgsOutputLogicalTensorList.size();
    if (inputSize != inputs.size() || outputSize != outputs.size()) {
        return "mismatch input/output";
    }

    if (config::GetOption<bool>(PROFILE_ENABLE)) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        if (EmulationLauncher::EmulationLaunchDeviceTensorData(func, inputs, outputs, config) != 0) {
            return "emulation run failed";
        }
    }

    auto incomingStream = static_cast<uintptr_t>(incomingStreamPython);
    if (incomingStream == 0) {
        return "invalid incoming stream";
    }

    auto aicoreStream = incomingStream;
    auto aicpuStream = DeviceGetAicpuStream();
    auto workspaceDataAddr = static_cast<uintptr_t>(workspaceData);
    auto config = DeviceLauncherConfig::CreateConfigWithWorkspaceAddr(workspaceDataAddr);
    try {
        std::cout << "[PyPTO] Config Dump Start" << std::endl;
        std::cout << config::Dump() << std::endl;
        std::cout << "[PyPTO] Config Dump End" << std::endl;

        auto hcclHandle = config::GetDistributedOption<uint64_t>("hccl_context");
        printf("[PyPTO] Debug: hcclHandle=%lu\n", hcclHandle);
        if (hcclHandle != 0) {
            std::lock_guard<std::mutex> lock(g_ctxMutex);
            if (g_hcclContextCache.find(hcclHandle) != g_hcclContextCache.end()) {
                 config.hcclContext.push_back(g_hcclContextCache[hcclHandle]);
                 printf("[PyPTO] Debug: Used cached context\n");
            } else {
                auto groupName = config::GetDistributedOption<std::string>("hccl_context_name");
                printf("[PyPTO] Debug: groupName=%s\n", groupName.c_str());
                if (!groupName.empty()) {
                    struct Mc2CommConfig commConfig = {};
                    if (MakeMc2TilingStruct(commConfig, groupName) == 0) {
                         void* commContext = nullptr;
                         // Using aicoreStream for resource allocation might be correct if it's the execution stream
                         int ret = HcclAllocComResourceByTiling((void*)hcclHandle, (void*)aicoreStream, &commConfig, &commContext);
                         printf("[PyPTO] Debug: Alloc ret=%d, commContext=%p\n", ret, commContext);
                         if (ret == 0 && commContext != nullptr) {
                             uint64_t contextVal = (uint64_t)commContext;
                             g_hcclContextCache[hcclHandle] = contextVal;
                             config.hcclContext.push_back(contextVal);
                         }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        printf("[PyPTO] Error in distributed setup: %s\n", e.what());
    } catch (...) {
        printf("[PyPTO] Unknown error in distributed setup\n");
    }

    int rc =
        ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(op, inputs, outputs, aicpuStream, aicoreStream, false, config);
    if (rc < 0) {
        return "device run failed";
    }
#endif
    return "";
}

uint64_t GetWorkSpaceSize(uintptr_t opAddr, const std::vector<DeviceTensorData> &inputs,
    const std::vector<DeviceTensorData> &outputs) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    if (op) {
        return op->GetWorkSpaceSize(inputs, outputs);
    }
    return 0;
}

std::string OperatorDeviceSynchronize(py::int_ incomingStreamPython) {
    auto incomingStream = static_cast<uintptr_t>(incomingStreamPython);
    if (incomingStream == 0) {
        return "invalid incoming stream";
    }

    auto aicpuStream = incomingStream;
    auto aicoreStream = DeviceGetAicoreStream();
    int rc = DeviceSynchronize(aicpuStream, aicoreStream);
    if (rc < 0) {
        return "device sync failed";
    }
    return "";
}

void DeviceInit() {
    DeviceLauncherInit();
}

void DeviceFini() {
    DeviceLauncherFini();
}

uintptr_t OperatorBegin() {
    ExportedOperator *op = ExportedOperatorBegin();
    auto opAddr = reinterpret_cast<uintptr_t>(op);
    return opAddr;
}

std::string OperatorEnd(uintptr_t opAddr) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    ExportedOperatorEnd(op);

    return "";
}

std::string BuildCache(uintptr_t opAddr, const std::vector<DeviceTensorData> &inputList,
        const std::vector<DeviceTensorData> &outputList) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);

    if (config::GetRuntimeOption<int64_t>(CFGCACHE_DEVICE_TASK_NUM) != 0) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        if (EmulationLauncher::BuildControlFlowCache(op->GetFunction(), inputList, outputList, config) != 0) {
            return "control flow cache failed";
        }
    }

    return "";
}

void BindRuntime(py::module &m) {
    m.def("DeviceInit", &DeviceInit);
    m.def("DeviceFini", &DeviceFini);
    m.def("DeviceRunOnceDataFromHost", &DeviceRunOnceDataFromHost);
    m.def("OperatorDeviceRunOnceDataFromDevice", &OperatorDeviceRunOnceDataFromDevice);
    m.def("OperatorDeviceSynchronize", &OperatorDeviceSynchronize);
    m.def("GetWorkSpaceSize", &GetWorkSpaceSize);
    m.def("OperatorBegin", OperatorBegin);
    m.def("OperatorEnd", OperatorEnd);
    m.def("SetVerifyData", &SetVerifyData);
    m.def("BuildCache", BuildCache);
    m.def("CopyToHost", &CopyToHost);

    py::class_<DeviceTensorData>(m, "DeviceTensorData")
        .def(py::init<DataType, uintptr_t, const std::vector<int64_t> &>(), py::arg("dtype"), py::arg("addr"),
            py::arg("shape"))
        .def("GetDataPtr", &DeviceTensorData::GetAddr)
        .def("GetShape", &DeviceTensorData::GetShape)
        .def("GetDataType", &DeviceTensorData::GetDataType);
}
} // namespace pypto
