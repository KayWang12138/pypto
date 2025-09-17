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
 * \file controller.cpp
 * \brief
 */

#include "pybind_common.h"

#include <utility>
#include <vector>

#include "interface/interpreter/raw_tensor_data.h"
#include "runtime/device_launcher_binding.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace pypto {

#ifdef ENABLE_BUILD_WITH_CANN

static RawTensorDataPtr HostCastPythonToNative(const std::shared_ptr<LogicalTensor> &tensor, py::object &pythonData) {
    DataType dataType = tensor->Datatype();
    const Shape &shape = tensor->GetShape();

    RawTensorDataPtr nativeData;
    nativeData = std::make_shared<RawTensorData>(dataType, shape);
    int64_t size = py::len(pythonData);
    switch (dataType) {
    case DataType::DT_INT32: {
            for (int64_t i = 0; i < size; i++) {
                nativeData->Get<int32_t>(i) = static_cast<int32_t>(py::int_((pythonData.attr("__getitem__")(i))));
            }
        } break;
    case DataType::DT_INT16: {
            for (int64_t i = 0; i < size; i++) {
                nativeData->Get<int16_t>(i) = static_cast<int16_t>(py::int_((pythonData.attr("__getitem__")(i))));
            }
        } break;
    case DataType::DT_INT8: {
            for (int64_t i = 0; i < size; i++) {
                nativeData->Get<int8_t>(i) = static_cast<int8_t>(py::int_((pythonData.attr("__getitem__")(i))));
            }
        } break;
    case DataType::DT_FP32: {
        for (int64_t i = 0; i < size; i++) {
            nativeData->Get<float>(i) = static_cast<float>(py::float_((pythonData.attr("__getitem__")(i))));
        }
    } break;
    case DataType::DT_FP16: {
        for (int64_t i = 0; i < size; i++) {
            nativeData->Get<float16>(i) = static_cast<float>(py::float_((pythonData.attr("__getitem__")(i))));
        }
    } break;
    case DataType::DT_BF16: {
        for (int64_t i = 0; i < size; i++) {
            nativeData->Get<bfloat16>(i) = static_cast<float>(py::float_((pythonData.attr("__getitem__")(i))));
        }
    } break;
    default:
        break;
    }
    return nativeData;
}

static void HostCastNativeToPython(const std::shared_ptr<LogicalTensor> &tensor, RawTensorDataPtr nativeData, py::object &pythonData) {
    DataType dataType = tensor->Datatype();
    int64_t size = nativeData->GetSize();
    switch (dataType) {
    case DataType::DT_INT32: {
            for (int64_t i = 0; i < size; i++) {
                pythonData.attr("__setitem__")(i, nativeData->Get<int32_t>(i));
            }
        } break;
    case DataType::DT_INT16: {
            for (int64_t i = 0; i < size; i++) {
                pythonData.attr("__setitem__")(i, static_cast<int32_t>(nativeData->Get<int16_t>(i)));
            }
        } break;
    case DataType::DT_INT8: {
            for (int64_t i = 0; i < size; i++) {
                pythonData.attr("__setitem__")(i, static_cast<int32_t>(nativeData->Get<int8_t>(i)));
            }
        } break;
    case DataType::DT_FP32: {
            for (int64_t i = 0; i < size; i++) {
                pythonData.attr("__setitem__")(i, nativeData->Get<float>(i));
            }
        } break;
    case DataType::DT_FP16: {
            for (int64_t i = 0; i < size; i++) {
                pythonData.attr("__setitem__")(i, static_cast<float>(nativeData->Get<float16>(i)));
            }
        } break;
    case DataType::DT_BF16: {
            for (int64_t i = 0; i < size; i++) {
                pythonData.attr("__setitem__")(i, static_cast<float>(nativeData->Get<bfloat16>(i)));
            }
        } break;
    default:
        break;
    }
}

static std::string DeviceRunOnceDataFromHost(py::list &inputPythonDataList, py::list &outputPythonDataList) {
    Function *func = Program::GetInstance().GetLastFunction();
    if (!func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC, GraphType::TENSOR_GRAPH)) {
        return "Invalid function format";
    }
    auto attr = func->GetDyndevAttribute();
    if (attr == nullptr) {
        return "Invalid function format";
    }
    auto &inputLogicalTensorList = attr->startArgsInputLogicalTensorList;
    auto &outputLogicalTensorList = attr->startArgsOutputLogicalTensorList;
    if (inputLogicalTensorList.size() != inputPythonDataList.size()) {
        return "mismatch input";
    }
    if (outputLogicalTensorList.size() != outputPythonDataList.size()) {
        return "mismatch output";
    }
    for (size_t i = 0; i < inputLogicalTensorList.size(); i++) {
        std::shared_ptr<LogicalTensor> inputLogicalTensor = inputLogicalTensorList[i];
        py::object inputPythonData = inputPythonDataList[i];
        RawTensorDataPtr inputNativeData = HostCastPythonToNative(inputLogicalTensor, inputPythonData);
        ProgramData::GetInstance().AppendInput(inputNativeData);
    }
    for (size_t i = 0; i < outputLogicalTensorList.size(); i++) {
        std::shared_ptr<LogicalTensor> outputLogicalTensor = outputLogicalTensorList[i];
        RawTensorDataPtr outputNativeData = std::make_shared<RawTensorData>(outputLogicalTensor->Datatype(), outputLogicalTensor->GetShape());
        ProgramData::GetInstance().AppendOutput(outputNativeData);
    }

    if (DeviceRunOnce(Program::GetInstance().GetLastFunction()) != 0) {
        return "device run failed";
    }

    for (size_t i = 0; i < outputLogicalTensorList.size(); i++) {
        std::shared_ptr<LogicalTensor> outputLogicalTensor = outputLogicalTensorList[i];
        py::object outputPythonData = outputPythonDataList[i];
        RawTensorDataPtr outputNativeData = ProgramData::GetInstance().GetOutputData(i);
        HostCastNativeToPython(outputLogicalTensor, outputNativeData, outputPythonData);
    }

    if (HasInplaceArgs(Program::GetInstance().GetLastFunction())) {
        for (size_t i = 0; i < inputLogicalTensorList.size(); i++) {
            std::shared_ptr<LogicalTensor> inputLogicalTensor = inputLogicalTensorList[i];
            py::object inputPythonData = inputPythonDataList[i];
            RawTensorDataPtr inputNativeData = ProgramData::GetInstance().GetInputData(i);
            HostCastNativeToPython(inputLogicalTensor, inputNativeData, inputPythonData);
        }
    }
    return "";
}

static std::string DeviceRunOnceDataFromDevice(py::list &inputDeviceAddrList, py::list &outputDeviceAddrList, py::int_ incomingStreamPython) {
    (void)incomingStreamPython;
    Function *func = Program::GetInstance().GetLastFunction();
    if (!func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC, GraphType::TENSOR_GRAPH)) {
        return "Invalid function format";
    }
    auto attr = func->GetDyndevAttribute();
    if (attr == nullptr) {
        return "Invalid function format";
    }
    auto &inputLogicalTensorList = attr->startArgsInputLogicalTensorList;
    auto &outputLogicalTensorList = attr->startArgsOutputLogicalTensorList;
    if (inputLogicalTensorList.size() != inputDeviceAddrList.size()) {
        return "mismatch input";
    }
    if (outputLogicalTensorList.size() != outputDeviceAddrList.size()) {
        return "mismatch output";
    }
    std::vector<DeviceTensorData> inputList;
    std::vector<DeviceTensorData> outputList;
    for (size_t i = 0; i < inputLogicalTensorList.size(); i++) {
        std::shared_ptr<LogicalTensor> inputLogicalTensor = inputLogicalTensorList[i];
        uintptr_t inputDeviceAddr = static_cast<uintptr_t>(py::int_(inputDeviceAddrList[i]));
        inputList.emplace_back(inputDeviceAddr, inputLogicalTensor->GetShape());
    }
    for (size_t i = 0; i < outputLogicalTensorList.size(); i++) {
        std::shared_ptr<LogicalTensor> outputLogicalTensor = outputLogicalTensorList[i];
        uintptr_t outputDeviceAddr = static_cast<uintptr_t>(py::int_(outputDeviceAddrList[i]));
        outputList.emplace_back(outputDeviceAddr, outputLogicalTensor->GetShape());
    }
    auto incomingStream = static_cast<uintptr_t>(incomingStreamPython);
    if (incomingStream == 0) {
        return "invalid incoming stream";
    }

    auto aicpuStream = incomingStream;
    auto aicoreStream = DeviceGetAicoreStream();
    int rc = DeviceLaunchOnceWithDeviceTensorData(
        Program::GetInstance().GetLastFunction(), inputList, outputList, aicpuStream, aicoreStream, false);
    if (rc < 0) {
        return "device run failed";
    }
    return "";
}

void DeviceInit() {
    DeviceLauncherInit();
}

void DeviceFini() {
    DeviceLauncherFini();
}

void BindRuntime(py::module &m) {
    m.def("_DeviceInit", &DeviceInit);
    m.def("_DeviceFini", &DeviceFini);
    m.def("_DeviceRunOnceDataFromHost", &DeviceRunOnceDataFromHost);
    m.def("_DeviceRunOnceDataFromDevice", &DeviceRunOnceDataFromDevice);
}

#else

void BindRuntime(py::module &m) {
    (void)m;
}

#endif

} // namespace pypto
