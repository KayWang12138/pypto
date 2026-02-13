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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/interpreter/raw_tensor_data.h"
#include "interface/utils/op_info_manager.h"
#include "machine/runtime/device_launcher_binding.h"
#include "machine/runtime/emulation_launcher.h"
#include "machine/runtime/device_launcher.h"
#include "machine/utils/dynamic/dev_start_args.h"
#include "machine/host/perf_analysis.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace pypto {

// DLPack ABI-compatible structs for parsing __dlpack__ capsule in TorchTensorToDeviceTensorData.
struct DLTensorView {
    void *data;
    int32_t device_type;
    int32_t device_id;
    int32_t ndim;
    uint8_t dtype_code;
    uint8_t dtype_bits;
    uint16_t dtype_lanes;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
};
struct DLManagedTensorView {
    DLTensorView dl_tensor;
    void *manager_ctx;
    void (*deleter)(void *);
};


// Cached torch._C._to_dlpack for fast DLPack path; avoids Python __dlpack__() call overhead.
static py::object GetTorchToDlpack() {
    try {
        py::module torch = py::module::import("torch");
        return torch.attr("_C").attr("_to_dlpack");
    } catch (...) {
        return py::none();
    }
}

void CopyToHost(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor) {
    CopyDevToHost(devTensor, hostTensor);
}

void CopyToDev(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor) {
    CopyHostToDev(devTensor, hostTensor);
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

static std::string ValidateFunctionAndIO(Function *func, const std::vector<DeviceTensorData> &inputs,
                                   const std::vector<DeviceTensorData> &outputs) {
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
    return "";
}

static void InitializeInputOutputData(const std::vector<DeviceTensorData> &inputs,
                               const std::vector<DeviceTensorData> &outputs) {
    for (size_t i = 0; i < inputs.size(); i++) {
        auto rawData = RawTensorData::CreateTensor(inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t *)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(outputs[i].GetDataType(), outputs[i].GetShape());
        ProgramData::GetInstance().AppendOutput(rawData);
    }
}

std::string DeviceRunOnceDataFromHost(
    const std::vector<DeviceTensorData> &inputs, const std::vector<DeviceTensorData> &outputs) {
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_ALL_COMPLETE) {
        return "";
    }
    ProgramData::GetInstance().Reset();
    Function *func = Program::GetInstance().GetLastFunction();
    auto errorMsg = ValidateFunctionAndIO(func, inputs, outputs);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    InitializeInputOutputData(inputs, outputs);

    DevControlFlowCache* hostCache = nullptr;
    if (config::GetRuntimeOption<int64_t>(STITCH_CFGCACHE_SIZE) != 0) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        EmulationLauncher::BuildControlFlowCache(func, inputs, outputs, &hostCache, config);
    }

    if (config::GetDebugOption<int>(CFG_RUNTIME_DBEUG_MODE) == 1 && EmulationLauncher::EmulationRunOnce(func, hostCache) != 0) {
        return "emulation run failed";
    }

    if (DeviceRunOnce(func, reinterpret_cast<uint8_t*>(hostCache)) != 0) {
        return "device run failed";
    }

    if (hostCache) {
        free(hostCache);
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
    [[maybe_unused]] py::int_ incomingStreamPython, [[maybe_unused]] py::int_ workspaceData,
    [[maybe_unused]] py::int_ devCtrlCache) {

    if (config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_ALL_COMPLETE) {
        return "";
    }
    HOST_PERF_TRACE_START();
    HOST_PERF_EVT_BEGIN(EventPhase::RunDevice);

#ifdef BUILD_WITH_CANN
    auto opAddr = static_cast<uintptr_t>(pythonOperatorPython);
    if (opAddr == 0) {
        return "invalid operator";
    }

    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    Function *func = op->GetFunction();
    auto errorMsg = ValidateFunctionAndIO(func, inputs, outputs);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    if (config::GetDebugOption<int>(CFG_RUNTIME_DBEUG_MODE) == 1) {
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
    auto ctrlCache = static_cast<uintptr_t>(devCtrlCache);
    int rc = ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(op, inputs, outputs,
        aicpuStream, aicoreStream, false, reinterpret_cast<uint8_t*>(ctrlCache),
        DeviceLauncherConfig::CreateConfigWithWorkspaceAddr(workspaceDataAddr));
    if (rc < 0) {
        return "device run failed";
    }
#endif

    HOST_PERF_EVT_END(EventPhase::RunDevice);
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

int64_t BuildCache(uintptr_t opAddr, const std::vector<DeviceTensorData> &inputList,
        const std::vector<DeviceTensorData> &outputList, [[maybe_unused]] bool isCapturing) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    if (config::GetRuntimeOption<int64_t>(STITCH_CFGCACHE_SIZE) != 0) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        uint8_t* ctrlCache = op->FindCtrlFlowCache(inputList, outputList);
        if (ctrlCache == nullptr) {
            HOST_PERF_EVT_BEGIN(EventPhase::BuildCtrlFlowCache);
            DevControlFlowCache* hostCache = nullptr;
            if (EmulationLauncher::BuildControlFlowCache(op->GetFunction(),
                inputList, outputList, &hostCache, config) != 0) {
                return 0;
            }

#ifdef BUILD_WITH_CANN
            if (isCapturing) {
                ChangeCaptureModeRelax();
            }

            if (hostCache) {
                ctrlCache = CopyHostToDev(reinterpret_cast<uint8_t*>(hostCache),
                    reinterpret_cast<DevControlFlowCache*>(hostCache)->usedCacheSize);
                free(hostCache);
            }

            if (isCapturing) {
                ChangeCaptureModeGlobal();
            }
#else
            ctrlCache = reinterpret_cast<uint8_t*>(hostCache);
#endif

            if (ctrlCache) {
                op->InsertCtrlFlowCache(inputList, outputList, ctrlCache);
            }
            HOST_PERF_EVT_END(EventPhase::BuildCtrlFlowCache);
        }

        return ctrlCache == nullptr ? 0 : reinterpret_cast<int64_t>(ctrlCache);
    }

    return 0;
}

#ifdef BUILD_WITH_CANN
#define ENABALE_VERBOSE_LOG 0
struct ControlFlowCache {
    int64_t hash;
    std::vector<DeviceTensorData> inputs;
    uint8_t *devCache{nullptr};

    ControlFlowCache(std::vector<DeviceTensorData> &datas, uint8_t *tcache) : inputs(datas), devCache(tcache) {
        hash = Hash(inputs);
    }

    static int64_t Hash(const std::vector<DeviceTensorData> &datas) {
        // FNV-1a
        uint64_t h = 14695981039346656037ull;
        for (auto &data : datas) {
            for (auto x : data.GetShape()) {
                h ^= x;
                h *= 1099511628211ull;
            }
        }
        return h;
    }

    static int64_t Hash(const std::vector<std::vector<int64_t>> &shapes) {
        // FNV-1a
        uint64_t h = 14695981039346656037ull;
        for (auto &shape : shapes) {
            for (auto x : shape) {
                h ^= x;
                h *= 1099511628211ull;
            }
        }
        return h;
    }
};

class KernelBinary {
public:
    KernelBinary(std::shared_ptr<Function> func) : dynFunc(func) {
        dynAttr = dynFunc->GetDyndevAttribute().get();
        devProg = (DevAscendProgram *)dynAttr->devProgBinary.data();
        kernelBin = DeviceLauncher::RegisterKernelBin(dynAttr->kernelBinary);
        workspaceSize = devProg->memBudget.Total();
        InitCachedArgs();
        auto aicpuArgs = (AiCpuArgs *)aicpuArgBuf.data();
        DeviceLauncher::FillDeviceKernelArgs(dynAttr->devProgBinary, aicpuArgs->kArgs);
    }

    uint8_t *FindCtrlFlowCache(std::vector<std::vector<int64_t>> &inputs, bool isOriginShape) {
        int64_t inHash = ControlFlowCache::Hash(inputs);
        auto& caches = isOriginShape ? originShapeCaches : inferShapeCaches;
        for (auto &cache : caches) {
            if (cache.hash == inHash) {
                return cache.devCache;
            }
        }
        return nullptr;
    }

    uint8_t *FindCtrlFlowCache(std::vector<DeviceTensorData> &inputs, bool isOriginShape) {
        int64_t inHash = ControlFlowCache::Hash(inputs);
        auto& caches = isOriginShape ? originShapeCaches : inferShapeCaches;
        for (auto &cache : caches) {
            if (cache.hash == inHash) {
                return cache.devCache;
            }
        }
        return nullptr;
    }

    uint8_t *BuildControlFlowCache(std::vector<DeviceTensorData> &inputs, int64_t cfgCacheSize, bool isOriginShape) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        DevControlFlowCache *ctrlCache = nullptr;

        devProg->ctrlFlowCacheSize = cfgCacheSize;
        config.isCacheOriginShape = isOriginShape;
        int ret = EmulationLauncher::BuildControlFlowCache(dynFunc.get(), inputs, {}, &ctrlCache, config);
        if (ret != 0) {
            ALOG_ERROR("control flow cache failed", ret);
            return nullptr;
        }

        auto ctrlCachePtr = std::unique_ptr<uint8_t>((uint8_t *)ctrlCache);
        uint8_t *devCache = DeviceLauncher::CopyControlFlowCache(ctrlCache);
#if ENABALE_VERBOSE_LOG
        std::stringstream ss;
        for (auto &t : inputs) {
            for (auto x : t.GetShape()) {
                ss << x << " ";
            }
        }
        ALOG_ERROR_F("control flow cache: %p shape %s", devCache, ss.str().c_str());
#endif
        if (isOriginShape) {
            originShapeCaches.emplace_back(inputs, devCache);
        } else{
            inferShapeCaches.emplace_back(inputs, devCache);
        }
        return devCache;
    }

    int64_t GetWorkspaceSize(const std::vector<DeviceTensorData> &tensors) {
        if (dynAttr->maxDynamicAssembleOutcastMem.IsValid()) {
            Evaluator eval{dynAttr->inputSymbolDict, tensors, {}};
            return workspaceSize + eval.Evaluate(dynAttr->maxDynamicAssembleOutcastMem);
        }
        return workspaceSize;
    }

    std::pair<AiCpuArgs *, int64_t> BuildKernelArgs(const std::vector<DeviceTensorData> &tensors) {
        auto &disableL2List = dynAttr->disableL2List;
        auto aicpuArgs = (AiCpuArgs *)aicpuArgBuf.data();
        int64_t *inputp = (int64_t *)(aicpuArgs + 1);
        auto tensorData = (DevTensorData *)(inputp + 2);
        ASSERT((int64_t)tensors.size() == inputp[0]) << "mismatch tensor size";
        for (size_t i = 0; i < (size_t)inputp[0]; ++i) {
            auto &t = tensors[i];
            auto addr = (uint64_t)t.GetAddr();
            if (unlikely(addr && disableL2List.size() && disableL2List[i])) {
                ALOG_ERROR("mismatch tensor addr");
                addr += l2Offset;
            }
            tensorData->address = addr;
            auto &shape = t.GetShape();
            tensorData->shape.dimSize = shape.size();
            for (int j = 0; j < tensorData->shape.dimSize; ++j) {
                tensorData->shape.dim[j] = shape[j];
            }
            tensorData++;
        }

        return {aicpuArgs, aicpuArgBuf.size() * sizeof(int64_t)};
    }

    bool CheckArgs(const std::vector<DeviceTensorData> &tensors) const {
        if (tensors.size() != argTypes.size()) {
            return false;
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto &t = tensors[i];
            auto &type = argTypes[i];
            if (unlikely(t.GetDataType() != type.GetDataType())) {
                return false;
            }
            auto &shape1 = type.GetShape();
            auto &shape2 = t.GetShape();
            if (unlikely(shape1.size() != shape2.size())) {
                return false;
            }
            for (size_t j = 0; j < shape1.size(); ++j) {
                if (unlikely((shape1[j] != -1) && (shape1[j] != shape2[j]))) {
                    return false;
                }
            }
        }
        return true;
    }

    void *GetKernelBin() { return kernelBin; }
    auto &GetArgTypes() { return argTypes; }
    Function *GetFunction() { return dynFunc.get(); }

    ~KernelBinary() {
        DeviceLauncher::UnregisterKernelBin(kernelBin);
        for (auto &cache : originShapeCaches) {
            DeviceLauncher::FreeControlFlowCache(cache.devCache);
        }
        for (auto &cache : inferShapeCaches) {
            DeviceLauncher::FreeControlFlowCache(cache.devCache);
        }
    }

private:
    void InitCachedArgs() {
        auto argNum =
            dynAttr->startArgsInputLogicalTensorList.size() + dynAttr->startArgsOutputLogicalTensorList.size();
        auto argSize = sizeof(AiCpuArgs) + 2 * sizeof(int64_t) + argNum * sizeof(DevTensorData);
        ASSERT(argSize % 8 == 0);
        aicpuArgBuf.resize(argSize / 8);

        auto aicpuArgs = new (aicpuArgBuf.data()) AiCpuArgs();
        aicpuArgs->kArgs.inputs = nullptr;
        aicpuArgs->kArgs.outputs = nullptr;

        int64_t *inputp = (int64_t *)(aicpuArgs + 1);
        inputp[0] = dynAttr->startArgsInputLogicalTensorList.size();
        inputp[1] = dynAttr->startArgsOutputLogicalTensorList.size();

        l2Offset = DeviceLauncher::GetL2Offset();

        for (auto &t : dynAttr->startArgsInputLogicalTensorList) {
            argTypes.emplace_back(t->Datatype(), nullptr, t->GetShape());
        }
        for (auto &t : dynAttr->startArgsOutputLogicalTensorList) {
            argTypes.emplace_back(t->Datatype(), nullptr, t->GetShape());
        }
    }

private:
    std::shared_ptr<Function> dynFunc;
    DyndevFunctionAttribute *dynAttr{nullptr};
    DevAscendProgram *devProg{nullptr};
    void *kernelBin{nullptr};
    int64_t workspaceSize{0}; // static workspace size
    std::vector<ControlFlowCache> inferShapeCaches;
    std::vector<ControlFlowCache> originShapeCaches;

    std::vector<int64_t> aicpuArgBuf;
    uint64_t l2Offset{0};
    std::vector<DeviceTensorData> argTypes;
};

class KernelModule {
public:
    KernelModule(py::object &module) {
        InitCachedArgs();
        InitConfigOptions(module);
    }

    ~KernelModule() {
        for (auto &k : kernels) {
            delete k;
        }
    }

    bool IsTripleStream() { return tripleStream; }

    KernelBinary *GetKernelBinary(std::vector<DeviceTensorData> &tensors) {
        for (auto &k : kernels) {
            if (k->CheckArgs(tensors)) {
                return k;
            }
        }
        return nullptr;
    }

    uint8_t *FindCtrlFlowCache(KernelBinary *kernel, py::object &module,
        std::vector<DeviceTensorData> &tensors, bool isCaptureMode) {
        if (!IsCacheEnabled()) {
            return nullptr;
        }

        auto devCache = kernel->FindCtrlFlowCache(tensors, true);
        if (devCache == nullptr) {
            std::vector<std::vector<int64_t>> shape;
            if (isCaptureMode) {
                AclModeGuard guard(ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
                devCache = kernel->BuildControlFlowCache(tensors, stitchCfgCacheSize, true);
            } else if (InferCacheShape(module, tensors, shape)) {
                devCache = kernel->FindCtrlFlowCache(shape, false);
            } else {
                AclModeGuard guard(ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
                devCache = kernel->BuildControlFlowCache(tensors, stitchCfgCacheSize, true);
            }
        }
#if ENABALE_VERBOSE_LOG
        std::stringstream ss;
        for (auto &t : tensors) {
            for (auto &s : t.GetShape()) {
                ss << s << " ";
            }
        }
        ALOG_ERROR_F("find ctrlflow cache: %p shape %s", devCache, ss.str().c_str());
#endif
        return devCache;
    }

    KernelBinary *Compile(py::object &module, py::args &args) {
        auto compile = py::getattr(module, "compile");
        compile(args);
        return RegisterLastCompiledKernel(module);
    }

    KernelBinary *CompileFromTorch(py::object &module, py::sequence tensors, py::sequence tensor_defs) {
        auto compile = py::getattr(module, "compile");
        compile(tensors, tensor_defs);
        return RegisterLastCompiledKernel(module);
    }

    KernelBinary *RegisterLastCompiledKernel(py::object &module) {
        auto func = Program::GetInstance().GetLastFunction();
        auto kernel = new KernelBinary(Program::GetInstance().GetFunctionSharedPtr(func));
        kernels.push_back(kernel);
        if (inferCacheShape) {
#if ENABALE_VERBOSE_LOG
            ALOG_ERROR("build default cache");
#endif
            BuildDefaultCache(kernel, module);
        }
        return kernel;
    }

    int64_t GetWorkspaceSize(KernelBinary *kernel, std::vector<DeviceTensorData> &tensors) {
        return kernel->GetWorkspaceSize(tensors);
    }

    void Launch(KernelBinary *kernel, bool isCaptureMode, aclrtStream aicoreStream,
        std::vector<DeviceTensorData> &tensors, uint8_t *ctrlFlowCache, int64_t *workspace) {
        auto [args, argsSize] = kernel->BuildKernelArgs(tensors);
        rtAicpuArgs.args = args;
        rtAicpuArgs.argsSize = argsSize;

        args->kArgs.ctrlFlowCache = (int64_t *)ctrlFlowCache;
        args->kArgs.workspace = workspace;
        args->kArgs.parameter.globalRound = ++sequence;

        bool debugEnable = !isCaptureMode && isDebugMode;

#if ENABALE_VERBOSE_LOG
        ALOG_ERROR_F("triple stream %d sequence %ld workspace %p cfgcache %p", tripleStream, sequence.load(), workspace,
            ctrlFlowCache);
#endif
        int ret = DeviceLauncher::LaunchAicpuKernel(rtAicpuArgs, tripleStream, debugEnable, kernel->GetFunction());
        ASSERT(ret == RT_ERROR_NONE) << "launch aicpu failed: " << ret;

        kernelArgs[5] = args->kArgs.cfgdata; // 5 is cfgdata
        ret = DeviceLauncher::LaunchAicoreKernel(aicoreStream, kernel->GetKernelBin(), rtAicoreArgs, rtTaskCfg, debugEnable);
        ASSERT(ret == RT_ERROR_NONE) << "launch aicore failed: " << ret;
    }

    void EmulationLaunch(KernelBinary *kernel, std::vector<DeviceTensorData> &tensors) {
        if (!isDebugMode) {
            return;
        }

        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        int ret = EmulationLauncher::EmulationLaunchDeviceTensorData(kernel->GetFunction(), tensors, {}, config);
        ASSERT(ret == RT_ERROR_NONE) << "emulation run failed: " << ret;
    }

private:
    void InitCachedArgs() {
        memset_s(&rtAicpuArgs, sizeof(rtAicpuArgsEx_t), 0, sizeof(rtAicpuArgsEx_t));
        rtAicpuArgs.kernelNameAddrOffset = offsetof(dynamic::AiCpuArgs, kernelName);
        rtAicpuArgs.soNameAddrOffset = offsetof(dynamic::AiCpuArgs, soName);
        rtAicpuArgs.hostInputInfoNum = 1;
        hostInfo.addrOffset = offsetof(dynamic::AiCpuArgs, kArgs.inputs);
        hostInfo.dataOffset = sizeof(dynamic::AiCpuArgs);
        rtAicpuArgs.hostInputInfoPtr = &hostInfo;

        memset_s(&rtAicoreArgs, sizeof(rtArgsEx_t), 0, sizeof(rtArgsEx_t));
        kernelArgs.resize(7, nullptr); // see aicore.ascpp
        rtAicoreArgs.args = kernelArgs.data();
        rtAicoreArgs.argsSize = kernelArgs.size() * sizeof(void *);

        memset_s(&rtTaskCfg, sizeof(rtTaskCfgInfo_t), 0, sizeof(rtTaskCfgInfo_t));
        rtTaskCfg.schemMode = RT_SCHEM_MODE_BATCH;
    }

    void InitConfigOptions(py::object &module) {
        auto options = module.attr("_runtime_options").cast<py::dict>();
        if (options.contains("triple_stream_sched")) {	 
            tripleStream = options["triple_stream_sched"].cast<bool>(); 
        }
        if (options.contains("stitch_cfgcache_size")) {
            stitchCfgCacheSize = options["stitch_cfgcache_size"].cast<int64_t>();
        }
        if (!module.attr("_debug_options").is_none()) {
            auto debugOptions = module.attr("_debug_options").cast<py::dict>();
            if (debugOptions.contains("runtime_debug_mode")) {
                isDebugMode = debugOptions["runtime_debug_mode"].cast<int64_t>() == CFG_DEBUG_ALL;
            }
        }
        if (!module.attr("_infer_controlflow_shape").is_none()) {
            inferCacheShape = true;
        }
#if ENABALE_VERBOSE_LOG
        ALOG_ERROR("triple_stream_sched: ", tripleStream, " stitch_cfgcache_size: ", stitchCfgCacheSize,
            " infer_cache_shape: ", inferCacheShape);
#endif
    }

    void BuildDefaultCache(KernelBinary *kernel, py::object &module) {
        auto infershape = py::getattr(module, "_infer_controlflow_shape");
        auto cfshapes = infershape().cast<py::list>();
        auto tensors = kernel->GetArgTypes();
        for (auto &pyshape : cfshapes) {
            auto inputShapes = pyshape.cast<std::vector<std::vector<int64_t>>>();
            if (inputShapes.size() != tensors.size()) {
                ALOG_ERROR("Invalid input size, expect: ", tensors.size(), " got: ", inputShapes.size());
                continue;
            }
            std::vector<DeviceTensorData> inputs;
            for (size_t i = 0; i < tensors.size(); i++) {
                inputs.emplace_back(tensors[i].GetDataType(), nullptr, inputShapes[i]);
            }
            if (kernel->CheckArgs(inputs)) {
                kernel->BuildControlFlowCache(inputs, stitchCfgCacheSize, false);
            } else {
                ALOG_ERROR("Invalid cache shape, skip it");
            }
        }
    }

    bool InferCacheShape(py::object &module, std::vector<DeviceTensorData> &tensors,
        std::vector<std::vector<int64_t>> &shapes) {
        auto infershape = py::getattr(module, "_infer_controlflow_shape", py::none());
        if (infershape.is_none()) {
            return false;
        }
        py::list oriShapes;
        for (auto &t : tensors) {
            oriShapes.append(py::cast(t.GetShape()));
        }
        auto cfshape = infershape(*oriShapes);
        if (cfshape.is_none()) {
            return false;
        }
        shapes = cfshape.cast<std::vector<std::vector<int64_t>>>();
        return true;
    }

    bool IsCacheEnabled() { return stitchCfgCacheSize != 0; }

private:
    bool inferCacheShape{false};
    bool tripleStream{false};
    bool isDebugMode{false};
    int64_t stitchCfgCacheSize{0};

    rtHostInputInfo_t hostInfo;
    rtAicpuArgsEx_t rtAicpuArgs;

    rtArgsEx_t rtAicoreArgs;
    rtTaskCfgInfo_t rtTaskCfg;
    std::vector<void *> kernelArgs;
    std::vector<KernelBinary *> kernels;

    static std::atomic<int64_t> sequence;
};
using KernelModulePtr = std::shared_ptr<KernelModule>;

std::atomic<int64_t> KernelModule::sequence(0);

static int GetInputTensors(py::args &args, std::vector<DeviceTensorData> &tensors) {
    py::object device = py::none();
    for (auto &pt : args) {
        auto base = py::getattr(pt, "_base", py::none());
        if (py::isinstance<Tensor>(base)) {
            auto &t = base.cast<Tensor &>();
            auto data_ptr = py::cast<int64_t>(py::getattr(pt, "data_ptr"));
            auto shape = py::cast<std::vector<int64_t>>(py::getattr(pt, "ori_shape"));
            tensors.emplace_back(t.GetDataType(), data_ptr, shape);
            if (device.is_none()) {
                device = py::getattr(pt, "device");
            } else if (!device.equal(py::getattr(pt, "device"))) {
                throw std::runtime_error("All input tensors must be on the same device");
            }
        }
    }
    ASSERT(tensors.size()) << "No input tensors found";
    if (py::getattr(device, "type").cast<std::string>() != "npu") {
        throw std::runtime_error("Not npu device");
    }
    return py::getattr(device, "index").cast<int>();
}

static bool ParseDlpackCapsule(py::object cap, uintptr_t &data_ptr, std::vector<int64_t> &shape,
    int *device_id = nullptr) {
    if (cap.is_none()) return false;
    void *ptr = PyCapsule_GetPointer(cap.ptr(), "dltensor");
    if (!ptr) {
        PyErr_Clear();
        return false;
    }
    auto &t = static_cast<DLManagedTensorView *>(ptr)->dl_tensor;
    data_ptr = reinterpret_cast<uintptr_t>(static_cast<char *>(t.data) + t.byte_offset);
    shape.assign(t.shape, t.shape + t.ndim);
    if (device_id) *device_id = t.device_id;
    return true;
}

static py::object GetTorchToDlpackCapsule(py::object torch_tensor) {
    py::object to_dlpack = GetTorchToDlpack();
    if (to_dlpack.is_none()) return py::none();
    try {
        return to_dlpack(torch_tensor);
    } catch (const py::error_already_set &) {
        PyErr_Clear();
        return py::none();
    } catch (const std::exception &) {
        return py::none();
    }
}

static py::object GetNativeDlpackCapsule(py::object torch_tensor) {
    if (!py::hasattr(torch_tensor, "__dlpack__")) return py::none();
    try {
        return torch_tensor.attr("__dlpack__")(py::arg("stream") = -1);
    } catch (const py::error_already_set &) {
        PyErr_Clear();
        return py::none();
    } catch (const std::exception &) {
        return py::none();
    }
}

static bool TryParseDlpack(py::object torch_tensor, uintptr_t &data_ptr, std::vector<int64_t> &shape,
    int *device_id = nullptr) {
    py::object cap = GetTorchToDlpackCapsule(torch_tensor);
    if (!cap.is_none() && ParseDlpackCapsule(cap, data_ptr, shape, device_id)) return true;
    cap = GetNativeDlpackCapsule(torch_tensor);
    if (!cap.is_none() && ParseDlpackCapsule(cap, data_ptr, shape, device_id)) return true;
    return false;
}

class TorchTensorConverter {
public:
    static DeviceTensorData Convert(py::object torch_tensor, py::object tensor_def,
        int *device_id_out = nullptr) {
        std::vector<int64_t> shape;
        uintptr_t data_ptr = 0;

        if (!TryParseDlpack(torch_tensor, data_ptr, shape, device_id_out)) {
            ExtractTensorMetadata(torch_tensor, data_ptr, shape);
        }

        DataType dtype = GetDataType(tensor_def);
        return DeviceTensorData(dtype, data_ptr, shape);
    }

private:
    static void ExtractTensorMetadata(py::object torch_tensor, uintptr_t &data_ptr,
        std::vector<int64_t> &shape) {
        try {
            data_ptr = static_cast<uintptr_t>(py::cast<int64_t>(torch_tensor.attr("data_ptr")()));
            for (auto dim : torch_tensor.attr("shape")) {
                shape.push_back(py::cast<int64_t>(dim));
            }
        } catch (const py::error_already_set &) {
            PyErr_Clear();
            throw std::runtime_error("Input tensor is not a valid torch tensor type");
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("Invalid torch tensor: ") + e.what());
        }
    }

    static DataType GetDataType(py::object tensor_def) {
        auto base = py::getattr(tensor_def, "_base", py::none());
        if (!base.is_none() && py::isinstance<Tensor>(base)) {
            return base.cast<Tensor &>().GetDataType();
        }
        return tensor_def.attr("dtype").cast<DataType>();
    }
};

class DeviceValidator {
public:
    static int ValidateAndGetDeviceId(const std::vector<int> &device_ids) {
        int device_id = -1;
        for (size_t i = 0; i < device_ids.size(); i++) {
            if (device_ids[i] < 0) continue;

            if (device_id < 0) {
                device_id = device_ids[i];
            } else if (device_ids[i] != device_id) {
                throw std::runtime_error(
                    "Tensor at index " + std::to_string(i) +
                    " is on device " + std::to_string(device_ids[i]) +
                    ", expected device " + std::to_string(device_id));
            }
        }

        if (device_id < 0) {
            throw std::runtime_error(
                "Unable to determine device ID: ensure all tensors support DLPack");
        }
        return device_id;
    }
};

static size_t ValidateInputs(py::sequence tensors, py::sequence tensor_defs) {
    size_t n = static_cast<size_t>(py::len(tensors));
    if (n != static_cast<size_t>(py::len(tensor_defs))) {
        throw std::runtime_error(
            "Input length mismatch: tensors(" + std::to_string(n) +
            ") vs tensor_defs(" + std::to_string(py::len(tensor_defs)) + ")");
    }
    if (n == 0) {
        throw std::runtime_error("Empty tensor list");
    }
    return n;
}

static void DoLaunch(py::object &module, aclrtStream aicoreStream, KernelModulePtr kmodule,
    KernelBinary *kbinary, std::vector<DeviceTensorData> &tensors) {
    kmodule->EmulationLaunch(kbinary, tensors);
    HOST_PERF_TRACE(TracePhase::LaunchGetKernel);

#if ENABALE_VERBOSE_LOG
    ALOG_ERROR("alloc workspace");
#endif
    int64_t *wsAddr = nullptr;
    int64_t wsSize = kmodule->GetWorkspaceSize(kbinary, tensors);
    if (wsSize) {
        auto pyalloc = py::getattr(module, "alloc");
        wsAddr = (int64_t *)pyalloc(wsSize).cast<int64_t>();
    }
    HOST_PERF_TRACE(TracePhase::LaunchAllocWorkSpace);

    bool isCaptureMode = DeviceLauncher::AddAicpuStream(aicoreStream, kmodule->IsTripleStream());
    HOST_PERF_TRACE(TracePhase::LaunchAttachStream);
    
    uint8_t *ctrlFlowCache = kmodule->FindCtrlFlowCache(kbinary, module, tensors, isCaptureMode);
    HOST_PERF_TRACE(TracePhase::FindCtrlFlowCache);
    
    kmodule->Launch(kbinary, isCaptureMode, aicoreStream, tensors, ctrlFlowCache, wsAddr);
    HOST_PERF_TRACE(TracePhase::Launch);
    HOST_PERF_EVT_END(EventPhase::LaunchKernel);
}

void LaunchKernelTorch(py::object &module, int64_t stream, py::sequence &tensors,
                       py::sequence &tensor_defs) {
    HOST_PERF_TRACE_START();
    HOST_PERF_EVT_BEGIN(EventPhase::LaunchKernel);
    auto aicoreStream = (aclrtStream)stream;

    const size_t n = ValidateInputs(tensors, tensor_defs);
    std::vector<DeviceTensorData> tensors_data;
    std::vector<int> device_ids;
    tensors_data.reserve(n);
    device_ids.reserve(n);
    for (size_t i = 0; i < n; i++) {
        int device_id = -1;
        tensors_data.push_back(TorchTensorConverter::Convert(
            tensors[py::int_(i)], tensor_defs[py::int_(i)], &device_id));
        device_ids.push_back(device_id);
    }
    int devId = DeviceValidator::ValidateAndGetDeviceId(device_ids);
    DeviceGuard devGuard(devId);

    auto kmodule = py::getattr(module, "kmodule").cast<KernelModulePtr>();
    HOST_PERF_TRACE(TracePhase::LaunchInit);

    auto kbinary = kmodule->GetKernelBinary(tensors_data);
    if (kbinary == nullptr) {
        Program::GetInstance().Reset();
        // Set capture mode to relaxed to support rtmemcpy / rtmemset
        AclModeGuard guard(ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
#if ENABALE_VERBOSE_LOG
        ALOG_ERROR("compile kernel");
#endif
        kbinary = kmodule->CompileFromTorch(module, tensors, tensor_defs);
    }
    DoLaunch(module, aicoreStream, kmodule, kbinary, tensors_data);
}

void LaunchKernel(py::object &module, int64_t stream, py::args &args) {
    HOST_PERF_TRACE_START();
    HOST_PERF_EVT_BEGIN(EventPhase::LaunchKernel);
    auto aicoreStream = (aclrtStream)stream;

    std::vector<DeviceTensorData> tensors;
    auto devId = GetInputTensors(args, tensors);
    DeviceGuard devGuard(devId);

    auto kmodule = py::getattr(module, "kmodule").cast<KernelModulePtr>();
    HOST_PERF_TRACE(TracePhase::LaunchInit);

    auto kbinary = kmodule->GetKernelBinary(tensors);
    if (kbinary == nullptr) {
        Program::GetInstance().Reset();
        // Set capture mode to relaxed to support rtmemcpy / rtmemset
        AclModeGuard guard(ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
#if ENABALE_VERBOSE_LOG
        ALOG_ERROR("compile kernel");
#endif
        kbinary = kmodule->Compile(module, args);
    }
    DoLaunch(module, aicoreStream, kmodule, kbinary, tensors);
}
#else
void LaunchKernel(py::object &, int64_t, py::args &) { }
void LaunchKernelTorch(py::object &, py::object, py::sequence, py::sequence) {
    throw std::runtime_error("LaunchKernelTorch requires BUILD_WITH_CANN (NPU)");
}
class KernelModule {
public:
    KernelModule(py::object &) { }
};
using KernelModulePtr = std::shared_ptr<KernelModule>;
#endif

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
    m.def("CopyToDev", &CopyToDev);
    m.def("LaunchKernel", &LaunchKernel);
    m.def("LaunchKernelTorch", &LaunchKernelTorch);

    py::class_<DeviceTensorData>(m, "DeviceTensorData")
        .def(py::init<DataType, uintptr_t, const std::vector<int64_t> &>(), py::arg("dtype"), py::arg("addr"),
            py::arg("shape"))
        .def("GetDataPtr", &DeviceTensorData::GetAddr)
        .def("GetShape", &DeviceTensorData::GetShape)
        .def("GetDataType", &DeviceTensorData::GetDataType);

    py::class_<KernelModule, KernelModulePtr>(m, "KernelModule").def(py::init<py::object &>());
}
} // namespace pypto
