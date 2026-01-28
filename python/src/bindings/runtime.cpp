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
#include <utility>
#include <vector>
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/utils/op_info_manager.h"
#include "machine/runtime/device_launcher_binding.h"
#include "machine/runtime/emulation_launcher.h"
#include "machine/host/perf_analysis.h"
#include "tilefwk/aikernel_define.h"
#include "utils/log.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace pypto {

void CopyToHost(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor) {
    CopyDevToHost(devTensor, hostTensor);
}

void CopyToDev(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor) {
    CopyHostToDev(devTensor, hostTensor);
}

void SetVerifyData(const std::vector<DeviceTensorData> &inputs, const std::vector<DeviceTensorData> &outputs,
    const std::vector<DeviceTensorData> &goldens) {
    ProgramData::GetInstance().Reset();
    for (size_t i = 0; i < inputs.size(); i++) {
        auto rawData =
            RawTensorData::CreateTensor(inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t *)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(outputs[i].GetDataType(), outputs[i].GetShape());
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
        auto rawData =
            RawTensorData::CreateTensor(inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t *)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(outputs[i].GetDataType(), outputs[i].GetShape());
        ProgramData::GetInstance().AppendOutput(rawData);
    }

    DevControlFlowCache *hostCache = nullptr;
    if (config::GetRuntimeOption<int64_t>(STITCH_CFGCACHE_SIZE) != 0) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        EmulationLauncher::BuildControlFlowCache(func, inputs, outputs, &hostCache, config);
    }

    if (config::GetDebugOption<int>(CFG_RUNTIME_DBEUG_MODE) == 1 &&
        EmulationLauncher::EmulationRunOnce(func, hostCache) != 0) {
        return "emulation run failed";
    }

    if (DeviceRunOnce(func, reinterpret_cast<uint8_t *>(hostCache)) != 0) {
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
    [[maybe_unused]] const std::vector<DeviceTensorData> &inputs,
    [[maybe_unused]] const std::vector<DeviceTensorData> &outputs, [[maybe_unused]] py::int_ incomingStreamPython,
    [[maybe_unused]] py::int_ workspaceData, [[maybe_unused]] py::int_ devCtrlCache) {
    HOST_PERF_TRACE_START();
    HOST_PERF_EVT_BEGIN(EventPhase::RunDevice);

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
    int rc = ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(op, inputs, outputs, aicpuStream, aicoreStream, false,
        reinterpret_cast<uint8_t *>(ctrlCache), DeviceLauncherConfig::CreateConfigWithWorkspaceAddr(workspaceDataAddr));
    if (rc < 0) {
        return "device run failed";
    }
#endif

    HOST_PERF_EVT_END(EventPhase::RunDevice);
    return "";
}

uint64_t GetWorkSpaceSize(
    uintptr_t opAddr, const std::vector<DeviceTensorData> &inputs, const std::vector<DeviceTensorData> &outputs) {
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
    const std::vector<DeviceTensorData> &outputList, bool isCapturing) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    if (config::GetRuntimeOption<int64_t>(STITCH_CFGCACHE_SIZE) != 0) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        uint8_t *ctrlCache = op->FindCtrlFlowCache(inputList, outputList);
        if (ctrlCache == nullptr) {
            HOST_PERF_EVT_BEGIN(EventPhase::BuildCtrlFlowCache);
            DevControlFlowCache *hostCache = nullptr;
            if (EmulationLauncher::BuildControlFlowCache(
                    op->GetFunction(), inputList, outputList, &hostCache, config) != 0) {
                return 0;
            }

#ifdef BUILD_WITH_CANN
            if (isCapturing) {
                ChangeCaptureModeRelax();
            }

            if (hostCache) {
                ctrlCache = CopyHostToDev(reinterpret_cast<uint8_t *>(hostCache),
                    reinterpret_cast<DevControlFlowCache *>(hostCache)->allCacheSize);
                free(hostCache);
            }

            if (isCapturing) {
                ChangeCaptureModeGlobal();
            }
#else
            ctrlCache = reinterpret_cast<uint8_t *>(hostCache);
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

class DeviceGuard {
public:
    DeviceGuard(int32_t devId) : nDevId(devId) {
        (void)rtGetDevice(&oDevId);
        if (nDevId != oDevId) {
            rtSetDevice(nDevId);
        }
    }

    ~DeviceGuard() {
        if (nDevId != oDevId) {
            rtSetDevice(oDevId);
        }
    }

private:
    int32_t oDevId{0};
    int32_t nDevId{0};
};


class AclModeGuard {
public:
    AclModeGuard(aclmdlRICaptureMode tmode) : mode(tmode) { aclmdlRICaptureThreadExchangeMode(&mode); }
    ~AclModeGuard() { aclmdlRICaptureThreadExchangeMode(&mode); }
private:
    aclmdlRICaptureMode mode;
};


struct ControlFlowCache {
    int64_t hash;
    std::vector<DeviceTensorData> inputs;
    uint8_t *devCache {nullptr};

    ControlFlowCache(std::vector<DeviceTensorData> &datas, uint8_t *tcache)
        : inputs(std::move(datas)), devCache(tcache) {
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
    KernelBinary(std::shared_ptr<Function> func): dynFunc(func) {
        dynAttr = dynFunc->GetDyndevAttribute().get();
        devProg = (DevAscendProgram *)dynAttr->devProgBinary.data();
        kernelBin = RegisterAicoreKernel();
        workspaceSize = devProg->memBudget.Total();
        InitCachedArgs();
        auto aicpuArgs = (AiCpuArgs *)aicpuArgBuf.data();
        DeviceLauncher::FillDeviceKernelArgs(dynAttr->devProgBinary, aicpuArgs->kArgs);
    }

    uint8_t *FindCtrlFlowCache(std::vector<std::vector<int64_t>> &inputs) {
        int64_t inHash = ControlFlowCache::Hash(inputs);
        for (auto &cache : caches) {
            if (cache.hash == inHash) {
                return cache.devCache;
            }
        }
        return nullptr;
    }

    DevControlFlowCache *BuildControlFlowCache(std::vector<DeviceTensorData> &inputs, bool cache) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        DevControlFlowCache *ctrlCache = nullptr;

        int ret = EmulationLauncher::BuildControlFlowCache(dynFunc.get(), inputs, {}, &ctrlCache, config);
        if (ret != 0) {
            ALOG_ERROR("control flow cache failed", ret);
            return nullptr;
        }

        if (cache && ctrlCache) {
            auto devCache = CopyHostToDev(reinterpret_cast<uint8_t *>(ctrlCache), ctrlCache->allCacheSize);
            if (devCache) {
                caches.emplace_back(inputs, devCache);
            }
        }
        return ctrlCache;
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

    ~KernelBinary() {
        rtDevBinaryUnRegister(kernelBin);
    }

private:
    void *RegisterAicoreKernel() {
        void *hdl = nullptr;
        rtDevBinary_t binary = {
            .magic = RT_DEV_BINARY_MAGIC_ELF,
            .version = 0,
            .data = dynAttr->kernelBinary.data(),
            .length = dynAttr->kernelBinary.size(),
        };

        int ret = rtRegisterAllKernel(&binary, &hdl);
        if (ret != RT_ERROR_NONE) {
            ALOG_ERROR("register kernel failed, ret: %d", ret);
        }
        return hdl;
    }

    void InitCachedArgs() {
        auto genSize = devProg->memBudget.metadata.general;
        auto stitchPoolSize = devProg->memBudget.metadata.general;

        auto argNum = dynAttr->startArgsInputLogicalTensorList.size() +
            dynAttr->startArgsOutputLogicalTensorList.size();
        auto argSize = sizeof(AiCpuArgs) + 2 * sizeof(int64_t) + argNum * sizeof(DevTensorData);
        ASSERT(argSize % 8 == 0);
        aicpuArgBuf.resize(argSize / 8);

        auto aicpuArgs = new (aicpuArgBuf.data()) AiCpuArgs();
        aicpuArgs->kArgs.inputs = nullptr;
        aicpuArgs->kArgs.outputs = nullptr;

        int64_t *inputp = (int64_t *)(aicpuArgs + 1);
        inputp[0] = dynAttr->startArgsInputLogicalTensorList.size();
        inputp[1] = dynAttr->startArgsOutputLogicalTensorList.size();

        l2Offset = machine::GetRA()->GetL2Offset();

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
    std::vector<ControlFlowCache> caches;

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

    bool IsCompiled() { return kernel != nullptr; }
    bool IsCacheEnabled() { return stitchCfgCacheSize != 0; }
    bool IsTripleStream() { return tripleStream; }
    bool CheckArgs(std::vector<DeviceTensorData> &tensors) {
        return kernel->CheckArgs(tensors);
    }
    int64_t GetWorkspaceSize(std::vector<DeviceTensorData> &tensors) {
        return kernel->GetWorkspaceSize(tensors);
    }

    void Compile(py::object &module, py::args &args) {
        ASSERT(kernel == nullptr) << "function already set";
        auto compile = py::getattr(module, "compile");
        compile(args);
        auto func = Program::GetInstance().GetLastFunction();
        kernel = new KernelBinary(Program::GetInstance().GetFunctionSharedPtr(func));
        if (inferCacheShape) {
            ALOG_ERROR("build default cache");
            BuildDefaultCache(module);
        }
    }

    void Launch(aclrtStream aicoreStream, aclrtStream ctrlStream, aclrtStream schedtream,
        std::vector<DeviceTensorData> &tensors, uint8_t *ctrlFlowCache, int64_t *workspace) {

        auto [args, argsSize] = kernel->BuildKernelArgs(tensors);
        rtAicpuArgs.args = args;
        rtAicpuArgs.argsSize = argsSize;

        args->kArgs.ctrlFlowCache = (int64_t *)ctrlFlowCache;
        args->kArgs.workspace = workspace;
        args->kArgs.parameter.globalRound = ++sequence;
        ALOG_ERROR("start launch kernel triple stream ", tripleStream, " sequence ", sequence);
        if (tripleStream) {
            args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
            int ret = rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU_KFC,
                "AST_DYN_AICPU", 2, &rtAicpuArgs, nullptr, ctrlStream, 0);
            ASSERT(ret == RT_ERROR_NONE) << "launch aicpu ctrl failed: " << ret;
            args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_SCHE;
            ret = rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU_KFC,
                "AST_DYN_AICPU", 3, &rtAicpuArgs, nullptr, schedtream, 0);
            ASSERT(ret == RT_ERROR_NONE) << "launch aicpu sched failed: " << ret;
        } else {
            const int nrAicpu = 5; // see also device_runner.cpp
            args->kArgs.parameter.runMode = RUN_UNIFIED_STREAM;
            int ret = rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU_KFC,
                "AST_DYN_AICPU", nrAicpu, &rtAicpuArgs, nullptr, schedtream, 0);
            ASSERT(ret == RT_ERROR_NONE) << "launch aicpu def failed: " << ret;
        }

        ALOG_ERROR("launch aicore kernel");
        kernelArgs[5] = args->kArgs.cfgdata; // 5 is cfgdata
        auto tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
        auto ret = rtKernelLaunchWithHandleV2(kernel->GetKernelBin(), tilingKey, dynamic::GetCfgBlockdim(),
            &rtAicoreArgs, nullptr, aicoreStream, &rtTaskCfg);
        ASSERT(ret == RT_ERROR_NONE) << "launch aicore failed: " << ret;
    }

    uint8_t *FindCtrlFlowCache(py::object &module, py::args &args) {
        if (IsCacheEnabled()) {
            auto shape = InferCacheShape(module, args);
            return kernel->FindCtrlFlowCache(shape);
        }
        return nullptr;
    }

    uint8_t *BuildTempCache(py::object &module, std::vector<DeviceTensorData> &tensors) {
        auto ctrlCache = kernel->BuildControlFlowCache(tensors, false);
        auto size = ctrlCache->allCacheSize;
        auto pyalloc = py::getattr(module, "alloc");
        auto devCache = (uint8_t *)pyalloc(size).cast<int64_t>();
        AclModeGuard guard(ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
        int ret = rtMemcpy(devCache, size, (uint8_t *)ctrlCache, size, RT_MEMCPY_HOST_TO_DEVICE);
        if (ret != RT_ERROR_NONE) {
            ALOG_ERROR("memcpy failed ", ret);
        }
        return devCache;
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
        auto options = module.attr("runtime_options");
        if (py::hasattr(options, "triple_stream_sched")) {
            tripleStream = options.attr("triple_stream_sched").cast<bool>();
        }
        if (py::hasattr(options, "stitch_cfgcache_size")) {
            stitchCfgCacheSize = options.attr("stitch_cfgcache_size").cast<int64_t>();
        }
        if (!module.attr("infer_controlflow_shape").is_none()) {
            inferCacheShape = true;
        }
    }

    void BuildDefaultCache(py::object &module) {
        auto infershape = py::getattr(module, "infer_controlflow_shape");
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
            kernel->BuildControlFlowCache(inputs, true);
        }
    }

    std::vector<std::vector<int64_t>> InferCacheShape(py::object &module, py::args &args) {
        auto infershape = py::getattr(module, "infer_controlflow_shape");
        py::list oriShapes;
        for (auto &pt : args) {
            auto shape = py::getattr(pt, "ori_shape");
            if (!shape.is_none()) {
                oriShapes.append(shape);
            }
        }
        auto cfshape = infershape(*oriShapes);
        if (cfshape.is_none()) {
            return {};
        }
        return cfshape.cast<std::vector<std::vector<int64_t>>>();
    }

private:
    bool inferCacheShape{false};
    bool tripleStream{false};
    int64_t stitchCfgCacheSize{0};

    rtHostInputInfo_t hostInfo;
    rtAicpuArgsEx_t rtAicpuArgs;

    rtArgsEx_t rtAicoreArgs;
    rtTaskCfgInfo_t rtTaskCfg;
    std::vector<void *> kernelArgs;
    KernelBinary *kernel{nullptr};

    static std::atomic<int64_t> sequence;
};
using KernelModulePtr = std::shared_ptr<KernelModule>;

std::atomic<int64_t> KernelModule::sequence(0);

static int GetInputTensors(py::args &args, std::vector<DeviceTensorData> &tensors) {
    py::object device = py::none();
    for (auto &pt : args) {
        auto base = py::getattr(pt, "_base");
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

static bool AttachAicpuStream(aclrtStream aicoreStream, aclrtStream ctrlStream, aclrtStream schedtream, bool tripleStream) {
    aclmdlRI rtModel;
    aclmdlRICaptureStatus status = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    auto ret = aclmdlRICaptureGetInfo(aicoreStream, &status, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        return false;
    } else if (ret != ACL_SUCCESS) {
        ALOG_ERROR("get capture info failed: ", ret);
        return false;
    }
    if (status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        if (tripleStream) {
            rtStreamAddToModel(ctrlStream, rtModel);
        }
        rtStreamAddToModel(schedtream, rtModel);
        return true;
    }
    return false;
}

void LaunchKernel(py::object &module, int64_t stream, py::args &args) {
    auto aicoreStream = (aclrtStream)stream;
    auto schedtream = (aclrtStream)machine::GetRA()->GetScheStream();
    auto ctrlStream = (aclrtStream)machine::GetRA()->GetCtrlStream();

    ALOG_ERROR("parse input tensors");
    std::vector<DeviceTensorData> tensors;
    auto devId = GetInputTensors(args, tensors);
    DeviceGuard devGuard(devId);

    auto kmodule = py::getattr(module, "kmodule").cast<KernelModulePtr>();
    if (!kmodule->IsCompiled()) {
        Program::GetInstance().Reset();
        // Set capture mode to relaxed to support rtmemcpy / rtmemset
        AclModeGuard guard(ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
        ALOG_ERROR("compile kernel");
        kmodule->Compile(module, args);
    } else {
        ASSERT(kmodule->CheckArgs(tensors)) << "Invalid input tensors";
    }

    ALOG_ERROR("find ctrlflow cache");
    uint8_t *ctrlFlowCache = kmodule->FindCtrlFlowCache(module, args);
    auto captured = AttachAicpuStream(aicoreStream, ctrlStream, schedtream, kmodule->IsTripleStream());
    if (ctrlFlowCache == nullptr && kmodule->IsCacheEnabled()) {
        // TODO none reloc cache if captured
        ALOG_ERROR("build temp cache");
        ctrlFlowCache = kmodule->BuildTempCache(module, tensors);
    }

    int64_t *wsAddr = nullptr;
    int64_t wsSize = kmodule->GetWorkspaceSize(tensors);
    if (wsSize) {
        auto pyalloc = py::getattr(module, "alloc");
        wsAddr = (int64_t *)pyalloc(wsSize).cast<int64_t>();
    }
    ALOG_ERROR("start launch kernel");
    kmodule->Launch(ctrlStream, schedtream, aicoreStream, tensors, ctrlFlowCache, wsAddr);
    ALOG_ERROR("launch kernel end");
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
    m.def("CopyToDev", &CopyToDev);
    m.def("LaunchKernel", &LaunchKernel);

    py::class_<DeviceTensorData>(m, "DeviceTensorData")
        .def(py::init<DataType, uintptr_t, const std::vector<int64_t> &>(), py::arg("dtype"), py::arg("addr"),
            py::arg("shape"))
        .def("GetDataPtr", &DeviceTensorData::GetAddr)
        .def("GetShape", &DeviceTensorData::GetShape)
        .def("GetDataType", &DeviceTensorData::GetDataType);

    py::class_<KernelModule, KernelModulePtr>(m, "KernelModule").def(py::init<py::object &>());
}
} // namespace pypto
