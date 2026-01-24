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

#include <ctime>
#include <utility>
#include <vector>
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/utils/op_info_manager.h"
#include "machine/runtime/device_launcher_binding.h"
#include "machine/runtime/emulation_launcher.h"
#include "machine/host/perf_analysis.h"
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

struct ControlFlowCache {
    int64_t hash;
    std::vector<DeviceTensorData> inputs;
    uint8_t *devCache;

    ControlFlowCache(std::vector<DeviceTensorData> &datas, uint8_t *tcache)
        : inputs(std::move(datas)), devCache(tcache) {
        hash = Hash(inputs);
    }

    static int64_t Hash(std::vector<DeviceTensorData> &datas) {
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

    static int64_t Hash(std::vector<std::vector<int64_t>> &shapes) {
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

#define AICPU_META_BUFFER_NUM 2

struct KernelBinary {
    int64_t devId{0};
    Function *func{nullptr};
    void *kernelBin{nullptr};
    DevAscendProgram *devProg{nullptr};
    DyndevFunctionAttribute *dynAttr{nullptr};
    int64_t workspaceSize{0}; // static workspace size
    std::vector<ControlFlowCache> caches;

    std::vector<void *> devMems;
    std::vector<OpMetaAddrs> metas;
    int64_t metaIndex{0};

    std::vector<int64_t> inputInfo;
    AiCpuArgs *aicpuArgs{nullptr};
    uint64_t l2Offset{0};

    uint8_t *FindCtrlFlowCache(std::vector<std::vector<int64_t>> &inputs) {
        int64_t inHash = ControlFlowCache::Hash(inputs);
        for (auto &cache : caches) {
            if (cache.hash == inHash) {
                return cache.devCache;
            }
        }
        return nullptr;
    }

    bool ControlFlowCacheEnable() const { return devProg->ctrlFlowCacheSize != 0; }

    DevControlFlowCache *BuildControlFlowCache(std::vector<DeviceTensorData> &inputs, bool cache) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        DevControlFlowCache *ctrlCache = nullptr;

        int ret = EmulationLauncher::BuildControlFlowCache(func, inputs, {}, &ctrlCache, config);
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

    KernelBinary(int64_t tdevId, Function *funcp) : devId(tdevId), func(funcp) {
        dynAttr = func->GetDyndevAttribute().get();
        devProg = (DevAscendProgram *)dynAttr->devProgBinary.data();
        kernelBin = RegisterAicoreKernel();
        workspaceSize = devProg->memBudget.Total();
        InitCachedArgs();
        DeviceLauncher::FillDeviceKernelArgs(dynAttr->devProgBinary, aicpuArgs->kArgs);
        ASSERT(aicpuArgs->kArgs.inputs == inputInfo.data());
    }

    int64_t GetWorkspaceSize(std::vector<DeviceTensorData> &tensors) {
        if (dynAttr->maxDynamicAssembleOutcastMem.IsValid()) {
            Evaluator eval{dynAttr->inputSymbolDict, tensors, {}};
            return workspaceSize + eval.Evaluate(dynAttr->maxDynamicAssembleOutcastMem);
        }
        return workspaceSize;
    }

    AiCpuArgs *BuildKernelArgs(std::vector<DeviceTensorData> &tensors) {
        auto &disableL2List = dynAttr->disableL2List;
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
        aicpuArgs->kArgs.opMetaAddrs = metas[metaIndex % AICPU_META_BUFFER_NUM];
        return aicpuArgs;
    }

    bool Match(std::vector<std::reference_wrapper<Tensor>> &tensors) {
        auto size = tensors.size();
        ASSERT(size == dynAttr->startArgsInputLogicalTensorList.size()) << "mismatch input size";
        for (size_t i = 0; i < size; ++i) {
            auto &startArg = dynAttr->startArgsInputLogicalTensorList[i];
            auto t = tensors[i].get();
            if (startArg->shape == t.GetShape() && startArg->Datatype() == t.GetDataType() &&
                startArg->Format() == t.Format()) {
                return true;
            }
        }
        return false;
    }

    ~KernelBinary() {
        rtDevBinaryUnRegister(kernelBin);
        for (auto ptr : devMems) {
            rtFree(ptr);
        }
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

    void *AllocDevMem(int64_t size) {
        void *devPtr = nullptr;
        int ret = rtMalloc(&devPtr, size, RT_MEMORY_HBM, 0);
        if (ret != RT_ERROR_NONE) {
            ALOG_ERROR("malloc dev mem failed, ret: %d", ret);
        }
        devMems.emplace_back(devPtr);
        return devPtr;
    }

    void InitCachedArgs() {
        auto genSize = devProg->memBudget.metadata.general;
        auto stitchPoolSize = devProg->memBudget.metadata.general;
        for (int i = 0; i < AICPU_META_BUFFER_NUM; ++i) {
            OpMetaAddrs opMeta;
            opMeta.generalAddr = (uint64_t)AllocDevMem(genSize);
            opMeta.stitchPoolAddr = (uint64_t)AllocDevMem(stitchPoolSize);
            metas.push_back(opMeta);
        }

        auto argNum = dynAttr->startArgsInputLogicalTensorList.size() +
            dynAttr->startArgsOutputLogicalTensorList.size();
        auto argSize = sizeof(AiCpuArgs) + 2 * sizeof(int64_t) + argNum * sizeof(DevTensorData);
        ASSERT(argSize % 8 == 0);
        inputInfo.resize(argSize / 8);
        aicpuArgs = new (inputInfo.data()) AiCpuArgs();
        aicpuArgs->kArgs.inputs = inputInfo.data();
        aicpuArgs->kArgs.outputs = (int64_t *)argSize;

        int64_t *inputp = (int64_t *)(aicpuArgs + 1);
        inputp[0] = dynAttr->startArgsInputLogicalTensorList.size();
        inputp[1] = dynAttr->startArgsOutputLogicalTensorList.size();

        l2Offset = machine::GetRA()->GetL2Offset();
    }
};

struct KernelModule {
    KernelBinary *FindFunction(int32_t devId, std::vector<std::reference_wrapper<Tensor>> &tensors) {
        for (auto &kbinary : kernels) {
            if (kbinary.devId == devId && kbinary.Match(tensors)) {
                return &kbinary;
            }
        }
        return nullptr;
    }

    KernelBinary *AddFunction(int32_t devId, std::shared_ptr<Function> func) {
        kernels.emplace_back(devId, func.get());
        kfuncs.emplace_back(func);
        return &kernels.back();
    }

    void Launch(KernelBinary *kbinary, aclrtStream aicpuStream, aclrtStream aicoreStream,
        std::vector<DeviceTensorData> &tensors, uint8_t *ctrlFlowCache, int64_t *workspace) {
        auto args = kbinary->BuildKernelArgs(tensors);
        rtAicpuArgs.args = args->kArgs.inputs;
        rtAicpuArgs.argsSize = (int64_t)args->kArgs.outputs;

        args->kArgs.launchMode = AICPU_LAUNCH_MODE_CTRL;
        args->kArgs.ctrlFlowCache = (int64_t *)ctrlFlowCache;
        args->kArgs.workspace = workspace;

        int ret = rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU_KFC,
            "AST_DYN_AICPU", 5, &rtAicpuArgs, nullptr, aicpuStream, 0);
        ASSERT(ret == RT_ERROR_NONE) << "launch aicpu ctrl failed: " << ret;

        // ALOG_ERROR(__FUNCTION__, __LINE__);
        // args->kArgs.launchMode = AICPU_LAUNCH_MODE_SCHED;
        // ret = rtAicpuKernelLaunchExWithArgs(
        //     rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 3, &rtAicpuArgs, nullptr, aicpuStream, 0);
        // ASSERT(ret == RT_ERROR_NONE) << "launch aicpu sched failed: " << ret;

        kernelArgs[5] = args->kArgs.cfgdata; // 5 is cfgdata
        auto tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
        ret = rtKernelLaunchWithHandleV2(
            kbinary->kernelBin, tilingKey, blockDim, &rtAicoreArgs, nullptr, aicoreStream, &rtTaskCfg);
        ASSERT(ret == RT_ERROR_NONE) << "launch aicore failed: " << ret;
    }

    KernelModule() {
        memset_s(&rtAicpuArgs, sizeof(rtAicpuArgsEx_t), 0, sizeof(rtAicpuArgsEx_t));
        rtAicpuArgs.kernelNameAddrOffset = offsetof(dynamic::AiCpuArgs, kernelName);
        rtAicpuArgs.soNameAddrOffset = offsetof(dynamic::AiCpuArgs, soName);
        rtAicpuArgs.hostInputInfoNum = 1;
        hostInfo.addrOffset = offsetof(dynamic::AiCpuArgs, kArgs.inputs);
        hostInfo.dataOffset = sizeof(dynamic::AiCpuArgs);
        rtAicpuArgs.hostInputInfoPtr = &hostInfo;

        memset_s(&rtAicoreArgs, sizeof(rtArgsEx_t), 0, sizeof(rtArgsEx_t));
        kernelArgs.resize(6, nullptr);
        rtAicoreArgs.args = kernelArgs.data();
        rtAicoreArgs.argsSize = kernelArgs.size() * sizeof(void *);

        memset_s(&rtTaskCfg, sizeof(rtTaskCfgInfo_t), 0, sizeof(rtTaskCfgInfo_t));
        rtTaskCfg.schemMode = RT_SCHEM_MODE_BATCH;

        blockDim = dynamic::GetCfgBlockdim();
    }

    rtHostInputInfo_t hostInfo;
    rtAicpuArgsEx_t rtAicpuArgs;

    int blockDim;
    rtArgsEx_t rtAicoreArgs;
    rtTaskCfgInfo_t rtTaskCfg;
    std::vector<void *> kernelArgs;

    std::vector<std::shared_ptr<Function>> kfuncs;
    std::vector<KernelBinary> kernels;
};
using KernelModulePtr = std::shared_ptr<KernelModule>;

struct AclModeGuard {
    AclModeGuard(aclmdlRICaptureMode tmode) : mode(tmode) { aclmdlRICaptureThreadExchangeMode(&mode); }
    ~AclModeGuard() { aclmdlRICaptureThreadExchangeMode(&mode); }
    aclmdlRICaptureMode mode;
};

struct DeviceGuard {
    DeviceGuard(int32_t devId): nDevId(devId) {
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

    int32_t oDevId {0};
    int32_t nDevId {0};
};

static Function *Compile(py::object module, py::args args, py::kwargs kwargs) {
    Program::GetInstance().Reset();
    auto compile = py::getattr(module, "compile");
    compile(args, kwargs);
    return Program::GetInstance().GetLastFunction();
}

static void BuildDefaultCache(KernelBinary *kbinary, py::object module, std::vector<DeviceTensorData> &tensors) {
    auto infershape = py::getattr(module, "infer_controlflow_shape");
    if (!infershape.is_none()) {
        auto cfshapes = infershape().cast<py::list>();
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
            kbinary->BuildControlFlowCache(inputs, true);
        }
    }
}

static uint8_t *BuildTempCache(KernelBinary *kbinary, py::object module, std::vector<DeviceTensorData> &tensors) {
    auto ctrlCache = kbinary->BuildControlFlowCache(tensors, false);
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

static uint8_t *FindCtrlCache(KernelBinary *kbinary, py::object module, py::args args) {
    auto infershape = py::getattr(module, "infer_controlflow_shape");
    if (!infershape.is_none()) {
        py::list oriShapes;
        for (auto &pt : args) {
            auto shape = py::getattr(pt, "ori_shape");
            if (!shape.is_none()) {
                oriShapes.append(shape);
            }
        }
        auto cfshape = infershape(*oriShapes);
        if (cfshape.is_none()) {
            return nullptr;
        }
        auto shape = cfshape.cast<std::vector<std::vector<int64_t>>>();
        return kbinary->FindCtrlFlowCache(shape);
    }
    return nullptr;
}

static int GetInputTensors(py::args args, std::vector<DeviceTensorData> &tensors,
    std::vector<std::reference_wrapper<Tensor>> &ref_tensors) {
    py::object device = py::none();
    for (auto &pt : args) {
        auto base = py::getattr(pt, "_base");
        if (py::isinstance<Tensor>(base)) {
            auto &t = base.cast<Tensor &>();
            auto data_ptr = py::cast<int64_t>(py::getattr(pt, "data_ptr"));
            auto shape = py::cast<std::vector<int64_t>>(py::getattr(pt, "ori_shape"));
            tensors.emplace_back(t.GetDataType(), data_ptr, shape);
            ref_tensors.emplace_back(t);
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

static bool AttachAicpuStream(aclrtStream aicoreStream, aclrtStream aicpuStream) {
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
        rtStreamAddToModel(aicpuStream, rtModel);
        return true;
    }
    return false;
}

void LaunchKernel(py::object module, int64_t stream, py::args args, py::kwargs kwargs) {
    auto t0 = GetTimeMonotonic();
    auto aicoreStream = (aclrtStream)stream;
    auto aicpuStream = (aclrtStream)DeviceGetAicpuStream();

    std::vector<DeviceTensorData> tensors;
    std::vector<std::reference_wrapper<Tensor>> ref_tensors;
    auto devId = GetInputTensors(args, tensors, ref_tensors);
    DeviceGuard devGuard(devId);

    auto t1 = GetTimeMonotonic();
    auto kmodule = py::getattr(module, "kmodule").cast<KernelModulePtr>();
    auto kbinary = kmodule->FindFunction(devId, ref_tensors);
    if (kbinary == nullptr) {
        Program::GetInstance().Reset();
        // Set capture mode to relaxed to support rtmemcpy / rtmemset
        AclModeGuard guard(ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
        auto func = Compile(module, args, kwargs);
        kbinary = kmodule->AddFunction(devId, Program::GetInstance().GetFunctionSharedPtr(func));
        BuildDefaultCache(kbinary, module, tensors);
    }
    auto t2 = GetTimeMonotonic();
    uint8_t *ctrlFlowCache = nullptr;
    if (kbinary->ControlFlowCacheEnable()) {
        ctrlFlowCache = FindCtrlCache(kbinary, module, args);
    }
    auto captured = AttachAicpuStream(aicoreStream, aicpuStream);
    if (ctrlFlowCache == nullptr && captured) {
        ctrlFlowCache = BuildTempCache(kbinary, module, tensors);
    }
    auto t3 = GetTimeMonotonic();

    int64_t *wsAddr = nullptr;
    int64_t wsSize = kbinary->GetWorkspaceSize(tensors);
    if (wsSize) {
        auto pyalloc = py::getattr(module, "alloc");
        wsAddr = (int64_t *)pyalloc(wsSize).cast<int64_t>();
    }
    auto t4 = GetTimeMonotonic();

    kmodule->Launch(kbinary, aicpuStream, aicoreStream, tensors, ctrlFlowCache, wsAddr);
    auto t5 = GetTimeMonotonic();

    ALOG_ERROR("LaunchKernel time: %lu, %lu, %lu, %lu, %lu", t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4);
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

    py::class_<KernelModule, KernelModulePtr>(m, "KernelModule").def(py::init<>());
}
} // namespace pypto
