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
 * \file test_dynamic.h
 * \brief
 */

#pragma once

#include <gtest/gtest.h>
#include <cstdint>
#include "interface/interpreter/raw_tensor_data.h"
#include "runtime/utils/dynamic/dev_encode.h"
#include "runtime/device/dynamic/costmodel_utils.h"
#include "runtime/runtime.h"
#include "runtime/host/device_runner.h"
#include "simulation/backend.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;


struct MemoryHelper {
    MemoryHelper(bool isTest) : isTest_(isTest) {
        if (!isTest_) {
            l2Offset = runtime::GetRA()->GetL2Offset();
        }
    }

    uint8_t *CopyToDev(uint8_t *data, uint64_t size) {
        uint8_t *devPtr = AllocDev(size);
        if (isTest_)
            memcpy_s(devPtr, size, data, size);
        else
            rtMemcpy(devPtr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
        return devPtr;
    }

    void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) {
        if (isTest_)
            memcpy_s(data, size, devPtr, size);
        else
            rtMemcpy(data, size, devPtr, size, RT_MEMCPY_DEVICE_TO_HOST);
    }

    template <typename T>
    T *CopyToDev(std::vector<T> data) {
        return (T *)CopyToDev((uint8_t *)data.data(), data.size() * sizeof(T));
    }

    uint8_t *CopyToDev(RawTensorData &data) {
        if (data.GetDevPtr() == nullptr) {
            auto devPtr = CopyToDev((uint8_t *)data.data(), data.size());
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    uint8_t *AllocZero(uint64_t size) {
        uint8_t *devPtr = AllocDev(size);
        if (isTest_)
            memset(devPtr, 0, size);
        else
            rtMemset(devPtr, size, 0, size);
        return devPtr;
    }

    uint8_t *AllocDev(size_t size) {
        uint8_t *devPtr = nullptr;
        if (isTest_)
            devPtr = (uint8_t *)malloc(size);
        else
            runtime::GetRA()->AllocDevAddr(&devPtr, size);
        return devPtr;
    }

    void CopyFromDev(RawTensorData &t) { CopyFromDev(t.data(), t.GetDevPtr(), t.size()); }

    bool isTest_{true};
    uint64_t l2Offset{0};
};

extern "C" int DynamicServerKernel(void *targ);

struct DynFuncRunnerConfig {
    bool onBoard{true};
    int blockdim{25};
    int aicpunum{5};
    int64_t dynWorkspaceSize{0};

    DynFuncRunnerConfig() = default;
    DynFuncRunnerConfig(bool onboard, int tblockdim, int taicpunum) : onBoard(onboard), blockdim(tblockdim), aicpunum(taicpunum) {}
    DynFuncRunnerConfig(int tdynWorkspaceSize) : dynWorkspaceSize(tdynWorkspaceSize) {}
};

class DynFuncRunner {
public:
    DynFuncRunner(const std::vector<uint8_t> &devProg, const DynFuncRunnerConfig &config)
        : devProg_(devProg), config_(config){}

    static void RunModel(std::shared_ptr<DyndevFunctionAttribute> funcop, const std::vector<RawTensorDataPtr> &inputs,
        const std::vector<RawTensorDataPtr> &outputs, const DynFuncRunnerConfig &config = DynFuncRunnerConfig()) {
        auto runner = DynFuncRunner(funcop->devProgBinary, config);
        runner.RunModel(inputs, outputs);
    }

    static void Run(std::shared_ptr<DyndevFunctionAttribute> funcop, const std::vector<RawTensorDataPtr> &inputs,
        const std::vector<RawTensorDataPtr> &outputs, const DynFuncRunnerConfig &config = DynFuncRunnerConfig()) {
        auto runner = DynFuncRunner(funcop->devProgBinary, config);
        runner.RunModel(inputs, outputs);
        runner.Run(inputs, outputs);
    }

    // Run with incast/outcast from ProgramData
    static void Run(std::shared_ptr<DyndevFunctionAttribute> funcop, const DynFuncRunnerConfig &config = DynFuncRunnerConfig()) {
        auto runner = DynFuncRunner(funcop->devProgBinary, config);
        runner.KernelLaunchPrecheck(funcop);
        auto &inputs = ProgramData::GetInstance().GetInputDataList();
        auto &outputs = ProgramData::GetInstance().GetOutputDataList();
        runner.RunModel(inputs, outputs);
        if (config.onBoard) {
            runner.Run(inputs, outputs);
        }
    }

    static void Run(const std::vector<uint8_t> &devProg, const std::vector<RawTensorDataPtr> &inputs,
        const std::vector<RawTensorDataPtr> &outputs, const DynFuncRunnerConfig &config = DynFuncRunnerConfig()) {
        auto runner = DynFuncRunner(devProg, config);
        runner.RunModel(inputs, outputs);
        runner.Run(inputs, outputs);
    }

private:

    void RunModel(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
            for (int i = 0; i < 1; i++) {
                AstKernelArgs kArgs = BuildKernelArgs(inputs, outputs, true);
                std::cout << "!!! Run CostModel " << i << "\n";
                RunCostModel(&kArgs);
                std::cout << "!!! Run TestModel " << i << "\n";
                RunTestMode(&kArgs);
            }
    }

    bool HasInplaceArgs() {
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProg_.data()));
        return devProg->inplaceSlotList.size() != 0;
    }

    void Run(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
        std::cout << "!!! Kernel Launch " << "\n";
        int rc = aclInit(nullptr);
        if (rc == 0 || rc == ACL_ERROR_REPEAT_INITIALIZE) {
            rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
            AstKernelArgs kArgs = BuildKernelArgs(inputs, outputs, false);
            auto stream = runtime::GetRA()->GetStreamAICPU();
            rc = DeviceRunner::Get().DynamicRun(stream, 0, &kArgs, config_.blockdim, config_.aicpunum);
            CopyFromDev(outputs, false);
            if (HasInplaceArgs())
                CopyFromDev(inputs, false);
            EXPECT_EQ(rc, 0);
        }
    }

    void RunCostModel(AstKernelArgs *kArgs) {
        if (!config::GetPlatformConfig("ENABLE_DYN_COST_MODEL", true)) {
            return;
        }
        Function *function = Program::GetInstance().GetLastFunction();
        if (function == nullptr) {
            return;
        }
        config::SetSimConfig("SIM_MODE", CostModel::SimMode::LEAF_FUNCTION);
        CostModelAgent costModelAgent;
        costModelAgent.SubmitLeafFunctionsToCostModel();
        costModelAgent.RunCostModel();
        costModelAgent.TerminateCostModel();
        CostModel::ModelData* modelData = new CostModel::ModelData();
        auto attr = function->GetDyndevAttribute();
        modelData->functionTime.resize(attr->devLeafIndex2Hash.size(), 0);
        for (const auto& [index, hash] : attr->devLeafIndex2Hash) {
            auto time = costModelAgent.GetLeafFunctionTimeCost(hash);
            DEV_INFO("devLeafIndex2Hash, %d -> %lu: %lu\n", index, hash, time);
            modelData->functionTime[index] = time;
        }
        kArgs->costmodeldata = modelData;
    }

    void RunTestMode(AstKernelArgs *kArgs) {
        (void) kArgs;
        std::thread aicpus[6];
        std::atomic<int> idx{0};
        auto *devProg = (DevAscendProgram *)(kArgs->tilingdata);
        for (int i = 0; i < static_cast<int>(devProg->devArgs.nrAicpu); i++) {
            aicpus[i] = std::thread([&]() {
                int tidx = idx++;
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(tidx, &cpuset);
                char name[64];
                sprintf(name, "aicput%d", tidx);
                std::cout << "start thread: " << name << std::endl;
                pthread_setname_np(pthread_self(), name);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                auto rc = DynamicServerKernel(kArgs);
                EXPECT_EQ(rc, 0);
            });
        }

        for (int i = 0; i < 6; i++) {
            if (aicpus[i].joinable()) {
                aicpus[i].join();
            }
        }
    }

    void CopyFromDev(const std::vector<RawTensorDataPtr> &outputs, bool isTest_) {
        MemoryHelper h{isTest_};
        for (auto &output : outputs) {
            if (output)
                h.CopyFromDev(*output);
        }
    }

    AstKernelArgs BuildKernelArgs(const std::vector<RawTensorDataPtr> &inputs,
        const std::vector<RawTensorDataPtr> &outputs, bool isTest_ = true) {
        AstKernelArgs kArgs;
        MemoryHelper h{isTest_};

        auto buildInouts = [&](auto &tensorList) {
            std::vector<DevAscendTensorData> geTensors;
            for (auto &t : tensorList) {
                if (t) {
                    auto addrs = h.CopyToDev(*t);
                    if (t->l2Disable_) {
                        addrs += h.l2Offset;
                    }
                    geTensors.emplace_back(DevAscendTensorDataCreator::Create((uint64_t)addrs, t->GetShape()));
                    ALOG_ERROR_F("addrs is %zu, ptr %p\n", t->GetSize(), addrs);
                } else {
                    std::vector<int> shape;
                    geTensors.emplace_back(DevAscendTensorDataCreator::Create(0UL, shape));
                }
            }
            auto outs = DevAscendTensorDataCreator::Encode(geTensors);
            return h.CopyToDev(outs);
        };

        // auto *devProg = (DevAscendProgram *)devProg_.data();
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProg_.data()));
        devProg->devArgs.nrAic = 25;
        devProg->devArgs.nrAiv = 50;
        devProg->devArgs.nrAicpu = config_.aicpunum;
        devProg->devArgs.nrValidAic = config_.blockdim;
        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;

        for (auto &input: inputs) {
            if (input)
                input->SetDevPtr(nullptr);
        }
        for (auto &output: outputs) {
            if (output)
                output->SetDevPtr(nullptr);
        }
        kArgs.workspaceSize = devProg->aicoreLocalWorkspaceSize +
            devProg->aicpuCoherentWorkspaceSize + config_.dynWorkspaceSize;
        kArgs.inputs = buildInouts(inputs);
        kArgs.outputs = buildInouts(outputs);
        kArgs.workspace = (int64_t *)h.AllocDev(kArgs.workspaceSize);
        kArgs.tilingdata = (int64_t *)h.CopyToDev(devProg_);
        kArgs.machineConfig  = devProg->devArgs.machineConfig;
        ALOG_INFO_F("inputs %p outputs %p workspace %p tiledata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
            kArgs.tilingdata);
        return kArgs;
    }

    void KernelLaunchPrecheck(std::shared_ptr<DyndevFunctionAttribute> funcop) {
        auto checkInouts = [&](std::vector<std::reference_wrapper<const Tensor>> &tensorList,
                               const std::vector<RawTensorDataPtr> &dataList) {
            for (size_t i = 0; i < tensorList.size(); i++) {
                auto &t = tensorList[i].get();
                auto &d = dataList[i];
                if (d) {
                    EXPECT_EQ(t.GetDataType(), d->GetDataType());
                    auto rawShape = t->GetRawTensor()->GetDynRawShape();
                    auto shape = d->GetShape();
                    for (size_t k = 0; k < rawShape.size(); k++) {
                        if (rawShape[k].IsImmediate()) {
                            EXPECT_EQ(rawShape[k].Concrete(), shape[k]);
                        }
                    }
                }
            }
        };

        checkInouts(funcop->startArgsInputTensorList, ProgramData::GetInstance().GetInputDataList());
        checkInouts(funcop->startArgsOutputTensorList, ProgramData::GetInstance().GetOutputDataList());
    }

private:
    const std::vector<uint8_t> &devProg_;
    DynFuncRunnerConfig config_;
};
