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
#include "interface/configs/config_manager.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/device/dynamic/costmodel_utils.h"
#include "runtime.h"
#include "device_runner.h"
#include "cost_model/simulation/backend.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

constexpr uint32_t kDefaultAicNum = 25;
constexpr uint32_t kDefaultAivNum = 50;

struct MemoryHelper {
    MemoryHelper(bool isTest) : isTest_(isTest) {
        if (!isTest_) {
            l2Offset = machine::GetRA()->GetL2Offset();
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
            machine::GetRA()->AllocDevAddr(&devPtr, size);
        return devPtr;
    }

    void CopyFromDev(RawTensorData &t) { CopyFromDev(t.data(), t.GetDevPtr(), t.size()); }

    bool isTest_{true};
    uint64_t l2Offset{0};
};

extern "C" int DynTileFwkBackendKernelServer(void *targ);
extern "C" int DynTileFwkBackendKernelServerInit(void *targ);

struct DynFuncRunnerConfig {
    bool onBoard{true};
    int blockdim{25};
    int aicpuNum{5};
    int64_t dynWorkspaceSize{0};
    int64_t repeatNum{1};

    DynFuncRunnerConfig() = default;
    DynFuncRunnerConfig(bool onboard, int tblockdim, int taicpunum) : onBoard(onboard), blockdim(tblockdim), aicpuNum(taicpunum) {}
    DynFuncRunnerConfig(int tdynWorkspaceSize) : dynWorkspaceSize(tdynWorkspaceSize) {}
    DynFuncRunnerConfig(int tdynWorkspaceSize, int64_t trepeatNum) : dynWorkspaceSize(tdynWorkspaceSize),
                                                                     repeatNum(trepeatNum){}
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
        runner.KernelLaunchPrecheck(funcop, inputs, outputs);
        runner.RunModel(inputs, outputs);
        runner.Run(inputs, outputs);
    }

    // Run with incast/outcast from ProgramData
    static void Run(std::shared_ptr<DyndevFunctionAttribute> funcop, const DynFuncRunnerConfig &config = DynFuncRunnerConfig()) {
        auto runner = DynFuncRunner(funcop->devProgBinary, config);
        auto &inputs = ProgramData::GetInstance().GetInputDataList();
        auto &outputs = ProgramData::GetInstance().GetOutputDataList();
        runner.KernelLaunchPrecheck(funcop, inputs, outputs);
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
    void InitTilingData(AstKernelArgs &kArgs, bool isTest) {
        MemoryHelper h{isTest};
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProg_.data()));
        devProg->devArgs.nrAic = kDefaultAicNum;
        devProg->devArgs.nrAiv = kDefaultAivNum;
        devProg->devArgs.nrAicpu = config_.aicpuNum;
        devProg->devArgs.nrValidAic = config_.blockdim;
        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;
        devProg->workspaceSize = devProg->aicoreLocalWorkspaceSize + devProg->aicpuCoherentWorkspaceSize
                                 + config_.dynWorkspaceSize;
        kArgs.workspace = (int64_t *)h.AllocDev(devProg->workspaceSize);
        kArgs.cfgdata = (int64_t *)h.CopyToDev(devProg_);
        kArgs.machineConfig = devProg->devArgs.machineConfig;
        return;
    }

    void RunModel(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
            AstKernelArgs kArgs;
            InitTilingData(kArgs, true);
            for (int i = 0; i < config_.repeatNum; i++) {
                InitKernelInOuts(kArgs, inputs, outputs, true);
                std::cout << "!!! Run CostModel " << i << "\n";
                RunCostModel(&kArgs);
                std::cout << "!!! Run TestModel " << i << "\n";
                RunTestMode(&kArgs);
            }
            RunDynCostModel();
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
            AstKernelArgs kArgs;
            InitTilingData(kArgs, false);
            auto stream = machine::GetRA()->GetStreamAICPU();
            for (int i = 0; i < config_.repeatNum; i++) {
              InitKernelInOuts(kArgs, inputs, outputs, false);
              rc = DeviceRunner::Get().DynamicRun(stream, 0, &kArgs, config_.blockdim, config_.aicpuNum);
              EXPECT_EQ(rc, 0);
            }
            CopyFromDev(outputs, false);
            if (HasInplaceArgs())
                CopyFromDev(inputs, false);
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

    void RunDynCostModel()
    {
        if (!config::GetPlatformConfig("ENABLE_DYN_FULL_COST_MODEL", true)) {
            return;
        }
        config::SetSimConfig("SIM_MODE", CostModel::SimMode::NORMAL);
        CostModelAgent costModelAgent;
        std::string path = "./output/dyn_topo.txt";
        costModelAgent.SubmitTopo(path);
        costModelAgent.SubmitLeafFunctionsToCostModel();
        costModelAgent.RunCostModel();
        costModelAgent.TerminateCostModel();
    }

    void RunTestMode(AstKernelArgs *kArgs) {
        (void) kArgs;
        std::thread aicpus[6];
        std::atomic<int> idx{0};
        auto *devProg = (DevAscendProgram *)(kArgs->cfgdata);
        auto rc0 = DynTileFwkBackendKernelServerInit(kArgs);
        EXPECT_EQ(rc0, 0);
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
                auto rc = DynTileFwkBackendKernelServer(kArgs);
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

    void InitKernelInOuts(AstKernelArgs &kArgs, const std::vector<RawTensorDataPtr> &inputTensors,
        const std::vector<RawTensorDataPtr> &outputTensors, bool isTest = true) {
        MemoryHelper h{isTest};

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
        for (auto &input: inputTensors) {
            if (input)
                input->SetDevPtr(nullptr);
        }
        for (auto &output: outputTensors) {
            if (output)
                output->SetDevPtr(nullptr);
        }
        kArgs.inputs = buildInouts(inputTensors);
        kArgs.outputs = buildInouts(outputTensors);
        ALOG_INFO_F("Inputs %p outputs %p workspace %p cfgdata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
            kArgs.cfgdata);
        return;
    }

    void KernelLaunchPrecheck(std::shared_ptr<DyndevFunctionAttribute> funcop,
        const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
        auto checkInouts = [&](std::vector<std::reference_wrapper<const Tensor>> &tensorList,
                               const std::vector<RawTensorDataPtr> &dataList) {
            EXPECT_EQ(tensorList.size(), dataList.size()) << "argument num not match !!!!";
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

        checkInouts(funcop->startArgsInputTensorList, inputs);
        checkInouts(funcop->startArgsOutputTensorList, outputs);
    }

private:
    const std::vector<uint8_t> &devProg_;
    DynFuncRunnerConfig config_;
};
