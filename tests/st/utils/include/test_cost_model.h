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
 * \file test_cost_model.h
 * \brief
 */

#pragma once

#include <gtest/gtest.h>
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/host/device_runner.h"
#include "simulation/pv/PvModel.h"
#include "simulation/pv/PvModelFactory.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
using namespace CostModel;

extern "C" int DynTileFwkNSAKernelServer(void *targ);

class CostModelDynFuncRunner {
public:
    CostModelDynFuncRunner(Function *func): func_(func), devProg_(func->GetDyndevAttribute()->devProgBinary) {
        pv_ = CostModel::PvModelFactory::CreateDyn();
    }

    static void Run(Function* func,
        const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
            auto runner = CostModelDynFuncRunner(func);
            runner.RunModel(inputs, outputs);
    }

    // Run with incast/outcast from ProgramData
    static void Run(Function* func) {
        auto runner = CostModelDynFuncRunner(func);
        auto &inputs = ProgramData::GetInstance().GetInputDataList();
        auto &outputs = ProgramData::GetInstance().GetOutputDataList();
        runner.RunModel(inputs, outputs);
    }

private:

    void RunModel(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
            auto funcop = func_->GetDyndevAttribute();
            KernelLaunchPrecheck(funcop);
            pv_->Codegen(func_);

            for (int i = 0; i < 1; i++) {
                AstKernelArgs kArgs = BuildKernelArgs(inputs, outputs);
                std::cout << "!!! Run CostModel " << i << "\n";
                RunTestMode(&kArgs);
            }
    }

    bool HasInplaceArgs() {
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProg_.data()));
        return devProg->inplaceSlotList.size() != 0;
    }

    void RunTestMode(AstKernelArgs *kArgs) {
        (void) kArgs;
        constexpr int threadNum = 6;
        std::thread aicpus[threadNum];
        std::atomic<int> idx{0};
        for (int i = 0; i < threadNum; i++) {
            aicpus[i] = std::thread([&]() {
                int tidx = idx++;
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(tidx, &cpuset);
                constexpr int nameLen = 64;
                char name[nameLen];
                sprintf_s(name, sizeof(name), "aicput%d", tidx);
                pthread_setname_np(pthread_self(), name);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            });
        }

        for (int i = 0; i < threadNum; i++) {
            aicpus[i].join();
        }
    }

    void CopyFromDev(const std::vector<RawTensorDataPtr> &outputs) {
        for (auto &output : outputs) {
            if (output)
                pv_->CopyFromDev(output->data(), output->GetDevPtr(), output->size());
        }
    }

    AstKernelArgs BuildKernelArgs(const std::vector<RawTensorDataPtr> &inputs,
        const std::vector<RawTensorDataPtr> &outputs) {
        AstKernelArgs kArgs;

        auto buildInouts = [&](auto &tensorList) {
            std::vector<DevAscendTensorData> geTensors;
            for (auto &t : tensorList) {
                if (t) {
                    auto addrs = pv_->CopyToDev((uint8_t*)t->data(), t->size());
                    geTensors.emplace_back(DevAscendTensorDataCreator::Create((uint64_t)addrs, t->GetShape()));
                } else {
                    std::vector<int> shape;
                    geTensors.emplace_back(DevAscendTensorDataCreator::Create(0UL, shape));
                }
            }
            auto outs = DevAscendTensorDataCreator::Encode(geTensors);
            return (int64_t*)pv_->CopyToDev((uint8_t *)outs.data(), outs.size()*sizeof(int64_t));
        };

        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProg_.data()));
        devProg->devArgs.nrAic = 25;
        devProg->devArgs.nrAiv = 50;
        devProg->devArgs.nrAicpu = 6;
        devProg->devArgs.nrValidAic = 24;
        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;

        for (auto &input: inputs) {
            if (input)
                input->SetDevPtr(nullptr);
        }
        for (auto &output: outputs) {
            if (output)
                output->SetDevPtr(nullptr);
        }

        kArgs.inputs = buildInouts(inputs);
        kArgs.outputs = buildInouts(outputs);
        kArgs.workspace = (int64_t *)pv_->AllocDev(devProg->aicoreLocalWorkspaceSize + devProg->aicpuCoherentWorkspaceSize);
        kArgs.tilingdata = (int64_t *)pv_->CopyToDev(devProg_.data(), devProg_.size());
        kArgs.machineConfig  = devProg->devArgs.machineConfig;
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
                    EXPECT_EQ(t->GetShape(), d->GetShape());
                }
            }
        };

        checkInouts(funcop->startArgsInputTensorList, ProgramData::GetInstance().GetInputDataList());
        checkInouts(funcop->startArgsOutputTensorList, ProgramData::GetInstance().GetOutputDataList());
    }

private:
    Function *func_;
    const std::vector<uint8_t> &devProg_;
    std::shared_ptr<DynPvModel> pv_;
};
