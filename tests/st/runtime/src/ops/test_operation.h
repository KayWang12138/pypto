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
 * \file test_operation.h
 * \brief
 */

#pragma once
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <functional>

#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "runtime/utils/dynamic/dev_encode.h"
#include "test_dynamic.h"
#include "interface/tensor/float.h"

namespace ascend {
namespace test_operation {

struct OpFuncArgs {
};

using OpFunc = std::function<void(
    const std::vector<Tensor>&,
    std::vector<Tensor>&,
    const OpFuncArgs*
)>;

struct TestCaseDesc {
    std::vector<Tensor> inputTensors;
    std::vector<Tensor> outputTensors;
    std::vector<std::string> inputPaths;
    std::vector<std::string> goldenPaths;
    const OpFuncArgs* args;
    OpFunc opFunc;
};

class TestExecutor {
public:
    static void runTest(const TestCaseDesc& testCase) {
        init();
        verifyOpResults(testCase);
    }

private:
    static void init() {
        config::SetHostConfig(npu::tile_fwk::KEY_ONLY_CODEGEN, true);
        config::SetCodeGenConfig(npu::tile_fwk::KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    }

    static void verifyOpResults(const TestCaseDesc& testCase) {
        // 设置输入数据
        std::vector<RawTensorDataPtr> ascendInputs;
        ASSERT_EQ(testCase.inputTensors.size(), testCase.inputPaths.size());
        for (size_t i = 0; i < testCase.inputTensors.size(); ++i) {
            size_t elementCount = 1;
            for (int dim : testCase.inputTensors[i].GetShape()) {
                elementCount *= dim;
            }
            std::vector<uint8_t> input(elementCount * BytesOf(testCase.inputTensors[i].GetDataType()), 0);
            readInput<uint8_t>(testCase.inputPaths[i], input);
            ascendInputs.push_back(RawTensorData::CreateTensor(testCase.inputTensors[i], input));
        }
        ProgramData::GetInstance().AppendInputs({ascendInputs});
        
        // 设置输出Tensor
        std::vector<RawTensorDataPtr> ascendOutputs;
        for (const auto& tensor : testCase.outputTensors) {
            ascendOutputs.push_back(RawTensorData::CreateTensorZero(tensor));
        }
        ProgramData::GetInstance().AppendOutputs({ascendOutputs});

        std::vector<Tensor> nonConstOutputs = testCase.outputTensors;
        testCase.opFunc(testCase.inputTensors, nonConstOutputs, testCase.args);

        auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
        DynFuncRunner::Run(funcop);

        ASSERT_EQ(testCase.goldenPaths.size(), testCase.outputTensors.size());
        for (size_t i = 0; i < testCase.outputTensors.size(); ++i) {
            auto& tensor = testCase.outputTensors[i];
            switch (tensor.GetDataType()) {
                case DataType::DT_FP32:
                    readGoldenCmp<float>(tensor, testCase.goldenPaths[i], i, 0.005f);
                    break;
                case DataType::DT_FP16:
                    readGoldenCmp<npu::tile_fwk::float16>(tensor, testCase.goldenPaths[i], i, 0.005f);
                    break;
                case DataType::DT_BF16:
                    readGoldenCmp<npu::tile_fwk::bfloat16>(tensor, testCase.goldenPaths[i], i, 0.005f);
                    break;
                case DataType::DT_INT8:
                    readGoldenCmp<int8_t>(tensor, testCase.goldenPaths[i], i, 0);
                    break;
                case DataType::DT_INT32:
                    readGoldenCmp<int32_t>(tensor, testCase.goldenPaths[i], i, 0);
                    break;
                default:
                    ASSERT_TRUE(false) << "no support dtype " << tensor.GetDataType();
                    break;
            }
        }
    }

    template<typename T>
    static void readGoldenCmp(const Tensor& tensor, const std::string& goldenPath, size_t index, T tolerance) {
        size_t elementCount = 1;
        for (int dim : tensor.GetShape()) {
            elementCount *= dim;
        }
        std::vector<T> goldenOutput(elementCount, 0);
        readInput<T>(goldenPath, goldenOutput);
        auto actualData = ProgramData::GetInstance().GetOutputData(index);
        const T* actual = (T*)actualData->data();
        int ret = resultCmp(goldenOutput, actual, tolerance);
        EXPECT_EQ(ret, true);
    }
};
} // namespace test_operation
} // namespace ascend
