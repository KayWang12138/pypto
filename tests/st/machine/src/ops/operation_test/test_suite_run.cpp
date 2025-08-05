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
 * \file test_suite_run.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <functional>
#include <map>
#include <stdexcept>
#include <iostream>

#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dynamic.h"
#include "interface/tensor/float.h"

namespace test_suite_run {
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr int SPLIT_1_DIM = 1;
constexpr int SPLIT_2_DIM = 2;
constexpr int SPLIT_3_DIM = 3;
constexpr int SPLIT_4_DIM = 4;
constexpr int SPLIT_5_DIM = 5;

constexpr int INPUT_TWO_NUM = 2;
struct TestCaseDesc {
    std::vector<Tensor> inputTensors;
    std::vector<Tensor> outputTensors;
    std::string opName;
    json params;
    std::vector<int> tileShape;
    std::vector<int> viewShape;
    std::string name;
    std::string opFunction;

    std::vector<std::string> inputPaths;
    std::vector<std::string> goldenPaths;
};

DataType DtypeConv(const std::string &str) {
    const std::map<std::string, DataType> stringToDataType = {
        {  "INT4",   DataType::DT_INT4},
        {  "INT8",   DataType::DT_INT8},
        { "INT16",  DataType::DT_INT16},
        { "INT32",  DataType::DT_INT32},
        { "INT64",  DataType::DT_INT64},
        {   "FP8",    DataType::DT_FP8},
        {  "FP16",   DataType::DT_FP16},
        {  "FP32",   DataType::DT_FP32},
        {  "BF16",   DataType::DT_BF16},
        { "UINT8",  DataType::DT_UINT8},
        {"UINT16", DataType::DT_UINT16},
        {"UINT32", DataType::DT_UINT32},
        {"UINT64", DataType::DT_UINT64},
    };
    auto it = stringToDataType.find(str);
    if (it != stringToDataType.end()) {
        return it->second;
    }
    throw std::invalid_argument("Invalid dtype string: " + str);
}

CastMode CastModeConv(const std::string &str) {
    const std::map<std::string, CastMode> stringToCastModeMap = {
        { "CAST_NONE",  CAST_NONE},
        { "CAST_RINT",  CAST_RINT},
        {"CAST_ROUND", CAST_ROUND},
        {"CAST_FLOOR", CAST_FLOOR},
        { "CAST_CEIL",  CAST_CEIL},
        {"CAST_TRUNC", CAST_TRUNC},
        {  "CAST_ODD",   CAST_ODD}
    };
    auto it = stringToCastModeMap.find(str);
    return it != stringToCastModeMap.end() ? it->second : CAST_NONE;
}

inline SymbolicScalar CeilDivSymbolicScalar(SymbolicScalar a, int b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

static void AddOperationExeFuncDoubleCut(TestCaseDesc &testCase) {
    FUNCTION("main", FunctionType::DYNAMIC, {testCase.inputTensors[0], testCase.inputTensors[1]},
        {testCase.outputTensors[0]}) {
        SymbolicScalar firstDim = testCase.inputTensors[0]->shape[0];
        SymbolicScalar secondDim = testCase.inputTensors[0]->shape[1];
        const int firstViewShape = testCase.viewShape[0];
        const int secondViewShape = testCase.viewShape[1];

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx,
            LoopRange(0, CeilDivSymbolicScalar(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx,
                LoopRange(0, CeilDivSymbolicScalar(secondDim, secondViewShape), 1)) {
                auto tileTensor0 = DViewPad(testCase.inputTensors[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                auto tileTensor1 = DViewPad(testCase.inputTensors[1], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                Program::GetInstance().GetTileShape().SetVecTileShapes(testCase.tileShape);
                auto res = Add(tileTensor0, tileTensor1);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, testCase.outputTensors[0]);
            }
        }
    }
}

static void AddOperationExeFunc(TestCaseDesc &testCase) {
    FUNCTION("main", FunctionType::DYNAMIC, {testCase.inputTensors[0], testCase.inputTensors[1]},
        {testCase.outputTensors[0]}) {
        SymbolicScalar firstDim = testCase.inputTensors[0]->shape[0];
        SymbolicScalar secondDim = testCase.inputTensors[0]->shape[1];
        const int firstl0LoopLengthTile = testCase.viewShape[0];

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx,
            LoopRange(0, CeilDivSymbolicScalar(firstDim, firstl0LoopLengthTile), 1)) {
            auto tileTensor0 = DViewPad(testCase.inputTensors[0], {firstl0LoopLengthTile, secondDim},
                {std::min(firstDim - bIdx * firstl0LoopLengthTile, firstl0LoopLengthTile), secondDim},
                {bIdx * firstl0LoopLengthTile, 0});
            auto tileTensor1 = DViewPad(testCase.inputTensors[1], {firstl0LoopLengthTile, secondDim},
                {std::min(firstDim - bIdx * firstl0LoopLengthTile, firstl0LoopLengthTile), secondDim},
                {bIdx * firstl0LoopLengthTile, 0});
            Program::GetInstance().GetTileShape().SetVecTileShapes(testCase.tileShape);
            auto res = Add(tileTensor0, tileTensor1);
            DAssemble(res, {bIdx * firstl0LoopLengthTile, 0}, testCase.outputTensors[0]);
        }
    }
}

std::map<std::string, std::function<void(TestCaseDesc &)>> OpFunctionMap = {
    {         "AddOperationExeFunc",          AddOperationExeFunc},
    {"AddOperationExeFuncDoubleCut", AddOperationExeFuncDoubleCut},
};

struct LoopInfo {
    std::vector<int> splitDim;
    std::vector<int> viewShape;
};

LoopInfo PreprocessViewShape(const std::vector<int> &viewShape) {
    LoopInfo info;
    for (size_t i = 0; i < viewShape.size(); ++i) {
        if (viewShape[i] != 0) {
            info.splitDim.push_back(i);
            info.viewShape.push_back(viewShape[i]);
        }
    }
    return info;
}

// todo 支持多种比对方式
template <typename T>
static void ReadGoldenCmp(const Tensor &tensor, const std::string &goldenPath, size_t index, T tolerance) {
    size_t elementCount = 1;
    for (int dim : tensor.GetShape()) {
        elementCount *= dim;
    }
    std::vector<T> goldenOutput(elementCount, 0);
    readInput<T>(goldenPath, goldenOutput);
    auto actualData = ProgramData::GetInstance().GetOutputData(index);
    const T *actual = (T *)actualData->data();
    int ret = resultCmp(goldenOutput, actual, tolerance);
    EXPECT_EQ(ret, true);
}

// 定义嵌套循环宏（最多支持5维）
#define LOOP_1D(_loopRange0, _body)                                       \
    LOOP("LOOP_L0_aIdx", FunctionType::DYNAMIC_LOOP, aIdx, _loopRange0) { \
        _body;                                                            \
    }

#define LOOP_2D(_loopRange0, _loopRange1, _body)                              \
    LOOP("LOOP_L0_aIdx", FunctionType::DYNAMIC_LOOP, aIdx, _loopRange0) {     \
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, _loopRange1) { \
            _body;                                                            \
        }                                                                     \
    }

#define LOOP_3D(_loopRange0, _loopRange1, _loopRange2, _body)                     \
    LOOP("LOOP_L0_aIdx", FunctionType::DYNAMIC_LOOP, aIdx, _loopRange0) {         \
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, _loopRange1) {     \
            LOOP("LOOP_L2_cIdx", FunctionType::DYNAMIC_LOOP, cIdx, _loopRange2) { \
                _body;                                                            \
            }                                                                     \
        }                                                                         \
    }

#define LOOP_4D(_loopRange0, _loopRange1, _loopRange2, _loopRange3, _body)            \
    LOOP("LOOP_L0_aIdx", FunctionType::DYNAMIC_LOOP, aIdx, _loopRange0) {             \
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, _loopRange1) {         \
            LOOP("LOOP_L2_cIdx", FunctionType::DYNAMIC_LOOP, cIdx, _loopRange2) {     \
                LOOP("LOOP_L3_dIdx", FunctionType::DYNAMIC_LOOP, dIdx, _loopRange3) { \
                    _body;                                                            \
                }                                                                     \
            }                                                                         \
        }                                                                             \
    }

#define LOOP_5D(_loopRange0, _loopRange1, _loopRange2, _loopRange3, _loopRange4, _body)   \
    LOOP("LOOP_L0_aIdx", FunctionType::DYNAMIC_LOOP, aIdx, _loopRange0) {                 \
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, _loopRange1) {             \
            LOOP("LOOP_L2_cIdx", FunctionType::DYNAMIC_LOOP, cIdx, _loopRange2) {         \
                LOOP("LOOP_L3_dIdx", FunctionType::DYNAMIC_LOOP, dIdx, _loopRange3) {     \
                    LOOP("LOOP_L4_eIdx", FunctionType::DYNAMIC_LOOP, eIdx, _loopRange4) { \
                        _body;                                                            \
                    }                                                                     \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
    }

class OpExecutor {
public:
    virtual Tensor OperationExecute(TestCaseDesc &testCase, std::vector<Tensor> &tileInputs) = 0;
    virtual ~OpExecutor() = default;
};

// 二元算子执行器
template <typename Func>
class BinaryOpExecutor : public OpExecutor {
public:
    explicit BinaryOpExecutor(Func func) : func_(func) {}
    Tensor OperationExecute(TestCaseDesc &testCase, std::vector<Tensor> &tileInputs) override {
        (void)testCase;
        if (tileInputs.size() != INPUT_TWO_NUM) {
            throw std::invalid_argument("Binary operator requires 2 inputs");
        }
        return func_(tileInputs[0], tileInputs[1]);
    }

private:
    Func func_;
};

// 二元算子scalar执行器
template <typename Func>
class BinaryOpScalarExecutor : public OpExecutor {
public:
    explicit BinaryOpScalarExecutor(Func func) : func_(func) {}
    Tensor OperationExecute(TestCaseDesc &testCase, std::vector<Tensor> &tileInputs) override {
        if (tileInputs.size() != 1) {
            throw std::invalid_argument("Binary operator requires 1 input");
        }
        DataType dtype = DtypeConv(testCase.params.value("dtype", "FP32"));
        float scalar = testCase.params.value("scalar", 0.0f);
        return func_(tileInputs[0], Element(dtype, scalar));
    }

private:
    Func func_;
};

// 一元算子执行器
template <typename Func>
class UnaryOpExecutor : public OpExecutor {
public:
    explicit UnaryOpExecutor(Func func) : func_(func) {}
    Tensor OperationExecute(TestCaseDesc &testCase, std::vector<Tensor> &tileInputs) override {
        (void)testCase;
        if (tileInputs.size() != 1) {
            throw std::invalid_argument("Unary operator requires 1 input");
        }
        return func_(tileInputs[0]);
    }

private:
    Func func_;
};

template <typename Func>
class ReduceOpExecutor : public OpExecutor {
public:
    explicit ReduceOpExecutor(Func func) : func_(func) {}
    Tensor OperationExecute(TestCaseDesc &testCase, std::vector<Tensor> &tileInputs) override {
        if (tileInputs.size() != 1) {
            throw std::invalid_argument("Reduce operator requires 1 input");
        }
        int axis = testCase.params.value("axis", 0);
        return func_(tileInputs[0], axis);
    }

private:
    Func func_;
};

// cast算子执行器
template <typename Func>
class CastOpExecutor : public OpExecutor {
public:
    explicit CastOpExecutor(Func func) : func_(func) {}
    Tensor OperationExecute(TestCaseDesc &testCase, std::vector<Tensor> &tileInputs) override {
        if (tileInputs.size() != 1) {
            throw std::invalid_argument("Unary operator requires 1 input");
        }
        DataType castDtype = DtypeConv(testCase.params.value("newDataType", "FP32"));
        CastMode mode = CastModeConv(testCase.params.value("mode", "CAST_RINT"));
        return func_(tileInputs[0], castDtype, mode);
    }

private:
    Func func_;
};

class OpRegistry {
public:
    static OpRegistry &getInstance() {
        static OpRegistry instance;
        return instance;
    }

    template <typename Executor, typename Func>
    void RegisterOp(std::string opName, Func func) {
        executors_[opName] = std::make_unique<Executor>(func);
    }

    void ExecuteOp(TestCaseDesc &testCase) {
        std::vector<std::reference_wrapper<const Tensor>> inputTensorRefList;
        for (size_t i = 0; i < testCase.inputTensors.size(); ++i) {
            inputTensorRefList.push_back(std::cref(testCase.inputTensors[i]));
        }

        std::vector<std::reference_wrapper<const Tensor>> outputTensorRefList;
        for (size_t i = 0; i < testCase.outputTensors.size(); ++i) {
            outputTensorRefList.push_back(std::cref(testCase.outputTensors[i]));
        }

        FUNCTION("main", FunctionType::DYNAMIC, inputTensorRefList, outputTensorRefList) {
            LoopInfo info = PreprocessViewShape(testCase.viewShape);
            auto toVector = [](auto... args) {
                return std::vector<SymbolicScalar>{static_cast<SymbolicScalar>(args)...};
            };

            switch (info.splitDim.size()) {
                case SPLIT_1_DIM: {
                    LOOP_1D(
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[0]], info.viewShape[0]),
                            1),
                        BuildLoopBody(testCase, info, toVector(aIdx)));
                    break;
                }

                case SPLIT_2_DIM: {
                    LOOP_2D(
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[0]], info.viewShape[0]),
                            1),
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[1]], info.viewShape[1]),
                            1),
                        BuildLoopBody(testCase, info, toVector(aIdx, bIdx)));
                    break;
                }

                case SPLIT_3_DIM: {
                    LOOP_3D(
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[0]], info.viewShape[0]),
                            1),
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[1]], info.viewShape[1]),
                            1),
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[2]], info.viewShape[2]),
                            1),
                        BuildLoopBody(testCase, info, toVector(aIdx, bIdx, cIdx)));
                    break;
                }

                case SPLIT_4_DIM: {
                    LOOP_4D(
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[0]], info.viewShape[0]),
                            1),
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[1]], info.viewShape[1]),
                            1),
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[2]], info.viewShape[2]),
                            1),
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[3]], info.viewShape[3]),
                            1),
                        BuildLoopBody(testCase, info, toVector(aIdx, bIdx, cIdx, dIdx)));
                    break;
                }

                case SPLIT_5_DIM: {
                    LOOP_5D(
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[0]], info.viewShape[0]),
                            1),
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[1]], info.viewShape[1]),
                            1),
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[2]], info.viewShape[2]),
                            1),
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[3]], info.viewShape[3]),
                            1),
                        LoopRange(0,
                            CeilDivSymbolicScalar(testCase.inputTensors[0]->shape[info.splitDim[4]], info.viewShape[4]),
                            1),
                        BuildLoopBody(testCase, info, toVector(aIdx, bIdx, cIdx, dIdx, eIdx)));
                    break;
                }

                default: throw std::invalid_argument("Unsupported number of split dim");
            }
        }
    }

    void BuildLoopBody(TestCaseDesc &testCase, LoopInfo &info, const std::vector<SymbolicScalar> &loopVarArray) {
        const int dimNum = testCase.inputTensors[0]->shape.size();
        std::vector<SymbolicScalar> newOffsets(dimNum, 0);  // 如果不切分，为0； 如果切分，是Idx*viewShape
        std::vector<SymbolicScalar> newValidShapes(dimNum); // 如果不切分，为原shape； 如果切分，是min表达式
        std::vector<int> dViewShape(testCase.viewShape);    // 如果不切分，为原shape； 如果切分，是viewShape

        // 设置不需要切分的维度
        for (auto i = 0; i < dimNum; ++i) {
            if (testCase.viewShape[i] == 0) {
                newValidShapes[i] = testCase.inputTensors[0]->shape[i];
                dViewShape[i] = testCase.inputTensors[0]->shape[i];
            }
        }

        // 设置需要切分的维度
        for (auto i = 0; i < static_cast<int>(info.splitDim.size()); ++i) {
            const size_t dimIdx = info.splitDim[i];
            const int viewSize = info.viewShape[i];

            newOffsets[dimIdx] = loopVarArray[i] * viewSize;
            newValidShapes[dimIdx] = std::min(testCase.inputTensors[0]->shape[dimIdx] - newOffsets[dimIdx], (viewSize));
        }

        std::vector<Tensor> tileInputs;
        for (const auto &inputTensor : testCase.inputTensors) {
            std::vector<SymbolicScalar> tmpNewOffsets(newOffsets);
            std::vector<SymbolicScalar> tmpNewValidShapes(newValidShapes);
            std::vector<int> tmpDViewShape(dViewShape);
            for (auto i = 0; i < dimNum; ++i) {
                if (inputTensor->shape[i] == 1) {
                    tmpNewOffsets[i] = SymbolicScalar(0);
                    tmpNewValidShapes[i] = SymbolicScalar(1);
                    tmpDViewShape[i] = 1;
                }
            }
            auto tileTensor = DViewPad(inputTensor, tmpDViewShape, tmpNewValidShapes, tmpNewOffsets);
            tileInputs.push_back(tileTensor);
        }

        Program::GetInstance().GetTileShape().SetVecTileShapes(testCase.tileShape);

        auto it = executors_.find(testCase.opName);
        if (it != executors_.end()) {
            auto res = it->second->OperationExecute(testCase, tileInputs);
            DAssemble(res, newOffsets, testCase.outputTensors[0]);
        } else {
            throw std::runtime_error("OpExecutor not found for opName: " + testCase.opName);
        }
    }

private:
    OpRegistry() = default;
    std::map<std::string, std::unique_ptr<OpExecutor>> executors_;
};

class TestExecutor {
public:
    explicit TestExecutor(TestCaseDesc &tc) : testCase(tc) {}
    void Execute() {
        std::cout << "Executing operation: " << testCase.opName << " for test case: " << testCase.name << std::endl;
        Init();
        ProcessIOData();
        if (testCase.opFunction == "none") {
            OpRegistry::getInstance().ExecuteOp(testCase);
        } else {
            std::cout << "  Execute custom opFunction : " << testCase.opFunction << std::endl;
            auto it = OpFunctionMap.find(testCase.opFunction);
            if (it != OpFunctionMap.end()) {
                it->second(testCase);
            } else {
                std::cerr << " custom opFunction not found : " << testCase.opFunction << std::endl;
            }
        }

        auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
        DynFuncRunner::Run(funcop);
        CmpResults();

        std::cout << "Operation completed for test case: " << testCase.name << std::endl;
    }

    static void Init() {
        config::SetHostConfig(npu::tile_fwk::KEY_ONLY_CODEGEN, true);
        config::SetCodeGenConfig(npu::tile_fwk::KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    }

    void ProcessIOData() {
        // 设置输入数据
        std::vector<RawTensorDataPtr> inputs;
        ASSERT_EQ(testCase.inputTensors.size(), testCase.inputPaths.size());
        for (size_t i = 0; i < testCase.inputTensors.size(); ++i) {
            size_t elementCount = 1;
            for (int dim : testCase.inputTensors[i].GetShape()) {
                elementCount *= dim;
            }
            std::vector<uint8_t> input(elementCount * BytesOf(testCase.inputTensors[i].GetDataType()), 0);
            readInput<uint8_t>(testCase.inputPaths[i], input);
            inputs.push_back(RawTensorData::CreateTensor(testCase.inputTensors[i], input));
        }
        ProgramData::GetInstance().AppendInputs({inputs});

        // 设置输出Tensor
        std::vector<RawTensorDataPtr> outputs;
        for (auto &tensor : testCase.outputTensors) {
            outputs.push_back(RawTensorData::CreateTensorZero(tensor));
        }
        ProgramData::GetInstance().AppendOutputs({outputs});
    }

    void CmpResults() {
        ASSERT_EQ(testCase.goldenPaths.size(), testCase.outputTensors.size());
        for (size_t i = 0; i < testCase.outputTensors.size(); ++i) {
            auto &tensor = testCase.outputTensors[i];
            switch (tensor.GetDataType()) {
                case DataType::DT_FP32: ReadGoldenCmp<float>(tensor, testCase.goldenPaths[i], i, 0.005f); break;
                case DataType::DT_FP16:
                    ReadGoldenCmp<npu::tile_fwk::float16>(tensor, testCase.goldenPaths[i], i, 0.005f);
                    break;
                case DataType::DT_BF16:
                    ReadGoldenCmp<npu::tile_fwk::bfloat16>(tensor, testCase.goldenPaths[i], i, 0.005f);
                    break;
                case DataType::DT_INT8: ReadGoldenCmp<int8_t>(tensor, testCase.goldenPaths[i], i, 0); break;
                case DataType::DT_INT16: ReadGoldenCmp<int16_t>(tensor, testCase.goldenPaths[i], i, 0); break;
                case DataType::DT_INT32: ReadGoldenCmp<int32_t>(tensor, testCase.goldenPaths[i], i, 0); break;
                default: ASSERT_TRUE(false) << "no support dtype " << tensor.GetDataType(); break;
            }
        }
    }

private:
    TestCaseDesc &testCase;
};

void RegisterOperations() {
    auto &registry = OpRegistry::getInstance();
    registry.RegisterOp<BinaryOpExecutor<Tensor (*)(const Tensor &, const Tensor &)>>(std::string("add"), Add);
    registry.RegisterOp<BinaryOpExecutor<Tensor (*)(const Tensor &, const Tensor &)>>(std::string("sub"), Sub);
    registry.RegisterOp<BinaryOpExecutor<Tensor (*)(const Tensor &, const Tensor &)>>(std::string("mul"), Mul);
    registry.RegisterOp<BinaryOpExecutor<Tensor (*)(const Tensor &, const Tensor &)>>(std::string("div"), Div);

    registry.RegisterOp<UnaryOpExecutor<Tensor (*)(const Tensor &)>>(std::string("exp"), Exp);
    registry.RegisterOp<UnaryOpExecutor<Tensor (*)(const Tensor &)>>(std::string("abs"), Abs);
    registry.RegisterOp<UnaryOpExecutor<Tensor (*)(const Tensor &)>>(std::string("sqrt"), Sqrt);

    registry.RegisterOp<BinaryOpScalarExecutor<Tensor (*)(const Tensor &, const Element &)>>(std::string("adds"), AddS);
    registry.RegisterOp<BinaryOpScalarExecutor<Tensor (*)(const Tensor &, const Element &)>>(std::string("subs"), SubS);
    registry.RegisterOp<BinaryOpScalarExecutor<Tensor (*)(const Tensor &, const Element &)>>(std::string("muls"), MulS);
    registry.RegisterOp<BinaryOpScalarExecutor<Tensor (*)(const Tensor &, const Element &)>>(std::string("divs"), DivS);

    registry.RegisterOp<CastOpExecutor<Tensor (*)(const Tensor &, DataType, CastMode)>>(std::string("cast"), Cast);

    registry.RegisterOp<ReduceOpExecutor<Tensor (*)(const Tensor &, int)>>(std::string("reduce_max"), RowMaxSingle);
    registry.RegisterOp<ReduceOpExecutor<Tensor (*)(const Tensor &, int)>>(std::string("reduce_min"), RowMinSingle);
    registry.RegisterOp<ReduceOpExecutor<Tensor (*)(const Tensor &, int)>>(std::string("reduce_sum"), RowSumSingle);
}

class TestCasesManager {
public:
    static void AddTestCaseFile(const std::string &casePath) { caseFiles.push_back(casePath); }
    static void AddTestCaseFilesFromDir(const std::string &dirPath) {
        try {
            for (const auto &entry : fs::directory_iterator(dirPath)) {
                if (fs::is_regular_file(entry) && entry.path().extension() == ".json") {
                    AddTestCaseFile(entry.path().string());
                    std::cout << " Added test case file : " << entry.path().string() << std::endl;
                }
            }
        } catch (const fs::filesystem_error &ex) {
            std::cerr << "Error reading directory: " << ex.what() << std::endl;
        }
    }

    static std::vector<std::pair<std::string, std::string>> LoadAllTestCases() {
        std::vector<std::pair<std::string, std::string>> testCases;
        for (const auto &file : caseFiles) {
            try {
                std::ifstream inFile(file);
                if (!inFile.is_open()) {
                    throw std::runtime_error("Failed to open file: " + file);
                }

                json config = json::parse(inFile);
                for (const auto &testCase : config["test_cases"]) {
                    testCases.emplace_back(file, testCase["name"].get<std::string>());
                }
            } catch (const std::exception &ex) {
                std::cerr << "Error parsing JSON file [" << file << "] " << ex.what() << std::endl;
            }
        }

        return testCases;
    }

private:
    static std::vector<std::string> caseFiles;
};

std::vector<std::string> TestCasesManager::caseFiles;

struct TestInitializer {
    TestInitializer() {
        RegisterOperations();
        TestCasesManager::AddTestCaseFilesFromDir("../../../tests/st/machine/src/ops/operation_test/test_case");
    }
};

static TestInitializer initializer;

class OperationTest
    : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<std::pair<std::string, std::string>> {
protected:
    std::string GetCasePath() const { 
        return GetParam().first; 
    }
    std::string GetTestCaseName() const { 
        return GetParam().second; 
    }
};

INSTANTIATE_TEST_SUITE_P(AllOperations, OperationTest, ::testing::ValuesIn(TestCasesManager::LoadAllTestCases()));

TEST_P(OperationTest, RunTestCase) {
    std::string casePath = GetCasePath();
    std::string caseName = GetTestCaseName();

    std::string cmd =
        "python3 ../../../tests/st/machine/src/ops/operation_test/python_tools/generate_golden.py  --path " + casePath +
        "  --name " + caseName;
    ASSERT_EQ(std::system(cmd.c_str()), 0) << "Python script failed";

    std::filesystem::path p(casePath);
    std::string resultPath = "./" + p.stem().string() + "_result.json";

    std::ifstream resultFile(resultPath);
    ASSERT_TRUE(resultFile.is_open()) << "Failed to open result json file";
    json resultJson = json::parse(resultFile);

    json caseJson;
    for (const auto &item : resultJson["test_cases"]) {
        if (item["name"] == caseName) {
            caseJson = item;
            break;
        }
    }
    ASSERT_FALSE(caseJson.is_null()) << "Test case [" << caseName << "] not found in result json file";

    TestCaseDesc testCase;
    testCase.name = caseName;
    testCase.opName = caseJson["operation"].get<std::string>();
    testCase.tileShape = caseJson["tile_shape"].get<std::vector<int>>();
    testCase.viewShape = caseJson["view_shape"].get<std::vector<int>>();
    testCase.params = caseJson["params"];

    // 可选处理
    testCase.opFunction = caseJson.value("op_function", "none");

    for (const auto &tensorJson : caseJson["input_tensors"]) {
        DataType dtype = DtypeConv(tensorJson["dtype"].get<std::string>());
        testCase.inputTensors.emplace_back(
            dtype, tensorJson["shape"].get<std::vector<int>>(), tensorJson["name"].get<std::string>());
        testCase.inputPaths.emplace_back(tensorJson["path"].get<std::string>());
    }

    for (const auto &tensorJson : caseJson["output_tensors"]) {
        DataType dtype = DtypeConv(tensorJson["dtype"].get<std::string>());
        testCase.outputTensors.emplace_back(
            dtype, tensorJson["shape"].get<std::vector<int>>(), tensorJson["name"].get<std::string>());
        testCase.goldenPaths.emplace_back(tensorJson["path"].get<std::string>());
    }

    // 执行测试
    TestExecutor executor(testCase);
    executor.Execute();
}
}
