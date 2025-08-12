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
 * \file test_common.h
 * \brief
 */

#pragma once

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "runtime.h"
#include "operation/tilefwk_op.h"
#include "interface/tensor/float.h"
#include "interface/tensor/logical_tensor.h"
#include "cost_model/simulation/pv/PvData.h"
#include "cost_model/simulation/emulator/SoftMemory.h"

using namespace npu::tile_fwk;
using Json = nlohmann::json;
using namespace std;

const std::string TEST_TILE_OP_PATH = "tests/st/interface/tile_op/src/";

enum CpyMode { FULL, DIAG };

template <typename T>
DataType GetAstDtype() {
    DataType astDtype = DataType::DT_BOTTOM;
    if constexpr (std::is_same<T, npu::tile_fwk::float16>::value) {
        astDtype = DataType::DT_FP16;
    }
    if constexpr (std::is_same<T, float>::value) {
        astDtype = DataType::DT_FP32;
    }
    if constexpr (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        astDtype = DataType::DT_BF16;
    }
    if constexpr (std::is_same<T, int8_t>::value) {
        astDtype = DT_INT8;
    }
    if constexpr (std::is_same<T, int32_t>::value) {
        astDtype = DT_INT32;
    }
    EXPECT_NE(astDtype, DT_BOTTOM);
    return astDtype;
}

template <typename T = float>
static void readInput(std::string filename, vector<T> &inputData) {
    ifstream input_file(filename, ios::binary);
    if (!input_file) {
        std::cerr << "Failed to open file for writing input data! filename:" << filename;
        ASSERT(false);
    }
    input_file.read((char *)inputData.data(), inputData.size() * sizeof(T));
    input_file.close();
}

template <typename T>
static void writeInput(std::string filename, vector<T> outData) {
    std::ofstream ascendOutFile(filename, std::ios::out | std::ios::binary);
    if (!ascendOutFile) {
        std::cerr << "Can not open out file!" << std::endl;
    }
    ascendOutFile.write((char *)outData.data(), outData.size() * sizeof(T));
    ascendOutFile.close();
}

template <typename T = float>
static void writeInput(std::string filename, LogicalTensor outData) {
    std::ofstream ascendOutFile(filename, std::ios::out | std::ios::binary);
    if (!ascendOutFile) {
        std::cerr << "Can not open out file!" << std::endl;
    }

    if (std::is_integral_v<T>) {
        vector<int> out = npu::tile_fwk::ConvElementVecToIntVec(outData.tensor->GetData());
        ascendOutFile.write((char *)out.data(), out.size() * sizeof(int));
    } else if (std::is_floating_point_v<T>) {
        vector<float> out = npu::tile_fwk::ConvElementVecToFloatVec(outData.tensor->GetData());
        ascendOutFile.write((char *)out.data(), out.size() * sizeof(float));
    } else {
        std::cerr << "unknown type !!!" << std::endl;
    }
    ascendOutFile.close();
}

[[maybe_unused]] static void copyOutDataForGolden(
    vector<float> &outData, vector<float> &outDataVal, std::vector<int> &shape, CpyMode mode) {
    vector<float>::iterator itr = outData.begin();

    if (mode == DIAG) {
        for (int row = 0; row < shape[0]; row++) {
            if (row == 16) {
                if (shape[0] - 16 < 0) {
                    break;
                }
                row = ((shape[0] - 16) <= row) ? row : (shape[0] - 16);
            }
            for (int col = 0; col < shape[1]; col++) {
                if (col == 16) {
                    if (shape[1] - 16 < 0) {
                        break;
                    }
                    col = ((shape[1] - 16) <= col) ? col : (shape[1] - 16);
                }

                vector<float>::iterator itrTmp = itr + row * shape[1] + col;
                if (itrTmp == outData.end()) {
                    break;
                }
                outDataVal.push_back(*itrTmp);
            }
        }
    } else {
        outDataVal = outData;
    }
}

[[maybe_unused]] static void copyOutDataForCpu(LogicalTensor outData, vector<float> &outDataVal, CpyMode mode) {
    auto &data = outData.tensor->GetData();
    auto itr = data.begin();

    if (mode == DIAG) {
        for (int row = 0; row < outData.shape[0]; row++) {
            if (row == 16) {
                if (outData.shape[0] - 16 < 0) {
                    break;
                }
                row = ((outData.shape[0] - 16) <= row) ? row : (outData.shape[0] - 16);
            }
            for (int col = 0; col < outData.shape[1]; col++) {
                if (col == 16) {
                    if (outData.shape[1] - 16 < 0) {
                        break;
                    }
                    col = ((outData.shape[1] - 16) <= col) ? col : (outData.shape[1] - 16);
                }

                auto itrTmp = itr + row * outData.shape[1] + col;
                if (itrTmp == data.end()) {
                    break;
                }
                float val = static_cast<float>(itrTmp->GetFloatData());
                outDataVal.push_back(val);
            }
        }
    } else {
        while (itr != data.end()) {
            float val = static_cast<float>(itr->GetFloatData());
            outDataVal.push_back(val);
            itr++;
        }
    }
}

template <typename T = float>
static bool resultCmpUnary(const vector<T> &x, const vector<T> &outDataValExp, const vector<T> &outDataValAct,
    float eps, size_t threshold = 1, bool printAll = false, bool printErr = false) {
    if (outDataValExp.size() != outDataValAct.size()) {
        std::cout << "out size is not eq, golden: " << outDataValExp.size() << ", act: " << outDataValAct.size()
                  << std::endl;
        return false;
    }
    float maxDiff = 0;
    float maxDiffRatio = 0;
    size_t errCount = 0;

    bool rst = true;
    size_t eSize = outDataValExp.size();
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto inVal = static_cast<float>(x[eIdx]);
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = (std::abs(expVal) < 0.001 && std::abs(actVal) < 0.001) ? diff : std::abs(diff / expVal);
        maxDiff = std::max(diff, maxDiff);
        maxDiffRatio = std::max(relRatio, maxDiffRatio);

        auto eErr = (diff > eps && relRatio > eps);
        errCount += eErr ? 1 : 0;

        if ((printAll) || (eErr && printErr)) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", input->" << inVal << ", exp->" << expVal
                      << ", act->" << actVal << ", diff->" << diff << ", diff ratio->" << relRatio << std::endl;
        }
        rst = errCount <= threshold;
    }
    float errCountRatio = static_cast<float>(errCount) / static_cast<float>(eSize);
    std::cout << "max diff: " << maxDiff << ", max diff ratio: " << maxDiffRatio << ", err count: " << errCount
              << ", err threshold: " << threshold << ", err count ratio: " << errCountRatio << std::endl;
    if (rst || printAll || printErr) {
        return rst;
    }
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto inVal = static_cast<float>(x[eIdx]);
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = (std::abs(expVal) < 0.001 && std::abs(actVal) < 0.001) ? diff : std::abs(diff / expVal);

        auto eErr = (diff > eps && relRatio > eps);
        if (eErr) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", input->" << inVal << ", exp->" << expVal
                      << ", act->" << actVal << ", diff->" << diff << ", diff ratio->" << relRatio << std::endl;
        }
        rst = errCount <= threshold;
        if (!rst) {
            break;
        }
    }
    return false;
}

template <typename Ts, typename Td>
static bool resultCmpCast(const vector<Ts> &x, const vector<Td> &outDataValExp, const vector<Td> &outDataValAct,
    float eps, size_t threshold = 1, bool printAll = false, bool printErr = false) {
    if (outDataValExp.size() != outDataValAct.size()) {
        std::cout << "out size is not eq, golden: " << outDataValExp.size() << ", act: " << outDataValAct.size()
                  << std::endl;
        return false;
    }

    float maxDiff = 0;
    float maxDiffRatio = 0;
    size_t errCount = 0;

    bool rst = true;
    size_t eSize = outDataValExp.size();
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto inVal = static_cast<float>(x[eIdx]);
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = expVal - actVal;
        auto relRatio = std::abs(diff / expVal);
        maxDiff = std::max(diff, maxDiff);
        maxDiffRatio = std::max(relRatio, maxDiffRatio);

        auto eErr = (diff > eps && relRatio > eps);
        errCount += eErr ? 1 : 0;

        if ((printAll) || (eErr && printErr)) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", input->" << inVal << ", exp->" << expVal
                      << ", act->" << actVal << ", diff->" << diff << ", diff ratio->" << relRatio << std::endl;
        }
        rst = errCount <= threshold;
    }
    float errCountRatio = static_cast<float>(errCount) / static_cast<float>(eSize);
    std::cout << "max diff: " << maxDiff << ", max diff ratio: " << maxDiffRatio << ", err count: " << errCount
              << ", err threshold: " << threshold << ", err count ratio: " << errCountRatio << std::endl;
    if (rst || printAll || printErr) {
        return rst;
    }
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);

        auto eErr = (diff > eps && relRatio > eps);
        if (eErr) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal
                      << ", diff->" << diff << ", diff ratio->" << relRatio << std::endl;
        }
        rst = errCount <= threshold;
        if (!rst) {
            break;
        }
    }
    return false;
}

template <typename T = float>
static bool resultCmp(const vector<T> &outDataValExp, const T *outDataValAct, float eps, size_t threshold = 0,
    size_t zeroCountThreshold = 1000, bool printAll = false, bool printErr = false, size_t testNum = 0) {
    //
    threshold = threshold == 0 ? static_cast<int>(outDataValExp.size() * eps) : threshold;

    float maxDiff = 0;
    float maxDiffRatio = 0;
    size_t zeroCount = 0;
    size_t errCount = 0;

    bool rst = true;
    size_t eSize = outDataValExp.size();
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);
        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);
        maxDiff = std::max(diff, maxDiff);
        maxDiffRatio = std::max(relRatio, maxDiffRatio);
        zeroCount += std::abs(actVal - 0.0f) <= 1e-6 and std::abs(expVal - 0.0f) > 1e-6 ? 1 : 0;
        testNum = testNum - (testNum > 0 ? 1 : 0);

        auto eErr = ((diff > eps && relRatio > eps) || (zeroCount > zeroCountThreshold));
        errCount += eErr ? 1 : 0;

        if (std::isnan(expVal) || std::isnan(actVal)) {
            std::cout << "idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal << std::endl;
        }

        if ((printAll) || (eErr && printErr) || (testNum > 0)) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal
                      << ", diff->" << diff << ", diff ratio->" << relRatio << ", zero count->" << zeroCount
                      << ", zero threshold->" << zeroCountThreshold << std::endl;
        }
        rst = !((errCount > threshold || zeroCount > zeroCountThreshold));
    }

    float errCountRatio = static_cast<float>(errCount) / static_cast<float>(eSize);
    float zeroCountRatio = static_cast<float>(zeroCount) / static_cast<float>(eSize);
    std::cout << "max diff: " << maxDiff << ", max diff ratio: " << maxDiffRatio << ", err count: " << errCount
              << ", err threshold: " << threshold << ", err count ratio: " << errCountRatio
              << ", act zero count: " << zeroCount << ", act zero threshold: " << zeroCountThreshold
              << ", act zero ratio: " << zeroCountRatio << std::endl;
    if (rst || printAll || printErr) {
        return rst;
    }

    errCount = 0;
    zeroCount = 0;
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);
        zeroCount += std::abs(actVal - 0.0f) <= 1e-6 and std::abs(expVal - 0.0f) > 1e-6 ? 1 : 0;

        auto eErr = ((diff > eps && relRatio > eps) || (zeroCount > zeroCountThreshold));
        errCount += eErr ? 1 : 0;

        if (std::isnan(expVal) || std::isnan(actVal)) {
            std::cout << "idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal << std::endl;
        }

        if (eErr) {
            std::cout << "diff threshold: " << eps << ", idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal
                      << ", diff->" << diff << ", diff ratio->" << relRatio << ", zero count->" << zeroCount
                      << ", zero threshold->" << zeroCountThreshold << std::endl;
        }
        rst = !((errCount > threshold || zeroCount > zeroCountThreshold));
        if (!rst) {
            break;
        }
    }
    return false;
}

template <typename T = float>
static bool resultCmp(const vector<T> &outDataValExp, const vector<T> &outDataValAct, float eps, size_t threshold = 0,
    size_t zeroCountThreshold = 1000, bool printAll = false, bool printErr = false, size_t testNum = 0) {
    if (outDataValExp.size() != outDataValAct.size()) {
        std::cout << "out size is not eq, golden: " << outDataValExp.size() << ", act: " << outDataValAct.size()
                  << std::endl;
        return false;
    }
    return resultCmp(
        outDataValExp, outDataValAct.data(), eps, threshold, zeroCountThreshold, printAll, printErr, testNum);
}

template <class T = float>
void *readToDev(const std::string &path, int size) {
    size_t bytes = size * sizeof(T);
    std::vector<uint8_t> data(bytes);
    readInput(path, data);

    uint8_t *devPtr = nullptr;
    machine::GetRA()->AllocDevAddr(&devPtr, bytes);
    if (devPtr == nullptr) {
        std::cout << "rtMalloc failed" << std::endl;
        devPtr = reinterpret_cast<uint8_t *>(CostModel::SoftMemory::Instance().AllocateData(bytes, data));
        if (devPtr == nullptr) {
            std::cout << "SoftMemory rtMalloc failed" << std::endl;
            return nullptr;
        } else {
            CostModel::PvData::Instance().Put(devPtr, data);
            return devPtr;
        }
    }
    if (rtMemcpy(devPtr, bytes, data.data(), bytes, RT_MEMCPY_HOST_TO_DEVICE) != 0) {
        std::cout << "rtMalloc failed" << std::endl;
        return nullptr;
    }
    CostModel::PvData::Instance().Put(devPtr, data);
    return devPtr;
}

[[maybe_unused]] static uint8_t *allocDevAddr(uint64_t size) {
    uint8_t *devPtr = nullptr;
    machine::GetRA()->AllocDevAddr(&devPtr, size);
    if (devPtr == nullptr) {
        std::cout << "allocDevAddr rtMalloc failed" << std::endl;
        std::vector<uint8_t> data(size);
        devPtr = reinterpret_cast<uint8_t *>(CostModel::SoftMemory::Instance().AllocateData(size, data));
        if (devPtr == nullptr) {
            std::cout << "SoftMemory rtMalloc failed" << std::endl;
            return nullptr;
        } else {
            CostModel::PvData::Instance().Put(devPtr, data);
            return devPtr;
        }
        return nullptr;
    }

    return devPtr;
}

struct GMTensorInfoTest {
    float *Addr{nullptr};
    int64_t offset0{-1}; // TBD: should divide by TILESIZE or not? E.g . 128 or 128/128
    int64_t offset1{-1}; // TBD: should divide by TILESIZE or not? E.g . 128 or 128/128
};

struct InvokeEntryTest {
    int64_t SubGraphProgramId{-1};
    GMTensorInfoTest GMTensor[2];
};

using IfaTestParam = std::unordered_map<std::string, int>;

/* low latency params config */
inline IfaTestParam lowLatencyParams = {
    {            "b",   4},
    {           "nq",  32},
    {           "s2", 256},
    {"timethreshold",  55},
};

inline IfaTileShapeConfig lowLatencyTileParams{
    256, // block size
    32, // nTile
    {256, 128}, // v0 tile for qkv-view-concat, q-S1D:(32,64), k/v-S2D:(256,64), merge 2D to copy
    {32, 32, 256, 256, 128, 128}, // c1 tile for S1D@S2D
    {32, 256}, // v1 tile for S1S2
    {32, 32, 256, 256, 128, 128}, // c2 tile for S1S2@S2D
    {32, 256}, // v2 tile for S1D
};

/* hight throughput params param config */
inline IfaTestParam hightThroughputParams = {
    {            "b",   32},
    {           "nq",  128},
    {           "s2", 4096},
    {"timethreshold",  280},
};

inline IfaTileShapeConfig hightThroughputTileParams{
    512, // block size
    128, // nTile
    {256, 128}, // v0 tile for qkv-view-concat, q-S1D:(32,64), k/v-S2D:(256,64), merge 2D to copy
    {128, 128, 128, 256, 128, 128}, // c1 tile for S1D@S2D
    {32, 256}, // v1 tile for S1S2
    {128, 128, 128, 256, 128, 128}, // c2 tile for S1S2@S2D
    {32, 256}, // v2 tile for S1D
};

struct GraphInvokeInfoTest {
    int64_t GraphInvokeCount{0};
    InvokeEntryTest GraphInvokeList[40];
};

[[maybe_unused]] static std::string ParamLocToString(const int64_t &ParamLoc) {
    return std::to_string(ParamLoc >> 28) + ":" + std::to_string((ParamLoc >> 16) & 0xFFF) + ":" +
           std::to_string(ParamLoc & 0xFFFF);
}

[[maybe_unused]] static std::string GetGoldenDir() {
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    std::string fullName = std::string(testInfo->test_suite_name()) + "." + testInfo->name();
    // 先读取 AST_STEST_GOLDEN_PATH 环境变量, 否则使用当前目录
    char *path = getenv("AST_STEST_GOLDEN_PATH");
    std::string fullPath;
    if (path == nullptr) {
        fullPath = "./golden";
    } else {
        fullPath = std::string(path);
    }
    fullPath = fullPath + "/" + fullName;
    return fullPath;
}

static DataType GetDataType(const std::string &name) {
    static const std::map<std::string, DataType> name_to_dtype = {
        {  "int4",   DataType::DT_INT4},
        {  "int8",   DataType::DT_INT8},
        { "int16",  DataType::DT_INT16},
        { "int32",  DataType::DT_INT32},
        { "int64",  DataType::DT_INT64},
        {   "fp8",    DataType::DT_FP8},
        {  "fp16",   DataType::DT_FP16},
        {  "fp32",   DataType::DT_FP32},
        {  "bf16",   DataType::DT_BF16},
        {   "hf4",    DataType::DT_HF4},
        {   "hf8",    DataType::DT_HF8},
        { "uint8",  DataType::DT_UINT8},
        {"uint16", DataType::DT_UINT16},
        {"uint32", DataType::DT_UINT32},
        {"uint64", DataType::DT_UINT64},
        {  "bool",   DataType::DT_BOOL},
        {"double", DataType::DT_DOUBLE},
    };
    if (name_to_dtype.find(name) == name_to_dtype.end()) {
        ALOG_ERROR << "Not support type " << name << " yet, return fp32 as default.";
        return DataType::DT_FP32;
    }
    return name_to_dtype.at(name);
}

static std::vector<Tensor> GetTensors(const std::string &config, bool is_input = true) {
    std::ifstream json_file(config);
    ASSERT(json_file.is_open()) << "Fail to open " << config << ".";
    nlohmann::json json_data = nlohmann::json::parse(json_file);
    std::cout << "Create Tensors For " << json_data << std::endl;
    std::vector<Tensor> tensors;
    auto key = is_input ? "input_tensors" : "output_tensors";
    for (const auto &tensor_config : json_data.at(key)) {
        auto shape = tensor_config.at("shape").get<std::vector<int>>();
        auto dtype = GetDataType(tensor_config.at("dtype").get<std::string>());
        auto name = tensor_config.at("name").get<std::string>();
        tensors.push_back(Tensor(dtype, shape, name));
    }
    return tensors;
}

[[maybe_unused]] static std::vector<Tensor> GetInputTensors(const std::string &config) {
    return GetTensors(config, true);
}

[[maybe_unused]] static std::vector<Tensor> GetOutputTensors(const std::string &config) {
    return GetTensors(config, false);
}

template <typename T>
T GetValueByName(const std::string &config, const std::string &name) {
    std::ifstream json_file(config);
    ASSERT(json_file.is_open()) << "Fail to open " << config << ".";
    nlohmann::json json_data = nlohmann::json::parse(json_file);
    if (json_data.find(name) == json_data.end()) {
        json_data = json_data.at("params");
    }
    ASSERT(json_data.find(name) != json_data.end()) << "failed to load " << name << " in " << config << "!";
    return json_data.at(name).get<T>();
}

[[maybe_unused]] static std::vector<int> GetViewShape(const std::string &config) {
    return GetValueByName<std::vector<int>>(config, "view_shape");
}

[[maybe_unused]] static std::vector<int> GetTileShape(const std::string &config) {
    return GetValueByName<std::vector<int>>(config, "tile_shape");
}

[[maybe_unused]] static int GetFuncId(const std::string &config) {
    return GetValueByName<int>(config, "func_id");
}

inline int calcOffset(std::vector<int> shape, std::vector<int> offset) {
    int base = 1;
    int res = 0;
    for (int i = shape.size() - 1; i >= 0; i--) {
        res += offset[i] * base;
        base *= shape[i];
    }
    return res;
}

static std::string GetCurRunningPath() {
    constexpr int size = 1024;
    char buf[size] = {};
    std::string cwd = getcwd(buf, size);
    ASSERT(!cwd.empty()) << "failed to call getcwd()!";
    return cwd;
}

[[maybe_unused]] static int CompileCCEForSingleOpTest(
    const std::string &srcFile, const std::string &objFile, bool isCube) {
    std::string curPath = GetCurRunningPath();
    std::string codeSrcPath = curPath.append("/../../../");
    ALOG_INFO_F("codeSrcPath: %s", codeSrcPath.c_str());

    std::string coreType = isCube ? "dav-c220-cube" : "dav-c220-vec";
    const std::string envPath = std::string(std::getenv("ASCEND_AICPU_PATH"));
    std::string runtimePath = envPath + "/machine/include";
    std::string lib64Path = envPath + "/lib64";

    char ccecCmd[2048];
    std::string compileOptions = "";

    int ret = snprintf_s(ccecCmd, sizeof(ccecCmd), sizeof(ccecCmd) - 1,
        "ccec %s -lstdc++ -O2 -g -x cce -std=c++17 --shared -fPIC "
        "--cce-aicore-arch=%s "
        "--cce-enable-print "
        "-mllvm -cce-aicore-stack-size=0x8000 "
        "-mllvm -cce-aicore-function-stack-size=0x8000 "
        "-mllvm -cce-aicore-record-overflow=false "
        "-mllvm -cce-aicore-addr-transform "
        "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
        "-L%s "
        "-lruntime "
        "-I%s "
        "-I%s/include/tileop/a2a3 "
        "-I%s/src/machine/kernel/ "
        "-I%s/src/ "
        "-I%s/src/interface "
        "-o %s "
        "%s",
        compileOptions.c_str(), coreType.c_str(), lib64Path.c_str(), runtimePath.c_str(), codeSrcPath.c_str(),
        codeSrcPath.c_str(), codeSrcPath.c_str(), objFile.c_str(), srcFile.c_str());
    if (ret < 0) {
        ALOG_INFO << "CompileCCE snprintf_s failed " << ret;
    }

    ALOG_INFO << "compile kernel...\n" << ccecCmd;
    ret = std::system(ccecCmd);
    if (ret != 0) {
        ALOG_INFO << "CompileCce ccec failed " << ret;
    }
    return ret;
}
