/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_verifier.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

FlowVerifier::CompareResult FlowVerifier::VerifyResult(
        const std::shared_ptr<LogicalTensorData> &goldenDataView,
        const std::shared_ptr<LogicalTensorData> &outputDataView, float eps) {
    // tensor maybe padded during PadLocalBuffer Pass, tensor shape maybe changed, just check the valid data
    ASSERT(goldenDataView->GetValidShape() == outputDataView->GetValidShape());

    switch (goldenDataView->GetDataType()) {
        case DT_INT8: return CompareData<int8_t>(goldenDataView, outputDataView, eps);
        case DT_INT16: return CompareData<int16_t>(goldenDataView, outputDataView, eps);
        case DT_INT32: return CompareData<int32_t>(goldenDataView, outputDataView, eps);
        case DT_INT64: return CompareData<int64_t>(goldenDataView, outputDataView, eps);
        case DT_FP16: return CompareData<npu::tile_fwk::float16>(goldenDataView, outputDataView, eps);
        case DT_FP32: return CompareData<float>(goldenDataView, outputDataView, eps);
        case DT_BF16: return CompareData<npu::tile_fwk::bfloat16>(goldenDataView, outputDataView, eps);
        case DT_UINT8: return CompareData<uint8_t>(goldenDataView, outputDataView, eps);
        case DT_UINT16: return CompareData<uint16_t>(goldenDataView, outputDataView, eps);
        case DT_UINT32: return CompareData<uint32_t>(goldenDataView, outputDataView, eps);
        case DT_UINT64: return CompareData<uint64_t>(goldenDataView, outputDataView, eps);
        case DT_DOUBLE: return CompareData<double>(goldenDataView, outputDataView, eps);
        case DT_BOOL: return CompareData<uint8_t>(goldenDataView, outputDataView, eps);
        default: ASSERT(false); break;
    }
    return CompareResult();
}

bool FlowVerifier::VerifyResult(const std::string &key,
    const std::vector<std::shared_ptr<LogicalTensorData>> &goldenDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>> &outputDataViewList, float eps) {
    ASSERT(goldenDataViewList.size() == outputDataViewList.size());
    for (size_t k = 0; k < goldenDataViewList.size(); k++) {
        auto &goldenView = goldenDataViewList[k];
        auto &outputView = outputDataViewList[k];
        if (goldenView == nullptr || outputView == nullptr) {
            continue;
        }
        auto result = VerifyResult(goldenView, outputView, eps);
        if (!result.Check()) {
            ALOG_ERROR(key, ":\n    Verify for ", goldenDataViewList.size(), " data view list index ", k, " result ", TTY_RED("FAILED"));
            ALOG_ERROR(key, result.Dump());
            return false;
        } else {
            ALOG_INFO(key, ": Verify for data ", k, " result ", TTY_GREEN("SUCCEED"));
        }
    }
    return true;
}

void FlowVerifier::UpdateInterpreterCache() {
    auto &cache = Program::GetInstance().GetFunctionCache();
    std::unordered_map<FunctionHash, Function *> hashDict;
    cache.BuildHashDict(functionInterpreter_->GetEntry(), hashDict);
    functionInterpreter_->UpdateHashDict(hashDict);
}

void FlowVerifier::VerifyTensorGraph(Function *entry,
    const std::vector<std::shared_ptr<LogicalTensorData>> &inputDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>> &outputDataViewList,
    const std::vector<std::shared_ptr<LogicalTensorData>> &goldenDataViewList,
    const std::shared_ptr<TensorSlotManager> &slotManager) {
    entry_ = entry;
    inputDataViewList_ = inputDataViewList;
    outputDataViewList_ = outputDataViewList;
    goldenDataViewList_ = goldenDataViewList;

    ASSERT(!strcmp(calc::Model(), "torch")) << "Tensor graph verification requires torch model. Current model: '" << calc::Model() << "'. Please enable torch backend for tensor graph verification.";
    auto attr = entry->GetDyndevAttribute();
    std::vector<int> inputSlotList = slotManager->LookupSlotIndexConst(attr->startArgsInputTensorList);
    std::vector<int> outputSlotList = slotManager->LookupSlotIndexConst(attr->startArgsOutputTensorList);

    std::unordered_map<int, TileOpFormat> slotTileOpFormatDict;
    std::unordered_map<int, std::shared_ptr<LogicalTensorData>> slotDataViewDict;
    std::unordered_set<int> outputSlotSet;

    ASSERT(inputSlotList.size() == attr->startArgsInputTensorList.size());
    ASSERT(inputDataViewList.size() == inputSlotList.size());
    for (size_t i = 0; i < inputDataViewList.size(); i++) {
        auto inputTensor = attr->startArgsInputTensorList[i].get().GetStorage();
        if (inputTensor == nullptr) {
            continue;
        }
        auto tileop = inputTensor->GetTileOpFormat();

        auto input = inputDataViewList[i];
        ASSERT(inputTensor->Datatype() == input->GetDataType());
        ASSERT(inputTensor->GetShape() == input->GetShape());
        if (tileop == TileOpFormat::TILEOP_NZ) {
            slotTileOpFormatDict[inputSlotList[i]] = TileOpFormat::TILEOP_NZ;
        }
        slotDataViewDict[inputSlotList[i]] = input;
    }
    ASSERT(outputDataViewList.size() == outputSlotList.size());
    for (size_t i = 0; i < outputDataViewList.size(); i++) {
        slotDataViewDict[outputSlotList[i]] = outputDataViewList[i];
        auto outputTensor = attr->startArgsOutputTensorList[i].get().GetStorage();
        auto tileop = outputTensor->GetTileOpFormat();
        if (tileop == TileOpFormat::TILEOP_NZ) {
            slotTileOpFormatDict[outputSlotList[i]] = TileOpFormat::TILEOP_NZ;
        }
    }
    outputSlotSet.insert(outputSlotList.begin(), outputSlotList.end());

    std::unordered_map<std::string, ScalarImmediateType> controlFlowSymbolDict;
    const std::vector<std::string> &inputNameList = slotManager->GetInputNameList();
    const std::vector<std::string> &outputNameList = slotManager->GetOutputNameList();
    for (size_t i = 0; i < inputNameList.size(); i++) {
        controlFlowSymbolDict[AddArgPrefix(inputNameList[i])] = SymbolicScalar(i);
    }
    for (size_t i = 0; i < outputNameList.size(); i++) {
        controlFlowSymbolDict[AddArgPrefix(outputNameList[i])] = SymbolicScalar(i);
    }

    functionInterpreter_ = std::make_shared<FunctionInterpreter>();
    functionInterpreter_->Initialize(entry, inputDataViewList_);
    functionInterpreter_->verifyType = VerifyType::TENSOR_GRAPH;
    UpdateInterpreterCache();

    if (config::GetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_DUMP_OPERATION, false)) {
        functionInterpreter_->DumpSetLevelOperation();
    }
    if (config::GetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_DUMP_TENSOR, false)) {
        functionInterpreter_->DumpSetLevelTensor();
    }

    auto tensorDir = config::LogTopFolder() + "/tensor";
    CreateMultiLevelDir(tensorDir);

    controlFlowExecution_ =
        functionInterpreter_->RunForControlFlow("tensor_graph", goldenDataViewList_, slotTileOpFormatDict, slotDataViewDict, outputSlotSet, controlFlowSymbolDict);

    if (config::GetPlatformConfig(KEY_VERIFY_DUMP_PERF_DATA, false)) {
        ALOG_EVENT(entry->GetMagicName() + "_tensor_graph\n", functionInterpreter_->DumpStatistics());
    }
    functionInterpreter_->DumpReset();

    if (config::GetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_CHECK_PRECISION, true)) {
        auto tensorGraphResult = VerifyResult("Tensor graph", goldenDataViewList_, outputDataViewList_, static_cast<float>(1e-3));
        ASSERT(tensorGraphResult) << "Verify Tensor Graph Fail!";
    }
}

static bool VerifyPassDumpOperation(int passIndex) {
    if (config::GetPlatformConfig(KEY_VERIFY_PASS_DUMP_OPERATION, false)) {
        return true;
    }
    std::vector<int> select = config::GetPlatformConfig<std::vector<int>>(KEY_VERIFY_PASS_DUMP_OPERATION_SELECT, {});
    for (auto &i : select) {
        if (i == passIndex) {
            return true;
        }
    }
    return false;
}
static bool VerifyPassDumpTensor(int passIndex) {
    if (config::GetPlatformConfig(KEY_VERIFY_PASS_DUMP_TENSOR, false)) {
        return true;
    }
    std::vector<int> select = config::GetPlatformConfig<std::vector<int>>(KEY_VERIFY_PASS_DUMP_TENSOR_SELECT, {});
    for (auto &i : select) {
        if (i == passIndex) {
            return true;
        }
    }
    return false;
}

template <typename T>
static std::string ToString(const T &val, size_t totalSize) {
    std::string data = std::to_string(val);
    if (totalSize < data.size()) {
        return data;
    } else {
        return std::string(totalSize - data.size(), '0') + data;
    }
}

void FlowVerifier::VerifyPass(Function *func, int passIndex, const std::string &passIdentifier) {
    functionInterpreter_->verifyType = VerifyType::PASS;
    UpdateInterpreterCache();
    if (controlFlowExecution_->executionListDict.count(func) == 0) {
        return;
    }

    auto &captureList = controlFlowExecution_->executionListDict.find(func)->second;
    if (!lastCaptureExecution_.count(func)) {
        lastCaptureExecution_[func].resize(captureList.size());
    }

    if (VerifyPassDumpOperation(passIndex)) {
        functionInterpreter_->DumpSetLevelOperation();
    }
    if (VerifyPassDumpTensor(passIndex)) {
        functionInterpreter_->DumpSetLevelTensor();
    }
    for (size_t captureIndex = 0; captureIndex < captureList.size(); captureIndex++) {
        const std::string key = "function_" + func->GetMagicName() + ".pass_" + ToString(passIndex, 2) + "_" +
                                passIdentifier + ".capture_" + ToString(captureIndex, 3);
        ALOG_INFO(key, ": Verify");

        std::shared_ptr<FunctionCaptureExecution> capture = nullptr;
        float eps = static_cast<float>(1e-3);
        constexpr int SELECT_ALL_PASS = -1;
        int selectIndex = config::GetPlatformConfig(KEY_VERIFY_PASS_SELECT, SELECT_ALL_PASS);
        if (passIndex == 0 || passIndex == selectIndex - 1) {
            /* The first passes uses tensor graph's execution golden */
            capture = captureList[captureIndex];
        } else {
            /* The rest passes uses previous pass's execution result as input */
            capture = lastCaptureExecution_[func][captureIndex];
            eps = static_cast<float>(1e-3);
        }

        auto captureExecution = functionInterpreter_->RunForPass(key, func, capture);
        auto goldenDataViewList = capture->golden->outcastDataViewList;
        auto executeDataViewList = captureExecution->golden->outcastDataViewList;
        /* record it */
        lastCaptureExecution_[func][captureIndex] = captureExecution;

        if (config::GetPlatformConfig(KEY_VERIFY_PASS_CHECK_PRECISION, true)) {
            auto passResult = VerifyResult(key, goldenDataViewList, executeDataViewList, eps);
            ASSERT(passResult) << "Verify Pass Fail!";
        }
    }

    if (config::GetPlatformConfig(KEY_VERIFY_DUMP_PERF_DATA, false)) {
        ALOG_EVENT(func->GetMagicName() + "_" + passIdentifier + "\n", functionInterpreter_->DumpStatistics());
    }
    functionInterpreter_->DumpReset();
}

void FlowVerifier::VerifyExecuteGraph() {
    functionInterpreter_->verifyType = VerifyType::EXECUTE_GRAPH;
    UpdateInterpreterCache();

    for (auto &[func, captureList]: controlFlowExecution_->executionListDict) {
        for (size_t captureIndex = 0; captureIndex < captureList.size(); captureIndex++) {
            const std::string key =
                "function_" + func->GetMagicName() + ".exec_graph.capture_" + ToString(captureIndex, 3);
            ALOG_INFO(key, ": Verify");
            if (config::GetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_DUMP_OPERATION, false)) {
                functionInterpreter_->DumpSetLevelOperation();
            }
            if (config::GetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_DUMP_TENSOR, false)) {
                functionInterpreter_->DumpSetLevelTensor();
            }
            auto &capture = captureList[captureIndex];
            auto captureExecution = functionInterpreter_->RunForExecuteGraph(key, func, capture);

            if (config::GetPlatformConfig(KEY_VERIFY_DUMP_PERF_DATA, false)) {
                ALOG_EVENT(func->GetMagicName() + "_tensor_graph\n", functionInterpreter_->DumpStatistics());
            }
            functionInterpreter_->DumpReset();

            if (config::GetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_CHECK_PRECISION, true)) {
                auto executeResult = VerifyResult(key, capture->golden->outcastDataViewList, captureExecution->golden->outcastDataViewList,
                    static_cast<float>(1e-3));
                ASSERT(executeResult) << "Verify Execute Graph Fail!";
            }
        }
    }
}

FlowVerifier &FlowVerifier::GetInstance() {
    static FlowVerifier flowVerifier;
    return flowVerifier;
}
}
