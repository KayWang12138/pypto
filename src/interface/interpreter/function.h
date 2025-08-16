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
 * \file function.h
 * \brief
 */
/* for flow verify tool */

#pragma once

#include "interface/tensor/tensor_slot.h"
#include "interface/interpreter/operation.h"
#include "interface/tensor/symbolic_scalar_evaluate.h"


namespace npu::tile_fwk {

struct FunctionIODataPair {
    std::vector<std::shared_ptr<LogicalTensorData>> incastDataViewList;
    std::vector<std::shared_ptr<LogicalTensorData>> outcastDataViewList;

    FunctionIODataPair() {}
    FunctionIODataPair(std::vector<std::shared_ptr<LogicalTensorData>> incastDataViewList_,
        std::vector<std::shared_ptr<LogicalTensorData>> outcastDataViewList_)
        : incastDataViewList(incastDataViewList_), outcastDataViewList(outcastDataViewList_) {}

    // One tensor might occur in both incast and outcast simultaneously for multiple times, so we must copy
    // simultaneously
    static void CopyWithLinkRelationship(FunctionIODataPair &dst, const FunctionIODataPair &src) {
        struct CopyInfo {
            bool isIncast;
            int index;
            CopyInfo(bool isIncast_, int index_) : isIncast(isIncast_), index(index_) {}
        };

        std::unordered_map<std::shared_ptr<LogicalTensorData>, std::vector<CopyInfo>> copyInfoDict;
        for (size_t k = 0; k < src.incastDataViewList.size(); k++) {
            copyInfoDict[src.incastDataViewList[k]].emplace_back(true, k);
        }
        for (size_t k = 0; k < src.outcastDataViewList.size(); k++) {
            copyInfoDict[src.outcastDataViewList[k]].emplace_back(false, k);
        }

        dst.incastDataViewList.resize(src.incastDataViewList.size());
        dst.outcastDataViewList.resize(src.outcastDataViewList.size());
        for (auto &[srcDataView, copyInfoList] : copyInfoDict) {
            auto dstData = srcDataView->DeepCopy();
            for (auto &[isIncast, index] : copyInfoList) {
                if (isIncast) {
                    dst.incastDataViewList[index] = dstData;
                } else {
                    dst.outcastDataViewList[index] = dstData;
                }
            }
        }
        for (size_t k = 0; k < dst.incastDataViewList.size(); k++) {
            ASSERT(dst.incastDataViewList[k] != nullptr);
        }
        for (size_t k = 0; k < dst.outcastDataViewList.size(); k++) {
            ASSERT(dst.outcastDataViewList[k] != nullptr);
        }
    }
};

struct FunctionFrame {
    const Function *func;
    const Operation *callop;
    const std::shared_ptr<CallOpAttribute> callopAttr;
    std::shared_ptr<FunctionIODataPair> inoutDataPair;
    std::unordered_map<std::shared_ptr<RawTensor>, std::shared_ptr<RawTensorData>> rawTensorDataDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<RawTensor>> spillRawTensorDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensorData>> tensorDataViewDict;
    int frameIndex;

    Operation *currentOperation;

    const std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensorData>> &GetTensorDataViewDict() const { return tensorDataViewDict; }

    int GetFrameIndex() const { return frameIndex; }

    FunctionFrame(const Function *func_, const Operation *callop_,
        const std::shared_ptr<CallOpAttribute> &callopAttr_, std::shared_ptr<FunctionIODataPair> inoutDataPair_,
        int frameIndex_)
        : func(func_),
          callop(callop_),
          callopAttr(callopAttr_),
          inoutDataPair(inoutDataPair_),
          frameIndex(frameIndex_) {
        if (inoutDataPair != nullptr) {
            ASSERT(func->GetIncast().size() == inoutDataPair->incastDataViewList.size());
            for (size_t i = 0; i < inoutDataPair->incastDataViewList.size(); i++) {
                AddDataView(func->GetIncast()[i], inoutDataPair->incastDataViewList[i]);
            }
            ASSERT(func->GetOutcast().size() == inoutDataPair->outcastDataViewList.size());
            for (size_t i = 0; i < inoutDataPair->outcastDataViewList.size(); i++) {
                AddDataView(func->GetOutcast()[i], inoutDataPair->outcastDataViewList[i]);
            }
        }
    }

    void UpdateCurrentOperation(Operation *op) { currentOperation = op; }

    std::shared_ptr<LogicalTensorData> GetDataView(const std::shared_ptr<LogicalTensor> &tensor) {
        if (!tensorDataViewDict.count(tensor)) {
            return nullptr;
        }
        auto view = tensorDataViewDict[tensor];
        return view;
    }

    std::vector<std::shared_ptr<LogicalTensorData>> GetDataViewList(
        const std::vector<std::shared_ptr<LogicalTensor>> &tensorList) {
        std::vector<std::shared_ptr<LogicalTensorData>> viewList(tensorList.size());
        for (size_t i = 0; i < tensorList.size(); i++) {
            viewList[i] = GetDataView(tensorList[i]);
        }
        return viewList;
    }

    void AddDataView(
        const std::shared_ptr<LogicalTensor> &tensor, const std::shared_ptr<LogicalTensorData> &dataView) {
        if (tensorDataViewDict.count(tensor)) {
            ASSERT(tensorDataViewDict[tensor] == dataView);
        } else {
            DoAddTensorDataView(tensor, dataView);
            DoAddRawTensorDataView(tensor->GetRawTensor(), dataView->GetData());
        }
    }
    void AddDataViewList(const std::vector<std::shared_ptr<LogicalTensor>> &tensorList,
        const std::vector<std::shared_ptr<LogicalTensorData>> &dataViewList) {
        ASSERT(tensorList.size() == dataViewList.size());
        for (size_t i = 0; i < tensorList.size(); i++) {
            AddDataView(tensorList[i], dataViewList[i]);
        }
    }

    std::shared_ptr<LogicalTensorData> AllocateDataView(const std::shared_ptr<LogicalTensor> &tensor,
        const std::vector<int> &offset, const std::vector<int> &validShape) {
        if (tensorDataViewDict.count(tensor)) {
            return tensorDataViewDict[tensor];
        }

        auto raw = tensor->GetRawTensor();
        bool isSpilled = false;

        std::string spillRawMaigc = "1056964608";
        std::string rawMagic = std::to_string(tensor->GetRawTensor()->GetRawMagic());
        if (rawMagic.find(spillRawMaigc) != std::string::npos) {
            if (spillRawTensorDict.count(tensor)) {
                raw = spillRawTensorDict[tensor];
            } else {
                raw = std::make_shared<RawTensor>(raw->GetDataType(), raw->GetRawShape());
                DoAddSpillRawTensor(tensor, raw);
            }
            isSpilled = true;
        }
        std::shared_ptr<RawTensorData> rawData;
        if (rawTensorDataDict.count(raw)) {
            rawData = rawTensorDataDict[raw];
        } else {
            rawData = std::make_shared<RawTensorData>(raw->GetDataType(), raw->GetRawShape());
            rawData->resize(rawData->GetElementSize() * rawData->GetSize());
            DoAddRawTensorDataView(raw, rawData);
        }
        std::shared_ptr<LogicalTensorData> view =
            std::make_shared<LogicalTensorData>(rawData, tensor->GetShape(), validShape, offset);
        view->SetIsSpilled(isSpilled);
        DoAddTensorDataView(tensor, view);
        return view;
    }

    std::vector<std::shared_ptr<LogicalTensorData>> AllocateDataViewList(
        const std::vector<std::shared_ptr<LogicalTensor>> &tensorList, const std::vector<std::vector<int>> &offsetList,
        const std::vector<std::vector<int>> &validShapeList) {
        std::vector<std::shared_ptr<LogicalTensorData>> dataViewList(tensorList.size());
        for (size_t i = 0; i < tensorList.size(); i++) {
            dataViewList[i] = AllocateDataView(tensorList[i], offsetList[i], validShapeList[i]);
        }
        return dataViewList;
    }

private:
    void DoAddTensorDataView(
            const std::shared_ptr<LogicalTensor> &tensor,
            const std::shared_ptr<LogicalTensorData> &dataView) {
        ASSERT(!tensorDataViewDict.count(tensor));
        tensorDataViewDict[tensor] = dataView;
    }
    void DoAddRawTensorDataView(
            const std::shared_ptr<RawTensor> &rawTensor,
            const std::shared_ptr<RawTensorData> &data) {
        ASSERT(!rawTensorDataDict.count(rawTensor));
        rawTensorDataDict[rawTensor] = data;
    }
    void DoAddSpillRawTensor(
        const std::shared_ptr<LogicalTensor> &tensor, const std::shared_ptr<RawTensor> &rawtensor) {
        ASSERT(!spillRawTensorDict.count(tensor));
        spillRawTensorDict[tensor] = rawtensor;
    }
};

struct FunctionCaptureExecution {
    Function *func;
    std::shared_ptr<FunctionIODataPair> baseline;
    std::unordered_map<std::string, ScalarImmediateType> symbolDict;

    std::vector<std::shared_ptr<FunctionFrame>> frameList;

    std::shared_ptr<FunctionIODataPair> golden;

    FunctionCaptureExecution(Function *func_ = nullptr) : func(func_) {
        baseline = std::make_shared<FunctionIODataPair>();
        golden = std::make_shared<FunctionIODataPair>();
    }

    const std::vector<std::shared_ptr<FunctionFrame>> &GetFrameList() const { return frameList; }

    void CaptureFrom(
            const std::shared_ptr<FunctionIODataPair> &b,
            const std::unordered_map<std::string, ScalarImmediateType> &s) {
        FunctionIODataPair::CopyWithLinkRelationship(*baseline, *b);
        symbolDict = s;
    }

    void CaptureSymbolDictFrom(
            const std::unordered_map<std::string, ScalarImmediateType> &s) {
        symbolDict = s;
    }

    void CaptureGoldenFrom(
            const std::shared_ptr<FunctionIODataPair> &g) {
        FunctionIODataPair::CopyWithLinkRelationship(*golden, *g);
    }

    std::unordered_map<std::string, ScalarImmediateType> CaptureTo(
            std::shared_ptr<FunctionIODataPair> &c) const {
        FunctionIODataPair::CopyWithLinkRelationship(*c, *baseline);
        return symbolDict;
    }
};

struct FunctionControlFlowExecution {
    std::unordered_map<Function *, std::vector<std::shared_ptr<FunctionCaptureExecution>>> executionListDict;
};

constexpr int EXEC_DUMP_LEVEL_OPERATION = 1;
constexpr int EXEC_DUMP_LEVEL_TENSOR = 2;

enum class VerifyType { INVALID, TENSOR_GRAPH, PASS, EXECUTE_GRAPH };

struct FunctionInterpreter {
    FunctionInterpreter(int threadCount)
        : operationInterpreter(std::make_shared<OperationInterpreter>(threadCount)) {}

    Function *entry_;
    std::unordered_map<FunctionHash, Function *> calleeHashDict;
    std::unordered_set<int> outputSlotSet_;
    std::shared_ptr<OperationInterpreter> operationInterpreter;
    std::unordered_map<int, std::shared_ptr<LogicalTensorData>> slotDataViewDict_;
    std::vector<std::shared_ptr<FunctionFrame>> *captureFrameList{nullptr};

    int execDumpLevel{0};

    std::string execDumpDir;
    FILE *execDumpFile{nullptr};
    FILE *execDumpStyleFile{nullptr};
    std::string execDumpFuncKey;
    std::vector<ElementDump> execDumpElementList;
    std::vector<std::shared_ptr<FunctionFrame>> execDumpStack;
    int frameCount{0};

    std::map<std::string, uint64_t> opUsage;
    uint64_t dumpTensorUsage{0};
    uint64_t dumpOperationUsage{0};
    uint64_t totalTimeUsage{0};

    VerifyType verifyType{VerifyType::INVALID};

    int GetThreadCount() const { return operationInterpreter->GetThreadCount(); }

    std::vector<std::shared_ptr<LogicalTensorData>> &GetInputDataViewList() {
        return operationInterpreter->evaluateSymbol->GetInputDataViewList();
    }
    void UpdateInputDataViewList(size_t index, const std::shared_ptr<LogicalTensorData> &inputDataView) {
        operationInterpreter->evaluateSymbol->UpdateInputDataViewList(index,inputDataView);
    }
    void InitInputDataViewList(const std::vector<std::shared_ptr<LogicalTensorData>> &inputDataViewList) {
        operationInterpreter->evaluateSymbol->InitInputDataViewList(inputDataViewList);
    }
    const std::unordered_map<std::string, ScalarImmediateType> &GetSymbolDict() const {
        return operationInterpreter->evaluateSymbol->GetSymbolDict();
    }
    void UpdateSymbolDict(const std::string key, const ScalarImmediateType value) {
        operationInterpreter->evaluateSymbol->UpdateSymbolDict(key, value);
    }
    void SetSymbolDict(const std::unordered_map<std::string, ScalarImmediateType> &symbolDict) {
        operationInterpreter->evaluateSymbol->SetSymbolDict(symbolDict);
    }

    ScalarImmediateType EvaluateSymbolicScalar(const SymbolicScalar &ss) {
        return operationInterpreter->EvaluateSymbolicScalar(ss);
    }
    std::vector<int> EvaluateOffset(const std::vector<int> &offset, const std::vector<SymbolicScalar> &dynOffset){
        return operationInterpreter->EvaluateOffset(offset, dynOffset);
    }
    std::vector<int> EvaluateValidShape(const std::vector<SymbolicScalar> &dynValidShape) {
        return operationInterpreter->EvaluateValidShape(dynValidShape);
    }
    void EvaluateDynParam(
        const std::map<std::string, DynParamInfo> &dynParamTable, const std::vector<SymbolicScalar> &linearArgList) {
        operationInterpreter->evaluateSymbol->EvaluateDynParam(dynParamTable, linearArgList);
    }

    size_t GetFrameSize() const { return execDumpStack.size(); }
    std::string GetFrameIndex(const std::shared_ptr<FunctionFrame> &frame) const {
        if (frame == nullptr) {
            return "null";
        } else {
            return std::to_string(frame->GetFrameIndex());
        }
    }
    std::shared_ptr<FunctionFrame> GetFrameCurr() const {
        if (execDumpStack.size() == 0) {
            return nullptr;
        } else {
            return execDumpStack.back();
        }
    }
    std::string GetFrameCurrIndex() const {
        return GetFrameIndex(GetFrameCurr());
    }

    std::shared_ptr<LogicalTensorData> FormatNZ2ND(std::shared_ptr<LogicalTensorData> &view) {
        /*  For shape <a0, a1, ..., an, K * B> ND, NZ is generated by:
         *      1. reshape to <a0 * a1 * ... * an, K, B>
         *      2. transpose to <K, a0 * a1 * ... * an, B>
         *      3. reshape to <a0, a1, ..., an, K * B>
         *
         *  Note that the last's dimension's SIZE is 32 bytes. Hence B is computed by the type: 32 / GetDataSize(GetDataType())
         *
         *  Hence, when NZ is converted to ND, the steps are:
         *      1. reshape to <K, a0 * a1 * ... * an, B>
         *      2. transpose to <a0 * a1 * ... * an, K, B>
         *      3. reshape to <a0, a1, ..., an, K * B>
         */
        int amul = 1;
        constexpr int NZ_BLOCK_SIZE = 32;
        int block = NZ_BLOCK_SIZE / RawTensorData::GetDataSize(view->GetDataType());
        ASSERT(view->GetShape().back() % block == 0);
        for (size_t index = 0; index < view->GetShape().size() - 1; index++) {
            amul *= view->GetShape()[index];
        }
        int k = view->GetShape().back() / block;

        std::shared_ptr<LogicalTensorData> step0 =
            LogicalTensorData::CreateEmpty(view->GetDataType(), {k, amul, block}, std::vector<int>(0));
        std::shared_ptr<LogicalTensorData> step1 =
            LogicalTensorData::CreateEmpty(view->GetDataType(), {amul, k, block}, std::vector<int>(0));
        std::shared_ptr<LogicalTensorData> step2 =
            LogicalTensorData::CreateEmpty(view->GetDataType(), view->GetShape(), view->GetValidShape());

        Calculator::CalcReshape(step0.get(), view.get(), &operationInterpreter->GetPool());
        Calculator::CalcTransposeAdjDim(step1.get(), step0.get(), 0, &operationInterpreter->GetPool());
        Calculator::CalcReshape(step2.get(), step1.get(), &operationInterpreter->GetPool());
        return step2;
    }

    void Initialize(
            Function *entry,
            const std::vector<std::shared_ptr<LogicalTensorData>> &inputDataViewList) {
        entry_ = entry;
        InitInputDataViewList(inputDataViewList);
    }

    Function *GetEntry() const { return entry_; }

    Function *GetCallee(const Operation *callop) {
        auto calleeHash = callop->GetCalleeHash();
        ASSERT(calleeHashDict.count(calleeHash));
        Function *callee = calleeHashDict.find(calleeHash)->second;
        return callee;
    }

    void UpdateHashDict(const std::unordered_map<FunctionHash, Function *> &hashDict) {
        for (auto &[hash, callee] : hashDict) {
            if (calleeHashDict.count(hash)) {
                ASSERT(calleeHashDict.find(hash)->second == callee);
            } else {
                calleeHashDict[hash] = callee;
            }
        }
    }

    std::shared_ptr<LogicalTensorData> AllocateDataView(
        FunctionFrame &frame, const std::shared_ptr<LogicalTensor> &tensor) {
        std::vector<int> offset = EvaluateOffset(tensor->GetOffset(), tensor->GetDynOffset());
        // 方案待定std::vector<int> validShape = EvaluateValidShape(tensor->GetDynValidShape());.
        std::vector<int> validShape(0);
        auto ret = frame.AllocateDataView(tensor, offset, validShape);
        return ret;
    }
    std::vector<std::shared_ptr<LogicalTensorData>> AllocateDataViewList(
        FunctionFrame &frame, const std::vector<std::shared_ptr<LogicalTensor>> &tensorList) {
        std::vector<std::vector<int>> offsetList(tensorList.size());
        std::vector<std::vector<int>> validShapeList(tensorList.size());
        for (size_t k = 0; k < tensorList.size(); k++) {
            offsetList[k] = EvaluateOffset(tensorList[k]->GetOffset(), tensorList[k]->GetDynOffset());
            // 方案待定validShapeList[k] =  EvaluateValidShape(tensorList[k]->GetDynValidShape());
            validShapeList[k] =  {};
        }
        auto ret = frame.AllocateDataViewList(tensorList, offsetList, validShapeList);
        return ret;
    }

    void ExecuteOpCallLeaf(ExecuteOperationContext *ctx) {
        Function *callee = GetCallee(ctx->op);
        auto inoutDataPair =
            std::make_shared<FunctionIODataPair>(*ctx->ioperandDataViewList, *ctx->ooperandInplaceDataViewList);
        ExecuteFunctionFrame(callee, ctx->op, inoutDataPair);
    }

    void ExecuteOperation(FunctionFrame &frame, Operation *op) {
        std::vector<std::shared_ptr<LogicalTensorData>> ioperandDataViewList = frame.GetDataViewList(op->GetIOperands());
        for (size_t index = 0; index < ioperandDataViewList.size(); index++) {
            if (ioperandDataViewList[index] == nullptr) {
                auto iop = op->GetIOperands()[index];
                ASSERT(op->GetOpcode() == Opcode::OP_CALL);
                ioperandDataViewList[index] = AllocateDataView(frame, iop);
            }
        }
        std::vector<std::shared_ptr<LogicalTensorData>> ooperandDataViewList =
            AllocateDataViewList(frame, op->GetOOperands());
        ExecuteOperationContext ctx = {&frame, op, &ioperandDataViewList, {}, &ooperandDataViewList};

        if(op->GetOpcode() == Opcode::OP_CALL) {
            ExecuteOpCallLeaf(&ctx);
        } else {
            TimeStamp ts;
            operationInterpreter->ExecuteOperation(&ctx);
            opUsage[op->GetOpcodeStr()] += ts.Duration();
        }
        std::vector<std::shared_ptr<LogicalTensorData>> *ooperandDumpList =
            (&ctx)->ooperandInplaceDataViewList ? (&ctx)->ooperandInplaceDataViewList : (&ctx)->ooperandDataViewList;
        TimeStamp ts;
        DumpOperationTensor((&ctx)->op, ooperandDumpList, (&ctx)->ioperandDataViewList);
        dumpOperationUsage += ts.Duration();
    }

    void ExecuteHandleFunctionBegin(Function *func, std::shared_ptr<FunctionFrame> frame) {
        TimeStamp ts;
        execDumpStack.push_back(frame);
        DumpFunctionHead(func);
        if (frame->inoutDataPair != nullptr) {
            DumpTensorList("Incast", &func->GetIncast(), &frame->inoutDataPair->incastDataViewList);
            for (size_t k = 0; k < func->GetIncast().size(); k++) {
                DumpTensorBinary(func->GetIncast()[k], frame->inoutDataPair->incastDataViewList[k]);
            }
        }
        dumpTensorUsage += ts.Duration();
    }
    void ExecuteHandleFunctionEnd() { execDumpStack.pop_back(); }
    void ExecuteHandleOperationBegin(Operation *op) {
        execDumpStack.back()->UpdateCurrentOperation(op);
        TimeStamp ts;
        DumpOperation(op);
        dumpOperationUsage += ts.Duration();
    }
    void ExecuteHandleOperationEnd() {}

    std::shared_ptr<FunctionFrame> ExecuteFunctionFrame(
        Function *func, Operation *callop, std::shared_ptr<FunctionIODataPair> &inoutDataPair) {
        std::shared_ptr<CallOpAttribute> callopAttr;
        std::vector<SymbolicScalar> linearArgList;
        if (callop != nullptr) {
            callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
            linearArgList = callopAttr->GetLinearArgList();
        }
        std::shared_ptr<FunctionFrame> frame =
            std::make_shared<FunctionFrame>(func, callop, callopAttr, inoutDataPair, frameCount++);
        captureFrameList->push_back(frame);

        auto dynParamTable = func->GetDynParamTable();
        EvaluateDynParam(dynParamTable, linearArgList);

        ExecuteHandleFunctionBegin(func, frame);
        for (auto &op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_PRINT && verifyType != VerifyType::TENSOR_GRAPH)
                continue;
            ExecuteHandleOperationBegin(&op);
            ExecuteOperation(*frame, &op);
            ExecuteHandleOperationEnd();
        }
        ExecuteHandleFunctionEnd();
        return frame;
    }

    std::vector<std::shared_ptr<FunctionFrame>> ExecuteFunctionCapture(Function *func, Operation *callop, std::shared_ptr<FunctionIODataPair> &inoutDataPair) {
        std::vector<std::shared_ptr<FunctionFrame>> frameList;
        captureFrameList = &frameList;
        ExecuteFunctionFrame(func, callop, inoutDataPair);
        return frameList;
    }

    void ExecuteFunctionDynamic(Function *func, FunctionControlFlowExecution &controlFlowExecution) {
        std::shared_ptr<FunctionFrame> frame = std::make_shared<FunctionFrame>(func, nullptr, nullptr, nullptr, frameCount++);
        ExecuteHandleFunctionBegin(func, frame);

        std::vector<Operation *> callopList = func->GetCallopList();
        for (auto callop : callopList) {
            Function *callee = GetCallee(callop);

            ExecuteHandleOperationBegin(callop);
            ExecuteControlFlow(callee, controlFlowExecution);
            ExecuteHandleOperationEnd();
        }

        ExecuteHandleFunctionEnd();
    }

    Operation *ExecuteFunctionLoopLookupSat(const std::shared_ptr<DynloopFunctionAttribute> &controlFlowExecution) {
        for (auto &path : controlFlowExecution->pathList) {
            bool sat = true;
            for (auto cond : path.pathCondList) {
                if (static_cast<bool>(EvaluateSymbolicScalar(cond.GetCond())) != cond.IsSat()) {
                    sat = false;
                    break;
                }
            }
            if (!sat) {
                continue;
            }
            return path.callop;
        }
        return nullptr;
    }

    void ExecuteFunctionLoop(Function *func, FunctionControlFlowExecution &controlFlowExecution) {
        std::shared_ptr<FunctionFrame> frame = std::make_shared<FunctionFrame>(func, nullptr, nullptr, nullptr, frameCount++);
        ExecuteHandleFunctionBegin(func, frame);

        auto loop = func->GetDynloopAttribute();
        ScalarImmediateType begin = EvaluateSymbolicScalar(loop->Begin());
        ScalarImmediateType end = EvaluateSymbolicScalar(loop->End());
        ScalarImmediateType step = EvaluateSymbolicScalar(loop->Step());
        for (ScalarImmediateType idx = begin; idx < end; idx += step) {
            UpdateSymbolDict(loop->IterSymbolName(), idx);
            Operation *callop = ExecuteFunctionLoopLookupSat(loop);
            Function *callee = GetCallee(callop);

            ExecuteHandleOperationBegin(callop);
            ExecuteControlFlow(callee, controlFlowExecution);
            ExecuteHandleOperationEnd();
        }

        ExecuteHandleFunctionEnd();
    }

    void ExecuteControlFlow(Function *func, FunctionControlFlowExecution &controlFlowExecution) {
        func->SortOperations();
        auto funcType = func->GetFunctionType();
        if (funcType == FunctionType::DYNAMIC) {
            ExecuteFunctionDynamic(func, controlFlowExecution);
        } else if (funcType == FunctionType::DYNAMIC_LOOP) {
            ExecuteFunctionLoop(func, controlFlowExecution);
        } else if (func->IsGraphType(GraphType::TENSOR_GRAPH)) {
            std::vector<Operation *> callopList = func->GetCallopList();
            if (callopList.size() != 0) {
                ExecuteFunctionDynamic(func, controlFlowExecution);
            } else {
                auto &incastSlot = func->GetSlotScope()->ioslot.incastSlot;
                auto &outcastSlot = func->GetSlotScope()->ioslot.outcastSlot;

                auto getOutputSlot = [this](const std::vector<int> &slotList) {
                    for (auto &slot : slotList) {
                        if (this->outputSlotSet_.count(slot)) {
                            return slot;
                        }
                    }
                    return -1;
                };

                auto inoutDataPair = std::make_shared<FunctionIODataPair>();

                ASSERT(func->GetIncast().size() == incastSlot.size());
                for (size_t i = 0; i < func->GetIncast().size(); i++) {
                    int slot = incastSlot[i][0];
                    ASSERT(slotDataViewDict_.count(slot));
                    auto incastDataView = slotDataViewDict_[slot];
                    inoutDataPair->incastDataViewList.push_back(incastDataView);
                }

                ASSERT(func->GetOutcast().size() == outcastSlot.size());
                for (size_t i = 0; i < func->GetOutcast().size(); i++) {
                    int outputSlot = getOutputSlot(outcastSlot[i]);
                    std::shared_ptr<LogicalTensorData> outcastView;
                    if (outputSlot != -1) {
                        outcastView = slotDataViewDict_[outputSlot];
                    } else {
                        auto outcast = func->GetOutcast()[i];
                        // 方案待定std::vector<int> validShape = EvaluateValidShape(outcast->GetDynValidShape());
                        std::vector<int> validShape = {};
                        outcastView =
                            LogicalTensorData::CreateEmpty(outcast->Datatype(), outcast->GetShape(), validShape);
                    }
                    for (auto &s : outcastSlot[i]) {
                        slotDataViewDict_[s] = outcastView;
                    }
                    inoutDataPair->outcastDataViewList.push_back(outcastView);
                }

                auto capture = std::make_shared<FunctionCaptureExecution>(func);

                capture->CaptureFrom(inoutDataPair, GetSymbolDict());
                capture->frameList = ExecuteFunctionCapture(func, nullptr, inoutDataPair);
                capture->CaptureGoldenFrom(inoutDataPair);
                controlFlowExecution.executionListDict[func].emplace_back(capture);
            }
        } else {
            ASSERT(false);
        }
    }

    std::string DumpDataView(const std::shared_ptr<LogicalTensorData> &dataView);
    std::string DumpSymbolDict() const {
        std::ostringstream oss;
        for (auto &[name, value] : GetSymbolDict()) {
            oss << name << " = " << value << "\n";
        }
        return oss.str();
    }
    void DumpOperation(Operation *op);
    void DumpOperationTensor(
            Operation *op,
            const std::vector<std::shared_ptr<LogicalTensorData>> *ooperandDataViewList,
            const std::vector<std::shared_ptr<LogicalTensorData>> *ioperandDataViewList);
    void DumpTensorBinary(
            const std::shared_ptr<LogicalTensor> &tensor,
            const std::shared_ptr<LogicalTensorData> &dataView);
    void DumpTensorList(
            const std::string &name,
            const std::vector<std::shared_ptr<LogicalTensor>> *tensorList,
            const std::vector<std::shared_ptr<LogicalTensorData>> *dataViewList);
    void DumpFunctionHead(Function *func);
    void DumpBegin();
    void DumpEnd();
    void DumpPassTensorDiff(
            const std::shared_ptr<FunctionCaptureExecution> &captureExecution,
            const std::shared_ptr<FunctionCaptureExecution> &captureGolden);
    std::string GetDumpFilePath(const std::string &lv0, const std::string &lv1, const std::string &filename);

    std::string GetDumpFrameDirName() const {
        std::string dirName = "frame_" + GetFrameCurrIndex();
        return dirName;
    }
    std::string GetDumpTensorListFileName(const std::string &name) const {
        std::string fileName = "frame_" + GetFrameCurrIndex() + "_" + name + ".html";
        return fileName;
    }
    std::string GetDumpOperationTensorFileName(Operation *op) const {
        std::string fileName = "frame_" + GetFrameCurrIndex() + "_operation_" + std::to_string(op->GetOpMagic()) + ".html";
        return fileName;
    }
    std::string GetDumpTensorFileName(const std::shared_ptr<LogicalTensor> &tensor) const {
        std::string fileName = "frame_" + GetFrameCurrIndex() + "_tensor_" + std::to_string(tensor->GetMagic()) + ".data";
        return fileName;
    }
    std::string GetDumpTensorId(const std::shared_ptr<FunctionFrame> &frame, const std::shared_ptr<LogicalTensor> &tensor) const {
        std::string index = GetFrameIndex(frame);
        std::string magic = tensor != nullptr ? std::to_string(tensor->GetMagic()) : "null";
        std::string tensorId = "tensor_" + index + "_" + magic;
        return tensorId;
    }
    std::string GetDumpTensorId(const std::shared_ptr<FunctionFrame> &frame, Operation *op) const {
        std::shared_ptr<LogicalTensor> tensor = op->GetOOperands().size() != 0 ? op->GetOOperands()[0] : nullptr;
        return GetDumpTensorId(frame, tensor);
    }
    std::string GetDumpOperationId(const std::shared_ptr<FunctionFrame> &frame, Operation *op) const {
        std::string index = GetFrameIndex(frame);
        std::string magic = op != nullptr ? std::to_string(op->GetOpMagic()) : "null";
        std::string tensorId = "operation_" + index + "_" + magic;
        return tensorId;
    }

    void DumpSetLevelOperation() { execDumpLevel = EXEC_DUMP_LEVEL_OPERATION; }
    void DumpSetLevelTensor() { execDumpLevel = EXEC_DUMP_LEVEL_TENSOR; }

    void DumpReset() {
        execDumpLevel = 0;
        opUsage.clear();
        totalTimeUsage = 0;
        dumpTensorUsage = 0;
        dumpOperationUsage = 0;
    }

    std::string DumpStatistics() const {
        std::stringstream ss;
        const int labelWidth = 24;
        uint64_t totalOpUsage = 0;
        for (auto &[opcode, time] : opUsage) {
            if (time) {
                totalOpUsage += time;
                ss << std::left << std::setw(labelWidth) << opcode << ": " << time << "\n";
            }
        }
        ss << std::left << std::setw(labelWidth) << "TotalTimeUsage" << ": " << totalTimeUsage << "\n";
        ss << std::left << std::setw(labelWidth) << "TotalOpUsage" << ": " << totalOpUsage << "\n";
        if (dumpTensorUsage) {
            ss << std::left << std::setw(labelWidth) << "TotalDumpTensorUsage:" << ": " << dumpTensorUsage << "\n";
        }
        if (dumpOperationUsage) {
            ss << std::left << std::setw(labelWidth) << "TotalDumpOperationUsage" << ": " << dumpOperationUsage << "\n";
        }
        return ss.str();
    }

    std::shared_ptr<FunctionCaptureExecution> ExecuteUnit(
            Function *func,
            const std::shared_ptr<FunctionCaptureExecution> &capture) {
        auto unitCapture = std::make_shared<FunctionCaptureExecution>(func);
        auto symbolDict = capture->CaptureTo(unitCapture->baseline);
        SetSymbolDict(symbolDict);

        Function *target = func;
        if (func->GetRootFunction()) {
            target = func->GetRootFunction();
        }
        unitCapture->CaptureGoldenFrom(unitCapture->baseline);
        unitCapture->CaptureSymbolDictFrom(capture->symbolDict);
        unitCapture->frameList = ExecuteFunctionCapture(target, nullptr, unitCapture->golden);

        DumpTensorList("Golden", &target->GetOutcast(), &capture->golden->outcastDataViewList);
        return unitCapture;
    }

    std::shared_ptr<FunctionControlFlowExecution> RunForControlFlow(
            const std::string &funcKey,
            std::vector<std::shared_ptr<LogicalTensorData>> goldenDataViewList,
            const std::unordered_map<int, TileOpFormat> &slotTileOpFormatDict,
            const std::unordered_map<int, std::shared_ptr<LogicalTensorData>> &slotDataViewDict,
            const std::unordered_set<int> &outputSlotSet,
            const std::unordered_map<std::string, ScalarImmediateType> &controlFlowSymbolDict) {
        execDumpFuncKey = funcKey;
        std::shared_ptr<FunctionControlFlowExecution> execution = std::make_shared<FunctionControlFlowExecution>();

        SetSymbolDict(controlFlowSymbolDict);
        auto findInputIndex = [this](std::shared_ptr<LogicalTensorData> &inputDataView) -> int {
            for (size_t k = 0; k < this->GetInputDataViewList().size(); k++) {
                if (this->GetInputDataViewList()[k] == inputDataView) {
                    return k;
                }
            }
            return -1;
        };
        slotDataViewDict_ = slotDataViewDict;
        outputSlotSet_ = outputSlotSet;
        for (auto &[slot, tileOpFormat]: slotTileOpFormatDict) {
            if (tileOpFormat == TileOpFormat::TILEOP_NZ) {
                ASSERT(slotDataViewDict_.count(slot));
                auto dataView = slotDataViewDict_.find(slot)->second;
                auto inputIndex = findInputIndex(dataView);
                auto nzInputDataView = FormatNZ2ND(dataView);
                slotDataViewDict_[slot] = nzInputDataView;
                UpdateInputDataViewList(inputIndex, nzInputDataView);
            }
        }

        DumpBegin();
        TimeStamp ts;
        ExecuteControlFlow(entry_, *execution);

        std::vector<std::shared_ptr<LogicalTensor>> empty(goldenDataViewList.size(), nullptr);
        TimeStamp ts1;
        DumpTensorList("Golden", &empty, &goldenDataViewList);
        dumpTensorUsage += ts1.Duration();
        totalTimeUsage += ts.Duration();

        DumpEnd();
        return execution;
    }

    std::shared_ptr<FunctionCaptureExecution> RunForPass(
            const std::string &funcKey,
            Function *func,
            const std::shared_ptr<FunctionCaptureExecution> &capture) {
        execDumpFuncKey = funcKey;

        DumpBegin();
        TimeStamp ts;
        std::shared_ptr<FunctionCaptureExecution> unitCapture = ExecuteUnit(func, capture);
        DumpEnd();
        TimeStamp ts1;
        DumpPassTensorDiff(unitCapture, capture);
        dumpTensorUsage += ts1.Duration();
        totalTimeUsage += ts.Duration();
        return unitCapture;
    }

    std::shared_ptr<FunctionCaptureExecution> RunForExecuteGraph(
            const std::string &funcKey,
            Function *func,
            const std::shared_ptr<FunctionCaptureExecution> &capture) {
        execDumpFuncKey = funcKey;

        DumpBegin();
        TimeStamp ts;
        std::shared_ptr<FunctionCaptureExecution> unitCapture = ExecuteUnit(func, capture);
        totalTimeUsage += ts.Duration();
        DumpEnd();
        return unitCapture;
    }
};

} // namespace npu::tile_fwk