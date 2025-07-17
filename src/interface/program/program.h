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
 * \file src/include/program.h
 * \brief
 */

#pragma once

#include "interface/machine/host/host_machine.h"
#include "interface/operation/distributed/comm_group_recorder.h"
#include "interface/operation/distributed/comm_barrier_manager.h"

namespace npu::tile_fwk {
class Program {
public: // public api for torch
    int HostMachineInit(const HostMachineMode mode) { return hostMachine_.Init(mode); }
    HostMachine &GetHostMachine() { return hostMachine_; }
    int EndFunction(const bool isWaitTaskFinished);
    void *Compile(); // 返回handle
    int SubmitDyndev();
    uint64_t GetWorkSpaceSize(const void *handle);
    int RunAsync(const void *stream, const void *workSpaceGmAddr, void *handle, const std::vector<void *> &opOriginArgs,
        const std::vector<size_t> &argsSize);
    TileShape tileShape;
    MatrixSize matrixSize;

    std::vector<Function *> functionSequence_;
    explicit Program(const HostMachineMode mode = HostMachineMode::SERVER);
    ~Program();

    static Program &GetInstance();
    void Reset();
    bool BeginFunction(const std::string &funcName,
        const FunctionType funcType = FunctionType::STATIC,
        const GraphType graphType = GraphType::TENSOR_GRAPH,
        const std::vector<std::reference_wrapper<Tensor>> &explicitOpArgs = {});
    std::tuple<Function *, Operation *, bool> EndFunction(const std::string &funcName, bool generateCall = true);

    Operation &ConnectCallerGusket(Function &caller, FunctionCallArgs &args) const;

    Operation &AddOperation(const std::string &opName, const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
        const std::vector<std::shared_ptr<LogicalTensor>> &oOperand);

    Operation &AddOperation(const Opcode opCode, const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
        const std::vector<std::shared_ptr<LogicalTensor>> &oOperand);

    TileShape &GetTileShape() { return tileShape; }
    MatrixSize &GetMatrixSize() { return matrixSize; }
    const TileShape &GetTileShape() const { return tileShape; }
    void SetTileShape(const TileShape& tileshape) { tileShape = tileshape; }

    bool QueryAndUpdateCurrentFunction();
    bool OperatorChecker() const { return operatorChecker_; }
    void UpdateOperatorChecker(bool checker) { operatorChecker_ = checker; }

    TuningConfig &GetConfig() { return config_; }
    const TuningConfig &GetConfig() const { return config_; }
    void SetConfig(const TuningConfig &config) { config_ = config; }

    AscendPlatformConfig &GetPlatformConfig() { return platformconfig_; }
    const AscendPlatformConfig &GetPlatformConfig() const { return platformconfig_; }
    void SetPlatFormConfig(const AscendPlatformConfig &platformconfig) { platformconfig_ = platformconfig; }

    Distributed::CommGroupRecorder &GetCommGroupRecorder() { return commGroupRecorder_; }
    Distributed::CommBarrierManager &GetCommBarrierManager() { return commBarrierManager_; }

    std::string Name() const { return name_; }
    void SetName(std::string nStr) { name_ = nStr; }

    void InsertAliveTensor(Tensor *t) { aliveTensors_.insert(t); };
    void EraseAliveTensor(Tensor *t) { aliveTensors_.erase(t); }
    void UpdateAliveTensorsParent(int outcastRawMagic, Function &parent);
    const auto &GetAliveTensors() const { return aliveTensors_; }

    const std::map<std::string, std::shared_ptr<npu::tile_fwk::Function>> &GetFunctionMap() const {
        return functionmap_;
    }
    size_t FunctionMapSize() const { return functionmap_.size(); }
    void InsertFuncToFunctionMap(const std::string &magicName, std::shared_ptr<Function> func) {
        ASSERT(functionmap_.count(magicName) == 0);
        functionmap_.emplace(magicName, func);
    }
    std::shared_ptr<Function> GetFunctionByMagic(int funcMagic);
    Function *GetFunctionByMagicName(const std::string &magicName) const;
    Function *GetFunctionByRawName(const std::string &rawName) const;
    Function *GetCurrentFunction() { return currentFunctionPtr_; }
    void SetCurrentFunction(Function *function);

    std::shared_ptr<TensorSlotManager> GetTensorSlotManager() {
        if (tensorSlotManager_ == nullptr) {
            tensorSlotManager_ = std::make_shared<TensorSlotManager>();
        }
        return tensorSlotManager_;
    };
    Json DumpJson(Function *mainFunc = nullptr) const;
    void LoadJson(Json &programJson);
    void DumpJsonFile(const std::string &fileName, Function *mainFunc = nullptr);
    std::string Dump() const; // Serialize Program briefly
    void GraphCheck() const;
    std::string DumpStack(const std::string &funcName = "") const;
    void PopStackAndUpdateCurrent();

    std::vector<std::reference_wrapper<RecordLoopFunc>> loopStack_;
    std::vector<std::reference_wrapper<RecordLoopFunc>> &GetLoopStack() { return loopStack_; }

    bool GetUnderDyndevFunction() const { return underDyndevFunction_; }
    void SetUnderDyndevFunction(bool under) {
        ASSERT(underDyndevFunction_ != under) << "Under: " << underDyndevFunction_ << " " << under;
        underDyndevFunction_ = under;
    }

    void VerifyTensorGraph();
    void VerifyPass(Function *func, int passIndex, const std::string &passIdentifier);
    void VerifyExecuteGraph();

    void SetLastFunction(Function *func) { lastFunc_ = func; }
    Function *GetLastFunction() const { return lastFunc_; }

    void SubmitAllStashTask();

private:
    std::string name_;
    HostMachine hostMachine_;
    std::vector<std::string> functionMagicNameStack_;
    std::string currentFunctionMagicName_;
    Function *currentFunctionPtr_;
    Function *lastFunc_{nullptr};
    TuningConfig config_;
    AscendPlatformConfig platformconfig_;
    bool underDyndevFunction_{false};
    bool operatorChecker_{false};
    std::unordered_set<Tensor *> aliveTensors_;
    std::map<std::string, std::shared_ptr<npu::tile_fwk::Function>> functionmap_;
    std::shared_ptr<TensorSlotManager> tensorSlotManager_;

    Distributed::CommGroupRecorder commGroupRecorder_;
    Distributed::CommBarrierManager commBarrierManager_;

    void CreateInitFunction();
    Operation *FinishCurrentFunction(const std::shared_ptr<TensorSlotScope> &scope, bool generateCall);
};
} // namespace npu::tile_fwk