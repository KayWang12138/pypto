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
 * \file codegen.h
 * \brief
 */

#ifndef CODEGEN_LITENPU_H
#define CODEGEN_LITENPU_H

#include <unordered_set>
#include <utility>

#include "interface/operation/operation.h"
#include "codegen/codegen_cce.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/codegen_common.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
class CompileInfo_LiteNPU {
public:
    CompileInfo_LiteNPU(Function &topFunc, std::string cceDir, uint64_t subProgramId, bool isCube, bool isUnderDyn)
        : userSpecCCEDir_(std::move(cceDir)), isCube_(isCube), isUnderDyn_(isUnderDyn) {
        Init(topFunc, subProgramId);
    };
    std::string GetVFHeaderAbsPath() const { return vfHeaderAbsPath_; }
    std::string GetCCEAbsPath() const { return cceAbsPath_; }
    void SetCCEAbsPath(const std::string &cceAbsPath) { cceAbsPath_ = cceAbsPath; }

    std::string GetBinAbsPath() const { return binAbsPath_; }
    void SetBinAbsPath(const std::string &binAbsPath) { binAbsPath_ = binAbsPath; }
    static bool IsNeedCompileCCE() {
        // bool isNeedCompile = ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, true);
        bool isNeedCompile = true; // TODO...
        return isNeedCompile;
    }

    void SetKernelName(const std::string &kernelName) { kernelName_ = kernelName; }
    std::string GetKernelName() const { return kernelName_; }
    void SetFuncDeclare(const std::string &funcDeclare) { funcDeclare_ = funcDeclare; }
    std::string GetFuncDeclare() const { return funcDeclare_; }
    bool IsCube() const { return isCube_; }

private:
    void Init(Function &topFunc, uint64_t subProgramId) {
        std::string coreType = isCube_ ? "aic" : "aiv";
        std::ostringstream ss;
        (void)topFunc; // TODO: use below function...
        (void)subProgramId;
        // ss << topFunc.GetMagicName() << "_" << topFunc.GetFunctionHash() << "_" << subProgramId << "_" << coreType
        //    << "_rankId_" << npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId();
        cceFileName_ = ss.str();
        ss.str("");
        ss << userSpecCCEDir_ << "/" << cceFileName_ << GetSuffix();
        cceAbsPath_ = ss.str();
        ss.str("");
        ss << userSpecCCEDir_ << "/" << cceFileName_ << ".o";
        binAbsPath_ = ss.str();
        ss.str("");
        ss << userSpecCCEDir_ << "/" << cceFileName_ << "_vf.h";
        vfHeaderAbsPath_ = ss.str();
    }
    std::string GetSuffix() const {
        std::string suffix = ".cpp";
        return suffix;
    }

    std::string userSpecCCEDir_;
    bool isCube_{false};
    bool isUnderDyn_{false};
    std::string cceFileName_;
    std::string cceAbsPath_;
    std::string binAbsPath_;
    std::string kernelName_;
    std::string funcDeclare_;
    std::string vfHeaderAbsPath_;
};

class CodeGenLiteNPU : public CodeGenCCE {
public:
    explicit CodeGenLiteNPU(const CodeGenCtx &cctx) : CodeGenCCE(cctx) {};
    ~CodeGenLiteNPU() override = default;

    void GenCode(Function &topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) override;
    void GenCode(
        const std::string &jsonPath, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset);
    int CompileCCE(const CompileInfo_LiteNPU &compileInfo, const std::string &compileOptions) const;
    std::optional<std::string> GenExtraAlloc(SymbolManager &memAlloc, const std::shared_ptr<LogicalTensor> &tensor) const;
    std::string GenAllocForLocalBuffer(const Operation &op, SymbolManager &memAlloc) const;

private:
    std::string GenFuncBodyBefore(const std::pair<uint64_t, Function *> &subFuncPair, Function &topFunc,
                                  CompileInfo_LiteNPU &compileInfo) const;
    std::string GenInclude(const Function &topFunc) const;
    static std::string GenCommentBeforeFuncHeader(Function &subFunc);
    std::string GenFuncHeader(uint64_t programId, Function &topFunc, CompileInfo_LiteNPU &compileInfo) const;
    std::string GenFuncBody(Function &subFunc, Function &topFunc) const;
    static std::string GenFuncEnd();
    static std::string GenKernelName(Function &topFunc, uint64_t programId);

    bool IsNeedDumpCCE(const std::string &inputFile) const;
    void DumpCCE(const std::string &name, const std::string &code) const;

    void DoCompileCCE(const CompileInfo_LiteNPU &compileInfo, const std::string &compileOptions) const;

    std::string GenAlloc(SymbolManager &manager, BufferType bufferType, npu::tile_fwk::DataType dataType,
        const npu::tile_fwk::TileRange &range) const;

    bool IsCube(const OperationsViewer &operationList) const;
    std::string GetParamType(const Function &func) const;

    std::string GenDynParamForExpr(const npu::tile_fwk::Function &func) const;

    bool HandleForAICpuSubFunc(Function &subFunc);

    void UpdateSubFunc(std::pair<uint64_t, Function *> subFuncPair, const CompileInfo_LiteNPU &compileInfo) const;

    bool isUnderDynamicFunction_{false};

    std::string GetIncludePathForCompileCCE() const;

    void PrintOperand(const std::string &operIO, std::shared_ptr<LogicalTensor> operand) const;

    bool HasAllocAttr(const std::shared_ptr<LogicalTensor> &tensor) const;

    std::pair<std::string, std::string> GenAllocVarName(const std::string &prefix, const TileRange &range) const;

    int CheckInjectStr(const char cmdStr[], size_t strLen) const;

    std::string GetIncludePathByRelative() const;

    std::string GetIncludePathByLib() const;

    std::string GetIncludePathByEnv() const;

    std::string GenFuncGlobalCodeAfterReplace(const Function &func, std::pair<uint64_t, Function *> subFuncPair, const std::string &subProgramCode);
};

} // namespace npu::tile_fwk

#endif // CODEGEN_CLOUDNPU_H
