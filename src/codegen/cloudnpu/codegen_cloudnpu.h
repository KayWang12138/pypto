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

#ifndef CODEGEN_CLOUDNPU_H
#define CODEGEN_CLOUDNPU_H

#include <unordered_set>
#include <utility>

#include "interface/operation/operation.h"
#include "codegen/codegen_cce.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/codegen_common.h"
#include "codegen/cloudnpu/codegen_vf.h"

namespace npu::tile_fwk {
class CompileInfo {
public:
    CompileInfo(Function &topFunc, std::string ccePath, uint64_t subProgramId, bool isCube, bool isUnderDyn)
        : ccePath_(std::move(ccePath)), isCube_(isCube), isUnderDyn_(isUnderDyn) {
        Init(topFunc, subProgramId);
    };

    std::string GetCCEFileAsHeader() const {
        bool isCompileByMachine = ConfigManager::Instance().GetCodeGenConfig(KEY_COMPILE_CCE_BY_MACHINE, false);
        return isCompileByMachine && isUnderDyn_ ? cceFileName_ + ".h" : "";
    }
    std::string GetVFHeaderAbsPath() const { return vfHeaderAbsPath_; }
    std::string GetCCEAbsPath() const { return cceAbsPath_; }
    void SetCCEAbsPath(const std::string &cceAbsPath) { cceAbsPath_ = cceAbsPath; }

    std::string GetBinAbsPath() const { return binAbsPath_; }
    void SetBinAbsPath(const std::string &binAbsPath) { binAbsPath_ = binAbsPath; }

    bool IsNeedCompileCCE() const {
        bool isCompileByMachine = ConfigManager::Instance().GetCodeGenConfig(KEY_COMPILE_CCE_BY_MACHINE, false);
        if (isUnderDyn_ && isCompileByMachine) {
            return false;
        }
        return true;
    }

    bool IsCube() const { return isCube_; }

private:
    void Init(Function &topFunc, uint64_t subProgramId) {
        std::string coreType = isCube_ ? "aic" : "aiv";
        std::stringstream ss;
        ss << topFunc.GetMagicName() << "_" << topFunc.GetFunctionHash() << "_" << subProgramId << "_" << coreType
           << "_rankId_" << npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId();
        cceFileName_ = ss.str();
        ss.str("");
        ss << ccePath_ << "/" << cceFileName_ << GetSuffix();
        cceAbsPath_ = ss.str();
        ss.str("");
        ss << ccePath_ << "/" << cceFileName_ << ".o";
        binAbsPath_ = ss.str();
        ss.str("");
        ss << ccePath_ << "/" << cceFileName_ << "_vf.h";
        vfHeaderAbsPath_ = ss.str();
    }
    std::string GetSuffix() const {
        bool isCompileByMachine = ConfigManager::Instance().GetCodeGenConfig(KEY_COMPILE_CCE_BY_MACHINE, false);
        std::string suffix = isCompileByMachine && isUnderDyn_ ? ".h" : ".cpp";
        return suffix;
    }

    std::string ccePath_;
    bool isCube_{false};
    bool isUnderDyn_{false};
    std::string cceFileName_;
    std::string cceAbsPath_;
    std::string binAbsPath_;
    std::string vfHeaderAbsPath_;
};

class CodeGenCloudNPU : public CodeGenCCE {
public:
    explicit CodeGenCloudNPU(const CodeGenCtx &cctx) : CodeGenCCE(cctx){};
    ~CodeGenCloudNPU() override = default;

    void GenCode(Function &topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) override;
    void GenCode(
        const std::string &jsonPath, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) override;
    int CompileCCE(const CompileInfo &compileInfo, const std::string &compileOptions) const;
    std::optional<std::string> GenExtraAlloc(SymbolManager &memAlloc, const std::shared_ptr<LogicalTensor> &tensor,
        const npu::tile_fwk::Operation &op) const;
    std::string GenAllocForLocalBuffer(const Operation &op, SymbolManager &memAlloc) const;

private:
    std::string GenFuncBodyBefore(
        const std::pair<uint64_t, Function *> &subFuncPair, Function &topFunc, const VFCodegen &vfCg) const;
    std::string GenInclude(const VFCodegen &vfCg) const;
    static std::string GenCommentBeforeFuncHeader(Function &subFunc);
    std::string GenFuncHeader(uint64_t programId, Function &topFunc) const;
    std::string GenFuncBody(Function &subFunc, Function &topFunc) const;
    static std::string GenFuncEnd();
    static std::string GenKernelName(Function &topFunc, uint64_t programId);

    bool IsNeedDumpCCE(const std::string &inputFile) const;
    void DumpCCE(const std::string &name, const std::string &code) const;

    void DoCompileCCE(const CompileInfo &compileInfo, const std::string &compileOptions) const;

    std::string GenAlloc(SymbolManager &manager, SymbolManager::BufferType bufferType, npu::tile_fwk::DataType dataType,
        const npu::tile_fwk::TileRange &range) const;

    bool IsCube(const OperationsViewer &operationList) const;
    std::string GetParamType(const Function &func) const;

    std::string GenDynParamForExpr(const npu::tile_fwk::Function &func) const;

    bool HandleForAICpuSubFunc(Function &subFunc);

    void UpdateSubFunc(
        Function &topFunc, std::pair<uint64_t, Function *> subFuncPair, const CompileInfo &compileInfo) const;

    bool isUnderDynamicFunction_{false};
};

} // namespace npu::tile_fwk

#endif // CODEGEN_CLOUDNPU_H
