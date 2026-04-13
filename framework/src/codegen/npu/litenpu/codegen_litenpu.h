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

#include "tilefwk/platform.h"
#include "interface/operation/operation.h"
#include "codegen/codegen_cce.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/codegen_common.h"
#include "interface/configs/config_manager.h"
#include "codegen/npu/codegen_npu.h"

namespace npu::tile_fwk {
class CompileInfoLiteNPU : public CompileInfo {
public:
    CompileInfoLiteNPU(
        Function& topFunc, const CodeGenCtx& ctx, const std::pair<uint64_t, Function*>& subFuncPair, bool isCube,
        bool isUnderDyn)
        : CompileInfo(topFunc, ctx, subFuncPair, isCube, isUnderDyn)
    {
        std::ostringstream ss;
        ss << userSpecCCEDir_ << "/" << cceFileName_ << ".json";
        jsonAbsPath_ = ss.str();
    };
    std::string GetJsonAbsPath() const { return jsonAbsPath_; }
    void SetJsonAbsPath(const std::string& jsonAbsPath) { jsonAbsPath_ = jsonAbsPath; }

private:
    std::string jsonAbsPath_;
};

class CodeGenLiteNPU : public CodeGenCCE {
public:
    explicit CodeGenLiteNPU(const CodeGenCtx& cgCtx) : CodeGenCCE(cgCtx)
    {
        platform_ = Platform::Instance().GetSoc().GetNPUArch();
    };
    ~CodeGenLiteNPU() override = default;

    void GenCode(Function& topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>>& invokeParaOffset) override;
    std::pair<int, std::string> CompileCCE(const CompileInfoLiteNPU &compileInfo, const std::string &compileOptions) const;
    std::optional<std::string> GenExtraAlloc(const std::shared_ptr<SymbolManager> &symbolMgr, const std::shared_ptr<LogicalTensor> &tensor) const;
    std::string GenAllocForLocalBuffer(const Operation &op, const std::shared_ptr<SymbolManager> &symbolMgr) const;
    std::string GetCoreArch() const;

private:
    void GenFuncBody(Function &subFunc, Function &topFunc, std::ostringstream &oss) const;

    bool IsNeedDumpCCE(const std::string &inputFile) const;
    void DumpCCE(const std::string &name, const std::string &code) const;

    void DoCompileCCE(const CompileInfoLiteNPU &compileInfo, const std::string &compileOptions) const;
    void BuildArchOptions(std::ostringstream &oss) const;
    void BuildIncludes(std::ostringstream &oss) const;
    void BuildExtraOptions(std::ostringstream &oss, const std::string &compileOptions) const;

    std::string GetPtoTileLibPathByEnv() const;

    std::string GenAlloc(const std::shared_ptr<SymbolManager> &sm, BufferType bufferType, DataType dataType, const TileRange &range) const;

    bool IsCube(const OperationsViewer &operationList) const;
    std::string GetParamType(const Function &func) const;

    bool HandleForAICpuSubFunc(Function &subFunc);

    void UpdateSubFunc(std::pair<uint64_t, Function *> subFuncPair, const CompileInfoLiteNPU &compileInfo) const;

    bool isUnderDynamicFunction_{false};

    std::string GetIncludePathForCompileCCE() const;

    void PrintOperand(const std::string &operIO, std::shared_ptr<LogicalTensor> operand) const;

    bool HasAllocAttr(const std::shared_ptr<LogicalTensor> &tensor) const;

    std::pair<std::string, std::string> GenAllocVarName(const std::string &prefix, const TileRange &range) const;

    int CheckInjectStr(const char cmdStr[], size_t strLen) const;

    std::string GetIncludePathByRelative() const;

    std::string GetIncludePathByLib() const;

    std::string GetIncludePathByEnv() const;

    std::vector<std::string> GetInOutParams(std::pair<uint64_t, Function *> subFuncPair);

    void GenConfigJson(const std::string &jsonName, const std::string &cppName, const std::string &binName,
        const std::string &kernelName, const int &workspaceSize, const std::vector<std::string> &argNames,
        const int &blockDim) const;

    std::string GenFuncGlobalCodeAfterReplace(const Function &func, std::pair<uint64_t, Function *> subFuncPair, const std::string &subProgramCode);

    NPUArch platform_;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_CLOUDNPU_H
