/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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

#include <string>
#include <unordered_set>
#include <utility>
#include <mutex>
#include <vector>
#include <chrono>
#include <thread>

#include "tilefwk/platform.h"
#include "interface/operation/operation.h"
#include "codegen/codegen_cce.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/codegen_common.h"
#include "interface/configs/config_manager.h"
#include "codegen/npu/codegen_npu.h"

namespace npu::tile_fwk {

class CodeGenCloudNPU : public CodeGenNPU {
public:
    explicit CodeGenCloudNPU(const CodeGenCtx& cgCtx) : CodeGenNPU(cgCtx) {};
    ~CodeGenCloudNPU() override = default;

    void GenCode(Function& topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>>& invokeParaOffset) override;
    std::string PrepareCmd(const CompileInfo& compileInfo, const std::string& compileOptions) const;
    // only used to compile code directly when running under simulation mode.
    void CompileCode(const std::string& compileCmd) const;
    std::optional<std::string> GenExtraAlloc(
        const std::shared_ptr<SymbolManager>& sm, const std::shared_ptr<LogicalTensor>& tensor) const;
    std::string GenAllocForLocalBuffer(const Operation& op, const std::shared_ptr<SymbolManager>& sm) const;
    std::string GetCoreArch(const CompileInfo& compileInfo) const;
    static void AppendVFOptions(NPUArch platform, std::ostringstream& oss);

private:
    void GenFuncBodyBefore(
        const std::pair<uint64_t, Function*>& subFuncPair, Function& topFunc, CompileInfo& compileInfo,
        std::ostringstream& oss) const;
    void GenInclude(const Function& topFunc, std::ostringstream& oss) const;
    void GenCommentBeforeFuncHeader(Function& subFunc, std::ostringstream& oss) const;
    std::string GenFuncHeader(uint64_t programId, Function& topFunc, CompileInfo& compileInfo) const;
    void GenFuncBody(Function& subFunc, Function& topFunc, std::ostringstream& oss) const;
    void GenFuncEnd(std::ostringstream& oss) const;
    static std::string GenKernelName(Function& topFunc, uint64_t programId);

    void GenCodeToBinaryTask(
        std::ostringstream& code, const CompileInfo& compileInfo, const std::string& compileOptions) const;
    bool IsNeedDumpCode(const std::string& inputFile) const;
    void DumpCode(const std::string& name, std::ostringstream& code) const;
    int DoCompileCmd(const std::string& compileCmd) const;

    void BuildArchOptions(std::ostringstream& oss, const CompileInfo& compileInfo) const;
    void BuildIncludes(std::ostringstream& oss) const;
    void BuildExtraOptions(std::ostringstream& oss, const std::string& compileOptions) const;

    std::string GenAlloc(
        const std::shared_ptr<SymbolManager>& manager, BufferType bufferType, DataType dataType,
        const TileRange& range) const;

    std::string GetParamType(const Function& func, bool isUnderDynFunc) const;

    std::string GenDynParamForExpr(const Function& func) const;

    bool HandleForAICpuSubFunc(Function& subFunc);

    void UpdateSubFunc(std::pair<uint64_t, Function*> subFuncPair, const CompileInfo& compileInfo) const;

    std::string GetIncludePathForCompileCCE() const;
    std::string GetPtoTileLibPathByEnv() const;

    void CollectCompileTask(const CompileTaskInfo& task) const;
    void GenerateMakefile(const std::string& makefilePath) const;
    void ExecuteParallelCompile(const Function& topFunc);
    std::string GetOutputDir() const;

    mutable std::mutex compileTasksMutex_;
    mutable std::vector<CompileTaskInfo> compileTasks_;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_CLOUDNPU_H
