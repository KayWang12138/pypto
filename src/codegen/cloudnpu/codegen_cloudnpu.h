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
#include "codegen/codegen_symbol.h"
#include "codegen/codegen_common.h"

namespace npu::tile_fwk {
class CodeGenCloudNPU : public CodeGenCCE {
public:
    explicit CodeGenCloudNPU(const CodeGenCtx &cctx) : CodeGenCCE(cctx){};
    ~CodeGenCloudNPU() override = default;

    void GenCode(
        Function &topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) override;
    void GenCode(
        const std::string &jsonPath, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) override;
    int CompileCCE(
        const std::string &srcFile, const std::string &objFile, bool isCube, const std::string &compileOptions) const;
    std::optional<std::string> GenExtraAlloc(
        SymbolManager &memAlloc, const std::shared_ptr<LogicalTensor> &tensor, const npu::tile_fwk::Operation &op) const;

private:
    std::string GenCodeImpl(Function &subFunc, Function &topFunc);
    std::string GenAllocForLocalBuffer(Function &topFunc, const Operation &op, SymbolManager &memAlloc) const;

    bool DumpCCE(const std::string &name, const std::string &code) const;
    bool GenConfigJson(const std::string &configJson, const std::string &cppName, const std::string &binName,
        const std::string &kernelName, int workspaceSize) const;

    std::string GenAlloc(SymbolManager &manager, SymbolManager::BufferType bufferType, npu::tile_fwk::DataType dataType,
        const npu::tile_fwk::TileRange &range) const;

    std::string GenFuncCodeAfterReplace(const Function &func, std::pair<uint64_t, Function *> subFuncPair,
        const std::string &subProgramCode);

    bool IsCube(const OperationsViewer &operationList) const;
    std::string GetParamType(const Function &func);

    std::string GenDynParamForExpr(const npu::tile_fwk::Function &func) const;

    std::vector<std::pair<DataType, std::string>> globalTensorAddr;
    bool isUnderDynamicFunction{false};
};


} // namespace npu::tile_fwk

#endif // CODEGEN_CLOUDNPU_H
