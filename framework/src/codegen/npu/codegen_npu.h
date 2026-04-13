/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file codegen_npu.h
 * \brief
 */

#ifndef CODEGEN_NPU_H
#define CODEGEN_NPU_H

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

namespace npu::tile_fwk {
struct CompileTaskInfo {
    std::string outputPath;
    std::string inputPath;
    std::string compileCmd;
};

class CompileInfo {
public:
    CompileInfo(
        Function& topFunc, const CodeGenCtx& ctx, const std::pair<uint64_t, Function*>& subFuncPair, bool isCube,
        bool isUnderDyn)
        : userSpecCCEDir_(ctx.cceDir),
          isCube_(isCube),
          isUnderDyn_(isUnderDyn),
          attr_(subFuncPair.second->GetLeafFuncAttribute()),
          isMainBlock_(ctx.isMainBlock)
    {
        Init(topFunc, subFuncPair.first);
    };
    std::string GetCCEAbsPath() const { return cceAbsPath_; }
    void SetCCEAbsPath(const std::string& cceAbsPath) { cceAbsPath_ = cceAbsPath; }

    std::string GetBinAbsPath() const { return binAbsPath_; }
    void SetBinAbsPath(const std::string& binAbsPath) { binAbsPath_ = binAbsPath; }
    void SetKernelName(const std::string& kernelName) { kernelName_ = kernelName; }
    std::string GetKernelName() const { return kernelName_; }
    void SetFuncDeclare(const std::string& funcDeclare) { funcDeclare_ = funcDeclare; }
    std::string GetFuncDeclare() const { return funcDeclare_; }
    bool IsCube() const { return isCube_; }
    bool isUnderDyn() const { return isUnderDyn_; }

private:
    void Init(Function& topFunc, uint64_t subProgramId)
    {
        std::string coreType = isCube_ ? "aic" : "aiv";
        std::ostringstream ss;
        std::ostringstream tailStr;

        if ((attr_ != nullptr) && (attr_->mixId != -1)) {
            tailStr << "mix" << attr_->mixId << "_" << coreType;
            if (!isCube_) {
                int aivId = static_cast<int>(attr_->aivCore);
                if (aivId == -1) {
                    tailStr << "x";
                } else {
                    tailStr << aivId;
                }
            }
        } else {
            tailStr << coreType;
        }

        if (isMainBlock_) {
            tailStr << "_main";
        }

        ss << topFunc.GetMagicName() << "_" << topFunc.GetFunctionHash() << "_" << subProgramId << "_" << tailStr.str();
        cceFileName_ = ss.str();
        ss.str("");
        ss << userSpecCCEDir_ << "/" << cceFileName_ << GetSuffix();
        cceAbsPath_ = ss.str();
        ss.str("");
        ss << userSpecCCEDir_ << "/" << cceFileName_ << ".o";
        binAbsPath_ = ss.str();
    }
    std::string GetSuffix() const
    {
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
    std::shared_ptr<LeafFuncAttribute> attr_{nullptr};
    bool isMainBlock_{false};
};

class CodeGenNPU : public CodeGenCCE {
public:
    explicit CodeGenNPU(const CodeGenCtx& cgCtx) : CodeGenCCE(cgCtx)
    {
        platform_ = Platform::Instance().GetSoc().GetNPUArch();
    };
    ~CodeGenNPU() override = default;

protected:
    NPUArch platform_;
};

class FloatSpecValMgr {
public:
    void UpdateByOp(const Operation& op);
    void PrintFloatSpecVal(std::ostringstream& oss);

private:
    std::set<FloatSpecVal> floatSpecVals_;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_CLOUDNPU_H
