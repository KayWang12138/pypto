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
 * \file pass.cpp
 * \brief
 */

#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/utils/file_utils.h"

static constexpr size_t PASS_NUM_DIGITS = 2;
namespace npu::tile_fwk {
Pass::Pass(std::string name) : name_(std::move(name)) {}

const std::string &Pass::LogFolder(const std::string &topFolder, size_t i) const {
    if (CreateLogFolder(topFolder, i) == FAILED) {
        ALOG_WARN_F("Create log folder failed.");
        passFolder_ = topFolder;
    }
    return passFolder_;
}

Status Pass::CreateLogFolder(const std::string &topFolder, size_t i) const {
    if (passFolder_.empty()) {
        std::stringstream ss;
        ss << std::setw(PASS_NUM_DIGITS) << std::setfill('0') << i;
        passFolder_ = topFolder + "/Pass_" + ss.str() + "_" + name_;
        bool res = CreateDir(passFolder_);
        if (res == false) {
            ALOG_WARN_F("Failed to create directory: [%s].", passFolder_.c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Pass::Run(Function &function, const std::string &strategy,
                 const std::string &identifier, size_t runtimeIdx) {
    identifier_ = identifier;
    strategy_ = strategy;
    passRuntimeIndex_ = runtimeIdx;
    if (passDfxconfigs_.disablePass) {
        ALOG_WARN_F("Pass [%s] is skipped.", identifier_.c_str());
        return SUCCESS;
    }
    if (PreRun(function) == FAILED) {
        ALOG_ERROR_F("PreRun pass [%s] failed.", identifier_.c_str());
        return FAILED;
    } 
    if (RunOnFunction(function) == FAILED) {
        ALOG_ERROR_F("Run pass [%s] failed.", identifier_.c_str());
        return FAILED;
    }
    if (PostRun(function) == FAILED) {
        ALOG_ERROR_F("PostRun pass [%s] failed.", identifier_.c_str());
        return FAILED;
    }
    identifier_.clear();
    strategy_.clear();
    return SUCCESS;
}

Status Pass::PrintFunction(Function& function, const std::string &logFolder, bool beforeFunction = true) {
    constexpr int printWide = 3;
    constexpr int funcPrintWide = 2;
    const auto &filePrefix = identifier_ + "_" + function.GetMagicName();
    std::string stageName = beforeFunction ? "Before" : "After";
    ALOG_INFO_F("Dump function %s pass [%s].", stageName.c_str(), identifier_.c_str());
    if (function.rootFunc_ != nullptr) {
        std::stringstream ssRoot;
        ssRoot << stageName << "_" << std::setw(printWide) << std::setfill('0') << passRuntimeIndex_ << "_" << filePrefix << "_Root_.tifwkgr";
        std::ofstream file(logFolder + "/" + ssRoot.str());
        if (file.is_open()) {
            file << function.rootFunc_->Dump();
            file.close();
        }
        std::stringstream ss;
        for (auto &subProgram : function.rootFunc_->programs_) {
            ss.str("");
            ss << stageName << "_" << std::setw(printWide) << std::setfill('0') << passRuntimeIndex_ << "_" << filePrefix
                << "_LEAF_program_id_" << std::setw(funcPrintWide) << std::setfill('0') << subProgram.first << "_"
                << subProgram.second->GetFunctionHash().GetHash() << ".tifwkgr";
            std::ofstream subFile(logFolder + "/" + ss.str());
            if (subFile.is_open()) {
                subFile << subProgram.second->Dump();
                subFile.close();
            }
        }
    }
    {
        std::stringstream ssInner;
        ssInner << stageName << "_" << std::setw(printWide) << std::setfill('0') << passRuntimeIndex_ << "_" << filePrefix << ".tifwkgr";
        std::ofstream file(logFolder + "/" + ssInner.str());
        if (file.is_open()) {
            file << function.Dump();
            file.close();
        }
    }
    return SUCCESS;
}

Status Pass::DumpFunctionJson(Function& function, const std::string &logFolder, bool beforeFunction = true) {
    constexpr int printWide = 3;
    constexpr int funcPrintWide = 2;
    const auto &filePrefix = identifier_ + "_" + function.GetMagicName();
    std::string stageName = beforeFunction ? "Before" : "After";
    ALOG_INFO_F("Dump function %s pass [%s].", stageName.c_str(), identifier_.c_str());
    std::stringstream ss;
    ss << stageName << "_" << std::setw(printWide) << std::setfill('0') << passRuntimeIndex_ << "_" << filePrefix << ".json";
    function.DumpJsonFile(logFolder + "/" + ss.str());
    if (function.rootFunc_ != nullptr) {
        ss.str("");
        ss << stageName << "_" << std::setw(printWide) << std::setfill('0') << passRuntimeIndex_ << "_" << filePrefix
            << "_ROOT.json";
        function.rootFunc_->DumpJsonFile(logFolder + "/" + ss.str());
        for (auto &subProgram : function.rootFunc_->programs_) {
            ss.str("");
            ss << stageName << "_" << std::setw(printWide) << std::setfill('0') << passRuntimeIndex_ << "_" << filePrefix
                << "_LEAF_program_id_" << std::setw(funcPrintWide) << std::setfill('0') << subProgram.first << "_"
                << subProgram.second->GetFunctionHash().GetHash() << ".json";
            subProgram.second->DumpJsonFile(logFolder + "/" + ss.str());
        }
    }
    return SUCCESS;
}

Status Pass::PreRun(Function &function) {
    if (passDfxconfigs_.printFunction) {
        if (PrintFunction(function, passFolder_, true) != SUCCESS) {
            ALOG_WARN_F("Print function before pass failed.");
        }
    }
    if (passDfxconfigs_.dumpFunctionGraphBeforePass) {
        if (DumpFunctionJson(function, passFolder_, true) != SUCCESS) {
            ALOG_WARN_F("Dump function json before pass failed.");
        }
    }
    if (passDfxconfigs_.preCheck) {
        if (PreCheck(function) != SUCCESS) {
            ALOG_ERROR_F("Precheck of pass [%s] failed.", identifier_.c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Pass::PostRun(Function &function) {
    if (passDfxconfigs_.printFunction) {
        if (PrintFunction(function, passFolder_, false) != SUCCESS) {
            ALOG_WARN_F("Print function after pass failed.");
        }
    }
    if (passDfxconfigs_.dumpFunctionGraphAfterPass) {
        if (DumpFunctionJson(function, passFolder_, false) != SUCCESS) {
            ALOG_WARN_F("Dump function json after pass failed.");
        }
    }
    if (passDfxconfigs_.postCheck) {
        if (PostCheck(function) != SUCCESS) {
            ALOG_ERROR_F("Postcheck of pass [%s] failed.", identifier_.c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Pass::PreCheck(Function &function) {
    (void)function;
    return SUCCESS;
}

Status Pass::PostCheck(Function &function) {
    (void)function;
    return SUCCESS;
}
} // namespace npu::tile_fwk
