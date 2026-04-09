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
 * \file source_location.cpp
 * \brief
 */

#include "source_location.h"
#include "interface/utils/common.h"

#include <dlfcn.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <map>
#include <memory>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_set>

namespace npu::tile_fwk {

void SourceLocation::Init() const
{
    std::lock_guard<std::mutex> lock(mutex);
    if (pcSet.empty()) {
        return;
    }

    Dl_info dlinfo;
    std::map<std::pair<std::string, uint64_t>, std::vector<uint64_t>> dlMap;
    for (auto pc : pcSet) {
        if (dladdr((void*)pc, &dlinfo) != 0) {
            dlMap[{dlinfo.dli_fname, (uint64_t)dlinfo.dli_fbase}].push_back(pc);
        }
    }
    pcSet.clear();

    for (auto& [info, pcs] : dlMap) {
        std::vector<std::string> args = {"addr2line", "-i", "-p", "-e", info.first};
        for (auto pc : pcs) {
            std::stringstream ss;
            ss << std::hex << (pc - (intptr_t)info.second);
            args.push_back(ss.str());
        }

        std::string output = SafeExecCommandWithOutput(args);
        std::istringstream iss(output);
        std::string lineStr;

        for (auto pc : pcs) {
            bool parsed = false;
            if (std::getline(iss, lineStr)) {
                if (lineStr.find("inlined by") != std::string::npos) {
                    if (!std::getline(iss, lineStr)) {
                        lineStr.clear();
                    }
                }
                if (!lineStr.empty()) {
                    size_t colonPos = lineStr.find(":");
                    if (colonPos != std::string::npos) {
                        locMap[pc]->fname_ = lineStr.substr(0, colonPos);
                        long lineno = strtol(lineStr.substr(colonPos + 1).c_str(), nullptr, 10);
                        locMap[pc]->lineno_ = static_cast<int>(lineno);
                        parsed = true;
                    } else {
                        locMap[pc]->fname_ = lineStr;
                        locMap[pc]->lineno_ = 0;
                        parsed = true;
                    }
                }
            }

            if (!parsed) {
                std::stringstream os;
                // addr2line failed, use elfname + offset
                os << info.first << "(+" << std::hex << pc - (intptr_t)info.second << ")";
                locMap[pc]->fname_ = os.str();
                locMap[pc]->lineno_ = 0;
            }
        }
    }
}

int SourceLocation::GetLineno() const
{
    Init();
    return lineno_;
}

const std::string& SourceLocation::GetFileName() const
{
    Init();
    return fname_;
}

const std::string& SourceLocation::GetBacktrace() const { return backtrace_; }

bool SourceLocation::isCppMode_ = false;
std::mutex SourceLocation::mutex;
std::stack<std::shared_ptr<SourceLocation>> SourceLocation::callStack;
std::unordered_set<uint64_t> SourceLocation::pcSet;
std::unordered_map<uint64_t, std::shared_ptr<SourceLocation>> SourceLocation::locMap;
} // namespace npu::tile_fwk
