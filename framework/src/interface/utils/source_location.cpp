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

extern "C" {
#include <backtrace.h>
}

namespace npu::tile_fwk {

// Callback data structure for backtrace_pcinfo
struct PcInfoData {
    std::string* filename;
    int* lineno;
    bool found;
};

// Error callback for libbacktrace
static void BacktraceErrorCallback(void* data, const char* msg, int errnum)
{
    (void)data;
    (void)msg;
    (void)errnum;
}

// Callback for backtrace_pcinfo to receive file/line info
static int BacktracePcinfoCallback(void* data, uintptr_t pc, const char* filename, int lineno, const char* function)
{
    (void)pc;
    (void)function;
    PcInfoData* pcdata = static_cast<PcInfoData*>(data);
    if (filename != nullptr && lineno > 0) {
        *pcdata->filename = filename;
        *pcdata->lineno = lineno;
        pcdata->found = true;
        return 1; // Stop iteration
    }
    return 0; // Continue iteration
}

// Public interface for safe address resolution using libbacktrace
bool SourceLocation::ResolveAddressSafely(void* addr, std::string& filename, int& lineno)
{
    filename = "";
    lineno = 0;

    // Get library information
    Dl_info info;
    if (dladdr(addr, &info) == 0 || info.dli_fname == nullptr) {
        return false;
    }

    // Try to use libbacktrace for symbol resolution
    static std::mutex btMutex;
    static struct backtrace_state* btState = nullptr;
    static std::string btFilename;

    {
        std::lock_guard<std::mutex> lock(btMutex);
        // Initialize or update backtrace state if needed
        if (btState == nullptr || btFilename != info.dli_fname) {
            btFilename = info.dli_fname;
            btState = backtrace_create_state(info.dli_fname, 0, BacktraceErrorCallback, nullptr);
        }
    }

    if (btState != nullptr) {
        PcInfoData pcdata{&filename, &lineno, false};
        backtrace_pcinfo(
            btState, reinterpret_cast<uintptr_t>(addr), BacktracePcinfoCallback, BacktraceErrorCallback, &pcdata);
        return pcdata.found;
    }

    return false;
}

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
        for (auto pc : pcs) {
            std::string filename;
            int lineno;
            if (ResolveAddressSafely((void*)pc, filename, lineno)) {
                locMap[pc]->fname_ = filename;
                locMap[pc]->lineno_ = lineno;
            } else {
                // Fallback: use elfname + offset
                std::stringstream os;
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
