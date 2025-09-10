/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
#include <unordered_set>

namespace npu::tile_fwk {

void SourceLocation::Init(std::vector<std::shared_ptr<SourceLocation>> locs) {
    std::unordered_set<uint64_t> pcSet;

    for (auto &loc : locs) {
        if (loc == nullptr || loc->lineno_ != -1)
            continue;
        pcSet.insert(loc->pc_);
    }

    Dl_info dlinfo;
    std::map<std::pair<std::string, uint64_t>, std::vector<uint64_t>> locMap;
    for (auto pc : pcSet) {
        if (dladdr((void *)pc, &dlinfo) != 0) {
            locMap[{dlinfo.dli_fname, (uint64_t)dlinfo.dli_fbase}].push_back(pc);
        }
    }

    size_t n = 0;
    char *line = nullptr;
    std::map<uint64_t, std::pair<std::string, uint64_t>> pcMap;
    for (auto &[info, pcs] : locMap) {
        std::stringstream ss;
        ss << "addr2line -i -p -e " << info.first << " " << std::hex;
        for (auto pc : pcs)
            ss << " " << pc - (intptr_t)info.second;
        auto fp = popen(ss.str().c_str(), "r");
        if (fp == nullptr) {
            continue;
        }
        for (auto pc : pcs) {
            int rc = getline(&line, &n, fp);
            if (rc >= 0 && strstr(line, "inlined by")) {
                rc = getline(&line, &n, fp);
            }
            if (rc >= 0) {
                char *p = line;
                char *name = strsep(&p, ":");
                if (auto base = strrchr(name, '/'); base != nullptr) {
                    name = base + 1;
                }
                pcMap[pc] = {name, atoi(p)};
            }
        }
        pclose(fp);

        for (auto &loc : locs) {
            if (loc == nullptr || loc->lineno_ != -1)
                continue;
            loc->fname_ = pcMap[loc->pc_].first;
            loc->lineno_ = pcMap[loc->pc_].second;
        }
    }
    free(line);
}

void SourceLocation::Init() const {
    lineno_ = 0;
    fname_ = "??";

    Dl_info dlinfo;
    if (dladdr((void *)pc_, &dlinfo) < 0)
        return;

    std::stringstream ss;
    ss << "addr2line -i -p -e " << dlinfo.dli_fname << " " << std::hex
       << pc_ - (uint64_t)dlinfo.dli_fbase;
    size_t n = 0;
    char *line = nullptr;
    auto fp = popen(ss.str().c_str(), "r");
    if (getline(&line, &n, fp) >= 0) {
        char *p = line;
        fname_ = strsep(&p, ":");
        lineno_ = atoi(p);
    }
    free(line);
    pclose(fp);
}

int SourceLocation::GetLineno() const {
    if (lineno_ == -1) {
        Init();
    }
    return lineno_;
}

std::string SourceLocation::GetFileName() const {
    if (lineno_ == -1) {
        Init();
    }
    return fname_;
}

std::stack<std::shared_ptr<SourceLocation>> SourceLocation::callStack;
} // namespace npu::tile_fwk