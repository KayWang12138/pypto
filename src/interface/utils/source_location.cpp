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

namespace npu::tile_fwk {

void SourceLocation::Init(std::vector<std::shared_ptr<SourceLocation>> locs) {
    Dl_info dlinfo;
    std::map<std::pair<std::string, uint64_t>, std::vector<int>> locMap;
    for (size_t i = 0; i < locs.size(); i++) {
        if (locs[i] == nullptr || locs[i]->lineno_ != -1)
            continue;
        if (dladdr((void *)locs[i]->pc_, &dlinfo) < 0) {
            locs[i]->lineno_ = 0;
        } else {
            locMap[{dlinfo.dli_fname, (uint64_t)dlinfo.dli_fbase}].push_back(i);
        }
    }

    size_t n = 0;
    char *line = nullptr;
    for (auto &[info, idxs] : locMap) {
        std::stringstream ss;
        ss << "addr2line -i -p -e " << info.first << " " << std::hex;
        for (auto idx : idxs)
            ss << " " << locs[idx]->pc_ - (intptr_t)info.second;
        std::cout << ss.str() << std::endl;
        auto fp = popen(ss.str().c_str(), "r");
        for (auto idx : idxs) {
            locs[idx]->lineno_ = 0;
            int rc = getline(&line, &n, fp);
            if (rc >= 0 && strstr(line, "inlined by"))
                rc = getline(&line, &n, fp);
            if (rc >= 0) {
                char *p = line;
                locs[idx]->fname_ = strsep(&p, ":");
                locs[idx]->lineno_ = atoi(p);
            }
        }
        pclose(fp);
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