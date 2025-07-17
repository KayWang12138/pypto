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
 * \file codegen.cpp
 * \brief
 */

#include <unistd.h>
#include <sys/stat.h>

#include "codegen_cce.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
void CodeGenCCE::PrepareDefaultOutputPath() {
    constexpr int size = 1024;
    char buf[size] = {};
    std::string cwd = getcwd(buf, size);
    if (cwd.empty()) {
        ALOG_INFO << "failed to call getcwd()!";
        return;
    }

    std::string outputPath = cwd + "/kernel_meta";

    struct stat st = {};
    bool outPathExist = stat(outputPath.c_str(), &st) == 0;
    if (outPathExist) {
        ASSERT(S_ISDIR(st.st_mode)) << outputPath << " is not a directory!";
    } else {
        ASSERT(mkdir(outputPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0)
            << "failed to call mkdir to create " << outputPath;
    }

    path_ = outputPath;
}

std::map<int, int> GenRealizeIdMap(const SubfuncParam &subFuncParam) {
    auto &tensorInvokeArgs = subFuncParam.tensorsArgs_;
    auto &incastInvokeArgs = subFuncParam.inCastArgs_;
    auto &outcastInvokeArgs = subFuncParam.outCastArgs_;

    std::map<int, int> idMap;
    auto f = [&idMap](size_t offset, auto &invokeArgs) {
        ALOG_INFO << " start offset is " << offset << ", arg size is " << invokeArgs.size();
        for (size_t i = 0; i < invokeArgs.size(); i++) {
            size_t paramOff = (offset + i);
            uint32_t paramLoc = invokeArgs[i].paramLoc;
            ALOG_DEBUG("paramLoc ", paramLoc, " --> offset ", paramOff);
            ALOG_INFO << " paramLoc is " << paramLoc << ", paramOff is " << paramOff << ", SymDDRId is "
                      << invokeArgs[i].symDDRId << ", SymName is " << invokeArgs[i].symName;
            idMap.insert({paramLoc, paramOff});
        }
    };

    ALOG_INFO << "---  start tensorInvokeArgs paramLoc map ---- ";
    f(0, tensorInvokeArgs);
    ALOG_INFO << "---  start incastInvokeArgs paramLoc map ---- ";
    f(tensorInvokeArgs.size(), incastInvokeArgs);
    ALOG_INFO << "---  start outcastInvokeArgs paramLoc map ---- ";
    f(tensorInvokeArgs.size() + incastInvokeArgs.size(), outcastInvokeArgs);
    return idMap;
}
} // namespace npu::tile_fwk
