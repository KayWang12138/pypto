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
 * \file cann_host_runtime.cpp
 * \brief
 */

#define STRINGIFY(x) #x
#include "tilefwk/cann_host_runtime.h"

namespace npu {
namespace tile_fwk {
const uint32_t kMaxLength = 50;
const std::string version = "version";
#ifdef BUILD_WITH_CANN
void *GetSymbol(const std::string &sym) {
    void *ptr = nullptr;
    const char* CannPath = STRINGIFY(ASCEND_CANN_PACKAGE_PATH);
    std::string LibPathDir = std::string(CannPath) + "/lib64/libruntime.so";
    std::string soPath = RealPath(LibPathDir);
    void* handle = dlopen(soPath.c_str(), RTLD_LAZY);
    if (handle != nullptr) {
        ptr = dlsym(handle, sym.c_str());
    }
    return ptr;
}
#endif

bool GetSocVersion(std::string& socVersion) {
    int ret = 1;
    char socVer[kMaxLength] = {0x00};
#ifdef BUILD_WITH_CANN
    using GetSocVerFunc = int (*)(char *, const uint32_t);
    std::string socVerFuncName = "rtGetSocVersion";
    auto socVerFunc = (GetSocVerFunc)GetSymbol(socVerFuncName);
    ret = socVerFunc(socVer, kMaxLength);
#endif
    if (ret == 0) {
        socVersion = std::string(socVer);
        return true;
    }
    (void)socVersion;
    return false;
}
}  // namespace tile_fwk
}  // namespace npu