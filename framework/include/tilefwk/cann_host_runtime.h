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
 * \file cann_host_runtime.h
 * \brief
 */

#pragma once

#include <dlfcn.h>
#include <cstdint>
#include "file.h"

namespace npu {
namespace tile_fwk {
struct CannHostRuntime {
    static CannHostRuntime &GetObj() {
        static CannHostRuntime cannHostRuntime;
        return cannHostRuntime;
    }

    ~CannHostRuntime() {
        if (handle != nullptr) {
            dlclose(handle);
        }
        if (handleDep != nullptr) {
            dlclose(handleDep);
        }
    }
    bool GetSocVersion(std::string& socVersion);
    std::string GetPlatformFile(const std::string &socVersion);
private:
    CannHostRuntime();
#ifdef BUILD_WITH_CANN
    void *GetSymbol(const std::string &sym) {
        void *ptr = nullptr;
        if (handleDep != nullptr && handle != nullptr) {
            ptr = dlsym(handle, sym.c_str());
        }
        return ptr;
    }
#endif
    void *handleDep = nullptr;
    void *handle = nullptr;
};
}
}