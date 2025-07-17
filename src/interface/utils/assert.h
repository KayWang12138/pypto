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
 * \file assert.h
 * \brief
 */

#pragma once

#include <string>
#include <iostream>
#include <cassert>

namespace npu::tile_fwk {

#define ASSERT(condition) \
    if (!(condition))     \
    npu::tile_fwk::AssertInfo(#condition, __FILE__, __LINE__), std::cout << "\033[33m"

class AssertInfo {
public:
    AssertInfo(std::string messageIn, std::string fileIn, int lineIn)
        : message(std::move(messageIn)), file(std::move(fileIn)), line(lineIn) {}

    ~AssertInfo() {
        std::cout << "\033[0m" << std::endl
                  << "\033[31m"
                  << "Assertion failed: " << message << ", file " << file << ", line " << line << "\033[0m"
                  << std::endl;
        assert(false);
    }

private:
    std::string message;
    std::string file;
    int line;
};
}