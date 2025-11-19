/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file error.cpp
 * \brief
 */

#include "tilefwk/error.h"

#include <execinfo.h>
#include <cstring>
#include <sstream>
#include <functional>
#include <cxxabi.h>
#include <securec.h>

#include "interface/utils/string_utils.h"

namespace npu::tile_fwk {

class BacktraceImpl : public LazyValue<std::string> {
public:
    __always_inline BacktraceImpl(size_t skipFrames, size_t maxFrames) : callStack_(maxFrames, 0) {
        skipFrames += 1;
        auto nrFrames = static_cast<size_t>(::backtrace(callStack_.data(), static_cast<int>(callStack_.size())));
        skipFrames = std::min(skipFrames, nrFrames);
        callStack_.erase(callStack_.begin(), callStack_.begin() + static_cast<ssize_t>(skipFrames));
        callStack_.resize(nrFrames - skipFrames);
    }

    void ParseFrame(std::stringstream &ss, char *line) const {
        auto funcName = strstr(line, "(");
        auto funcOffset = strstr(line, "+");
        if (funcName == nullptr || funcOffset == nullptr) {
            ss << line << '\n';
            return;
        }

        *funcName++ = '\0';
        *funcOffset++ = '\0';
        int status = 0;
        std::unique_ptr<char, std::function<void(char *)>> demangled(
            abi::__cxa_demangle(funcName, nullptr, nullptr, &status),
            /* deleter */ free);
        if (status == 0)
            funcName = demangled.get();
        ss << line << '(' << funcName << '+' << funcOffset << '\n';
    }

    const std::string &Get() const {
        return symbols_.Ensure([this]() -> std::string {
            auto strings = backtrace_symbols(callStack_.data(), callStack_.size());
            if (strings == nullptr) {
                return "Backtrace Failed";
            }
            std::stringstream ss;
            for (size_t i = 0; i < callStack_.size(); i++) {
                ParseFrame(ss, strings[i]);
            }
            free(strings);
            return ss.str();
        });
    }

private:
    mutable LazyShared<std::string> symbols_;
    std::vector<void *> callStack_;
};

Backtrace GetBacktrace(size_t skipFrames, size_t maxFrames) {
    return std::make_shared<BacktraceImpl>(BacktraceImpl{skipFrames, maxFrames});
}

const char *Error::what() const noexcept {
    return what_
        .Ensure([this]() -> std::string {
            std::stringstream ss;
            ss << msg_ << ", func " << func_ << ", file " << StringUtils::BaseName(file_) << ", line " << line_ << "\n";
            ss << backtrace_->Get();
            return ss.str();
        })
        .c_str();
}

} // namespace npu::tile_fwk
