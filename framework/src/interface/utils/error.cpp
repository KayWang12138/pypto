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
 * \file error.cpp
 * \brief
 */

#include <cstring>
#include <sstream>
#include <functional>
#include <cxxabi.h>
#include <securec.h>
#include <fstream>
#include <dlfcn.h>
#include <mutex>
#include <unordered_map>

#include "error.h"
#include "interface/utils/string_utils.h"

namespace npu::tile_fwk {

// Helper function to read a specific line from a source file
static std::string ReadSourceLine(const std::string& filename, int lineno) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return "";
    }

    std::string line;
    int current_line = 0;
    while (std::getline(file, line)) {
        current_line++;
        if (current_line == lineno) {
            // Trim leading whitespace for display
            size_t start = line.find_first_not_of(" \t");
            if (start != std::string::npos) {
                return line.substr(start);
            }
            return line;
        }
    }
    return "";
}

// Structure to hold file location information
struct FileLocation {
    std::string filename;
    int lineno;
};

// Cache for symbol resolution to avoid repeated addr2line calls
static std::mutex locMapMutex;
static std::unordered_map<void*, FileLocation> locMap;

// Get file and line information from address using addr2line
static FileLocation GetFileLineFromAddr2line(void* addr) {
    // Check cache first
    {
        std::lock_guard<std::mutex> lock(locMapMutex);
        auto it = locMap.find(addr);
        if (it != locMap.end()) {
            return it->second;
        }
    }

    FileLocation loc{"", 0};

    // Get library information
    Dl_info info;
    if (dladdr(addr, &info) == 0 || info.dli_fname == nullptr) {
        return loc;
    }

    // Build addr2line command - use absolute address for executable
    std::stringstream cmd;
    cmd << "addr2line -e " << info.dli_fname << " -f -C -p " << addr << " 2>/dev/null";

    FILE* fp = popen(cmd.str().c_str(), "r");
    if (fp == nullptr) {
        return loc;
    }

    char buffer[2048];
    if (fgets(buffer, sizeof(buffer), fp) != nullptr) {
        std::string output(buffer);

        // Remove trailing newline
        if (!output.empty() && output.back() == '\n') {
            output.pop_back();
        }

        // Parse output format: "function at filename:lineno"
        // or "function at filename:lineno:column"
        size_t atPos = output.find(" at ");
        if (atPos != std::string::npos) {
            std::string location = output.substr(atPos + 4);

            // Find the last colon for line number
            size_t colonPos = location.rfind(':');
            if (colonPos != std::string::npos) {
                // Check if this is line:column format
                size_t prevColonPos = location.rfind(':', colonPos - 1);
                if (prevColonPos != std::string::npos) {
                    // Has column number, use the previous colon
                    loc.filename = location.substr(0, prevColonPos);
                    try {
                        std::string lineStr = location.substr(prevColonPos + 1, colonPos - prevColonPos - 1);
                        loc.lineno = std::stoi(lineStr);
                    } catch (...) {
                        loc.lineno = 0;
                    }
                } else {
                    // No column number
                    loc.filename = location.substr(0, colonPos);
                    try {
                        loc.lineno = std::stoi(location.substr(colonPos + 1));
                    } catch (...) {
                        loc.lineno = 0;
                    }
                }
            }
        } else if (output.find("??:?") == std::string::npos && output.find(":") != std::string::npos) {
            // Try alternate format: "filename:lineno"
            size_t colonPos = output.rfind(':');
            if (colonPos != std::string::npos) {
                loc.filename = output.substr(0, colonPos);
                try {
                    loc.lineno = std::stoi(output.substr(colonPos + 1));
                } catch (...) {
                    loc.lineno = 0;
                }
            }
        }
    }
    pclose(fp);

    // Cache the result (even if empty, to avoid repeated failed lookups)
    {
        std::lock_guard<std::mutex> lock(locMapMutex);
        locMap[addr] = loc;
    }

    return loc;
}

class BacktraceImpl : public LazyValue<std::string> {
public:
    __always_inline BacktraceImpl(size_t skipFrames, size_t maxFrames) : callStack_(maxFrames, 0) {
        skipFrames += 1;
        auto nrFrames = static_cast<size_t>(::backtrace(callStack_.data(), static_cast<int>(callStack_.size())));
        skipFrames = std::min(skipFrames, nrFrames);
        callStack_.erase(callStack_.begin(), callStack_.begin() + static_cast<ssize_t>(skipFrames));
        callStack_.resize(nrFrames - skipFrames);
    }

    void ParseFrame(std::stringstream &ss, char *line, void* addr, bool &isPyptoFrame) const {
        auto funcName = strchr(line, '(');
        auto funcOffset = strchr(line, '+');
        auto libname = strrchr(line, '/');
        if (funcName == nullptr || funcOffset == nullptr) {
            ss << line <<'\n';
            return;
        }

        *funcName++ = '\0';
        *funcOffset++ = '\0';
        libname = (libname == nullptr) ? line : libname + 1;
        if (!strncmp(libname, "pypto_impl", strlen("pypto_impl"))) {
            isPyptoFrame = true;
        } else if (isPyptoFrame) {
            // python frames after pypto frame, skip it
            return;
        }

        int status = 0;
        std::unique_ptr<char, std::function<void(char *)>> demangled(
            abi::__cxa_demangle(funcName, nullptr, nullptr, &status),
            /* deleter */ free);
        if (status == 0)
            funcName = demangled.get();

        // Try to get file and line information using addr2line
        FileLocation loc = GetFileLineFromAddr2line(addr);

        if (!loc.filename.empty() && loc.lineno > 0) {
            // Python-style format: File "filename", line X
            ss << " File \"" << loc.filename << "\", line " << loc.lineno << "\n";

            // Display source code line if available
            std::string sourceLine = ReadSourceLine(loc.filename, loc.lineno);
            if (!sourceLine.empty()) {
                ss << "   " << sourceLine << "\n";
            }
        } else {
            // Fallback to traditional format if addr2line fails
            ss << libname << '(' << funcName << '+' << funcOffset << '\n';
        }
    }

    const std::string &Get() const {
        return symbols_.Ensure([this]() -> std::string {
            auto strings = backtrace_symbols(callStack_.data(), callStack_.size());
            if (strings == nullptr) {
                return "Backtrace Failed";
            }
            std::stringstream ss;

            // Add Python-style traceback header
            ss << "\nC++ Traceback (most recent call last):\n";

            bool isPyptoFrame = false;
            // Reverse the frames to show most recent last (Python style)
            for (int i = static_cast<int>(callStack_.size()) - 1; i >= 0; i--) {
                ParseFrame(ss, strings[i], callStack_[i], isPyptoFrame);
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

static struct TerminateHandler terminateHandler;
} // namespace npu::tile_fwk
