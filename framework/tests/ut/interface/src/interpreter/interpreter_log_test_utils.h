/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file interpreter_log_test_utils.h
 * \brief Common helpers for interpreter/log related unit tests:
 *        - CaptureStdoutAndEcho: capture log output from log files (落盘形式，与 LogManager 路径一致)
 *        - VerifyLogContainsFailed: check VERIFY log failures
 *        - VerifyLogContainsIndex0Failed: check VERIFY index 0 failures
 */

#pragma once

#include <functional>
#include <string>
#include <cstdio>
#include <unistd.h>
#include <regex>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <map>

// 与 LogManager 落盘路径一致（仅本头文件内使用）
static constexpr const char *kInterpLogTestEnvProcessLogPath = "ASCEND_PROCESS_LOG_PATH";
static constexpr const char *kInterpLogTestHostLogFilePrefix = "pypto-log-";
static constexpr const char *kInterpLogTestLogFileSuffix = ".log";
static constexpr const char *kInterpLogTestHostLogSubDir = "/debug/plog";
static constexpr const char *kInterpLogTestDefaultLogSubDir = "/ascend/log";

static inline std::string InterpLogTestGetHostLogDir() {
    const char *envPath = std::getenv(kInterpLogTestEnvProcessLogPath);
    if (envPath != nullptr && envPath[0] != '\0') {
        return std::string(envPath) + kInterpLogTestHostLogSubDir;
    }
    const char *home = std::getenv("HOME");
    std::string base = (home != nullptr && home[0] != '\0') ? std::string(home) : ".";
    return base + kInterpLogTestDefaultLogSubDir + kInterpLogTestHostLogSubDir;
}

static inline std::map<std::string, size_t> InterpLogTestListHostLogFilesWithSize(const std::string &dir) {
    std::map<std::string, size_t> result;
    DIR *d = opendir(dir.c_str());
    if (d == nullptr) {
        return result;
    }
    struct dirent *dp = nullptr;
    while ((dp = readdir(d)) != nullptr) {
        if (dp->d_name[0] == '.') {
            continue;
        }
        std::string name = dp->d_name;
        if (name.find(kInterpLogTestHostLogFilePrefix) != 0 ||
            name.rfind(kInterpLogTestLogFileSuffix) != name.size() - std::strlen(kInterpLogTestLogFileSuffix)) {
            continue;
        }
        std::string path = dir + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            result[path] = static_cast<size_t>(st.st_size);
        }
    }
    closedir(d);
    return result;
}

// 从日志落盘目录捕获本次 func() 执行产生的新增日志内容（与 LogManager 落盘路径一致）
static inline std::string CaptureStdoutAndEcho(std::function<void()> func) {
    std::string logDir = InterpLogTestGetHostLogDir();
    std::map<std::string, size_t> before = InterpLogTestListHostLogFilesWithSize(logDir);

    func();

    std::map<std::string, size_t> after = InterpLogTestListHostLogFilesWithSize(logDir);
    std::string captured;
    for (const auto &p : after) {
        const std::string &path = p.first;
        size_t newSize = p.second;
        size_t oldSize = 0;
        auto it = before.find(path);
        if (it != before.end()) {
            oldSize = it->second;
            if (newSize <= oldSize) {
                continue;
            }
        }
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            continue;
        }
        ifs.seekg(static_cast<std::streamoff>(oldSize));
        captured.append(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    }
    return captured;
}

// 仅检查 [VERIFY] 日志行中是否出现 FAILED，其他模块日志不参与判断
inline bool VerifyLogContainsFailed(const std::string& logOutput) {
    // 匹配形如："...[VERIFY]...FAILED..."，且 [VERIFY] 与 FAILED 必须在同一行（中间不允许换行）
    static const std::regex kVerifyFailedPattern(R"(\[VERIFY][^\n]*FAILED)");
    return std::regex_search(logOutput, kVerifyFailedPattern);
}

// 仅检查 [VERIFY] 日志行中 index 0 是否出现 FAILED，用于 Topk 用例
inline bool VerifyLogContainsIndex0Failed(const std::string& logOutput) {
    // 只关心 flow_verifier 打印的 index 0 结果行：
    // "... [VERIFY]: ... Verify for ... index 0 result FAILED"
    static const std::regex kVerifyIndex0FailedPattern(R"(\[VERIFY][^\n]*index 0 result FAILED)");
    return std::regex_search(logOutput, kVerifyIndex0FailedPattern);
}

