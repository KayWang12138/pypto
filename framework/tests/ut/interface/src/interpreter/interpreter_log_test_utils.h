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
 *        - CaptureStdoutAndEcho: capture and echo stdout
 *        - VerifyLogContainsFailed: check VERIFY log failures
 *        - VerifyLogContainsIndex0Failed: check VERIFY index 0 failures
 */

#pragma once

#include <functional>
#include <string>
#include <cstdio>
#include <unistd.h>

static inline std::string CaptureStdoutAndEcho(std::function<void()> func) {
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return "";
    }
    int old_stdout = dup(STDOUT_FILENO);
    if (old_stdout == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        return "";
    }
    if (dup2(pipefd[1], STDOUT_FILENO) == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        close(old_stdout);
        return "";
    }
    close(pipefd[1]);
    func();
    fflush(stdout);
    if (dup2(old_stdout, STDOUT_FILENO) == -1) {
        close(pipefd[0]);
        close(old_stdout);
        return "";
    }
    char buffer[8192] = {0};
    ssize_t len = read(pipefd[0], buffer, sizeof(buffer) - 1);
    close(pipefd[0]);
    std::string captured(len > 0 ? static_cast<size_t>(len) : 0, '\0');
    if (len > 0) {
        captured.assign(buffer, static_cast<size_t>(len));
        ssize_t written = write(old_stdout, buffer, static_cast<size_t>(len));
        (void)written;
    }
    close(old_stdout);
    return captured;
}

// 仅检查 [VERIFY] 日志行中是否出现 FAILED，其他模块日志不参与判断
inline bool VerifyLogContainsFailed(const std::string& logOutput) {
    size_t pos = 0;
    while (pos < logOutput.size()) {
        size_t lineEnd = logOutput.find('\n', pos);
        size_t lineLen = (lineEnd == std::string::npos) ? logOutput.size() - pos : lineEnd - pos;
        std::string line = logOutput.substr(pos, lineLen);
        if (line.find("[VERIFY]") != std::string::npos && line.find("FAILED") != std::string::npos) {
            return true;
        }
        if (lineEnd == std::string::npos) {
            break;
        }
        pos = lineEnd + 1;
    }
    return false;
}

// 仅检查 [VERIFY] 日志行中 index 0 是否出现 FAILED，用于 Topk 用例
inline bool VerifyLogContainsIndex0Failed(const std::string& logOutput) {
    size_t pos = 0;
    while (pos < logOutput.size()) {
        size_t lineEnd = logOutput.find('\n', pos);
        size_t lineLen = (lineEnd == std::string::npos) ? logOutput.size() - pos : lineEnd - pos;
        std::string line = logOutput.substr(pos, lineLen);
        // 只关心 flow_verifier 打印的 index 0 结果行：
        // "... [VERIFY]: ... Verify for ... index 0 result FAILED"
        if (line.find("[VERIFY]") != std::string::npos &&
            line.find("index 0 result FAILED") != std::string::npos) {
            return true;
        }
        if (lineEnd == std::string::npos) {
            break;
        }
        pos = lineEnd + 1;
    }
    return false;
}

