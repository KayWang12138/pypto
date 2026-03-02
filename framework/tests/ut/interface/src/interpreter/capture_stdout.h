/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file capture_stdout.h
 * \brief Common helper to capture stdout into a std::string while also echoing
 *        the output back to the original stdout for debug visibility.
 */

#pragma once

#include <functional>
#include <string>
#include <cstdio>
#include <unistd.h>

inline std::string CaptureStdout(std::function<void()> func) {
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
        // Echo back to original stdout for debugging.
        ssize_t written = write(old_stdout, buffer, static_cast<size_t>(len));
        (void)written;
    }
    close(old_stdout);

    return captured;
}

