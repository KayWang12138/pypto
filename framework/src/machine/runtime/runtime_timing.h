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
 * \file runtime_timing.h
 * \brief Performance timing utilities for runtime operations
 */

#pragma once

#include <chrono>
#include <cstdint>
#include "interface/utils/log.h"

namespace npu::tile_fwk {

// Helper function to get microseconds timestamp
static inline uint64_t GetTimeUs() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

// Unified timing macro for RT operations
#define RT_TIMING_WRAP(func_call, func_name, result_var) \
    do { \
        uint64_t _rt_start_ = GetTimeUs(); \
        result_var = func_call; \
        uint64_t _rt_end_ = GetTimeUs(); \
        ALOG_INFO_F("[TIMING] RT:%s elapsed=%lu us, ret=%d", func_name, _rt_end_ - _rt_start_, result_var); \
    } while(0)

// RAII helper for function timing (unified format)
class FuncTimer {
public:
    FuncTimer(const char* func_name) : func_name_(func_name), start_time_(GetTimeUs()) {}
    ~FuncTimer() {
        uint64_t end_time = GetTimeUs();
        ALOG_INFO_F("[TIMING] FUNC:%s elapsed=%lu us", func_name_, end_time - start_time_);
    }
private:
    const char* func_name_;
    uint64_t start_time_;
};

} // namespace npu::tile_fwk
