/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_util.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace npu::tile_fwk {

MonitorManager::~MonitorManager() {
    Shutdown();
}

void MonitorManager::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return;
    }
    if (!enable_) {
        return;
    }
    impl_ = new MonitorImpl(this);
    current_stage_ = "Prepare";
    total_start_ = std::chrono::steady_clock::now();
    stage_start_ = total_start_;
    last_print_time_ = total_start_;
    next_function_index_ = 1;
    impl_->Start();
    initialized_ = true;
    // Mark that Prepare stage has started (use env var for cross-.so communication)
    (void)setenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED", "1", 1);
}

void MonitorManager::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return;
    }
    if (impl_) {
        impl_->Stop();
        delete impl_;
        impl_ = nullptr;
    }
    initialized_ = false;
}

void MonitorManager::MaybeStartTotalClock() {
    if (current_stage_.empty()) {
        total_start_ = std::chrono::steady_clock::now();
        stage_start_ = total_start_;
        last_print_time_ = total_start_;
    }
}

void MonitorManager::StartStage(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !impl_) {
        return;
    }
    MaybeStartTotalClock();
    current_stage_ = name;
    stage_start_ = std::chrono::steady_clock::now();
    last_print_time_ = stage_start_;
}

void MonitorManager::EndStage(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return;
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - stage_start_).count();
    stage_elapsed_totals_[name] += elapsed;
}

void MonitorManager::SetTotalFunctionCount(int n) {
    MonitorImpl* to_start = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Check if Prepare stage was started via Python (env var indicates this)
        bool prepare_started = (std::getenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED") != nullptr);
        if (!initialized_ && enable_ && !prepare_started) {
            // First time initialization (no Prepare stage from Python)
            if (!impl_) {
                impl_ = new MonitorImpl(this);
            }
            current_stage_ = "Prepare";
            total_start_ = std::chrono::steady_clock::now();
            stage_start_ = total_start_;
            last_print_time_ = total_start_;
            stage_elapsed_totals_.clear();
            python_stage_ended_ = false;
            to_start = impl_;
            initialized_ = true;
        }
        total_function_count_ = n;
        next_function_index_ = 1;
        current_function_index_ = 0;
        (void)setenv("PYPTO_COMPILER_MONITOR_CURRENT", "0", 1);  // 进程内唯一，避免多 .so 多单例
    }
    if (to_start != nullptr) {
        to_start->Start();
    }
}

int MonitorManager::GetAndIncrementNextFunctionIndex() {
    std::lock_guard<std::mutex> lock(mutex_);
    int k = next_function_index_++;
    return k;
}

void MonitorManager::SetCurrentFunctionIndex(int k) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_function_index_ = k;
    std::string val = std::to_string(k);
    (void)setenv("PYPTO_COMPILER_MONITOR_CURRENT", val.c_str(), 1);
}

void MonitorManager::TryEndPrepareStage() {
    std::lock_guard<std::mutex> lock(mutex_);
    // If not initialized but Prepare was started (env var set), initialize now
    if (!initialized_) {
        const char* prepare_started = std::getenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED");
        if (prepare_started != nullptr && prepare_started[0] == '1') {
            // Python side Initialize was called, initialize this instance
            if (enable_ && !impl_) {
                impl_ = new MonitorImpl(this);
            }

            // Try to get prepare start time from environment (set by Python)
            const char* prepare_start_time_str = std::getenv("PYPTO_COMPILER_MONITOR_PREPARE_START_TIME");
            if (prepare_start_time_str != nullptr) {
                // Python side recorded start time, use current time as end time
                // Elapsed = now - Python start time
                double prepare_start_time = std::atof(prepare_start_time_str);
                double now_time = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
                double elapsed = now_time - prepare_start_time;
                initialized_ = true;
                python_stage_ended_ = true;
                stage_elapsed_totals_["Prepare"] += elapsed;
                return;
            }

            current_stage_ = "Prepare";
            total_start_ = std::chrono::steady_clock::now();
            stage_start_ = total_start_;
            initialized_ = true;
        }
    }
    if (!initialized_ || python_stage_ended_) {
        return;
    }
    python_stage_ended_ = true;
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - stage_start_).count();
    stage_elapsed_totals_["Prepare"] += elapsed;
}

void MonitorManager::NotifyCompilationFinished() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return;
    }
    PrintCompilationFinished();
    if (impl_) {
        impl_->Stop();
        delete impl_;
        impl_ = nullptr;
    }
    initialized_ = false;
}

void MonitorManager::PrintCompilationFinished() {
    auto now = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(now - total_start_).count();
    // Calculate total from all stage elapsed totals (sum of all stages)
    // This ensures Total elapsed includes Prepare time from Python side
    double stage_total = 0.0;
    for (const auto& [stage, sec] : stage_elapsed_totals_) {
        stage_total += sec;
    }
    // Use the larger of: clock-based total vs sum of stages
    // (sum of stages may be more accurate when Prepare was tracked via Python)
    if (stage_total > total_elapsed) {
        total_elapsed = stage_total;
    }
    (void)fprintf(stderr, "[Compiler Monitor] Compilation finished | Total functions: %d\n",
                  total_function_count_ > 0 ? total_function_count_ : 1);
    (void)fprintf(stderr, "[Compiler Monitor] Stage timing (aggregated by stage):\n");
    int n = total_function_count_ > 0 ? total_function_count_ : 1;
    for (const auto& [stage, sec] : stage_elapsed_totals_) {
        if (stage == "Pass" || stage == "CodeGen") {
            (void)fprintf(stderr, "  %-8s %.1fs   (sum over %d functions)\n", stage.c_str(), sec, n);
        } else {
            (void)fprintf(stderr, "  %-8s %.1fs\n", stage.c_str(), sec);
        }
    }
    (void)fprintf(stderr, "[Compiler Monitor] Monitoring stopped | Total elapsed: %s\n",
                  FormatElapsed(total_elapsed).c_str());
    (void)fflush(stderr);
}

void MonitorManager::SetCompilerMonitorOptions(bool enable, int interval_sec) {
    std::lock_guard<std::mutex> lock(mutex_);
    enable_ = enable;
    interval_sec_ = (interval_sec > 0) ? interval_sec : 30;
    std::string interval_str = std::to_string(interval_sec_);
    (void)setenv("PYPTO_COMPILER_MONITOR_INTERVAL_SEC", interval_str.c_str(), 1);
}

bool MonitorManager::IsEnabled() const {
    return enable_;
}

int MonitorManager::GetIntervalSec() const {
    const char* v = std::getenv("PYPTO_COMPILER_MONITOR_INTERVAL_SEC");
    if (v != nullptr) {
        int env_val = std::atoi(v);
        if (env_val > 0) {
            return env_val;
        }
    }
    return interval_sec_;
}

std::string MonitorManager::GetCurrentStageName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_stage_;
}

std::chrono::steady_clock::time_point MonitorManager::GetStageStartTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stage_start_;
}

std::chrono::steady_clock::time_point MonitorManager::GetTotalStartTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_start_;
}

std::chrono::steady_clock::time_point MonitorManager::GetLastPrintTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_print_time_;
}

void MonitorManager::SetLastPrintTime(std::chrono::steady_clock::time_point t) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_print_time_ = t;
}

int MonitorManager::GetTotalFunctionCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_function_count_;
}

int MonitorManager::GetCurrentFunctionIndex() const {
    const char* v = std::getenv("PYPTO_COMPILER_MONITOR_CURRENT");
    if (v != nullptr) {
        int env_val = std::atoi(v);
        if (env_val >= 0) {
            return env_val;
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return current_function_index_;
}

std::map<std::string, double> MonitorManager::GetStageElapsedTotals() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stage_elapsed_totals_;
}

}  // namespace npu::tile_fwk
