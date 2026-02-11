/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\\file monitor_impl.cpp
 * \\brief Implementation of compiler monitoring
 */

#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_exception.h"
#include "interface/utils/log.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace npu::tile_fwk {

// Helper function to format elapsed time
static std::string FormatTime(int64_t elapsedSec) {
    if (elapsedSec < 60) {
        return std::to_string(elapsedSec) + "s";
    }

    int64_t minutes = elapsedSec / 60;
    int64_t seconds = elapsedSec % 60;

    std::ostringstream oss;
    oss << minutes << "min";
    if (seconds > 0) {
        oss << " " << seconds << "s";
    }
    return oss.str();
}

// Helper function to check if a stage is coarse-grained
static bool IsCoarseStage(const std::string& stageName) {
    static const std::vector<std::string> coarseStages = {
        "Python",
        "TensorGraphPass",
        "UpdateCompileTask",
        "TileGraphPass",
        "BlockGraphPass",
        "CodeGen",
        "BinaryGeneration",
        "FrontendParser"
    };

    for (const auto& coarseStage : coarseStages) {
        if (stageName == coarseStage) {
            return true;
        }
    }
    return false;
}

MonitorImpl::MonitorImpl(std::chrono::seconds interval,
                         std::chrono::seconds timeout,
                         std::chrono::seconds totalTimeout,
                         TimeoutAction timeoutAction)
    : running_(false),
      total_start_(std::chrono::steady_clock::now()),
      stage_start_(std::chrono::steady_clock::now()),
      current_stage_("Python"),
      stageCount_(0),
      last_print_time_(total_start_),
      python_elapsed_time_(std::chrono::seconds(0)),
      python_elapsed_captured_(false),
      interval_(interval),
      timeout_(timeout),
      totalTimeout_(totalTimeout),
      timeoutAction_(timeoutAction),
      timeoutThrown_(false) {
    ALOG_INFO("MonitorImpl: interval=%lds, timeout=%lds, total_timeout=%lds, action=%d",
              interval_.count(), timeout_.count(), totalTimeout_.count(), static_cast<int>(timeoutAction_));
}

MonitorImpl::~MonitorImpl() {
    Stop();
}

void MonitorImpl::Start() {
    if (running_.load()) {
        return;  // Already running
    }

    running_.store(true);
    monitor_thread_ = std::thread(&MonitorImpl::MonitorLoop, this);
}

void MonitorImpl::StartStage(const std::string& stageName) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::steady_clock::now();

    // Capture Python elapsed time when transitioning from initial "Python" state
    // to the first real compilation phase, then reset total_start_ to now.
    // This preserves the Python execution time while avoiding double-counting.
    if (current_stage_ == "Python" && IsCoarseStage(stageName) && !python_elapsed_captured_) {
        // Save Python execution time
        python_elapsed_time_ = std::chrono::duration_cast<std::chrono::seconds>(now - total_start_);
        python_elapsed_captured_ = true;
        // Reset total_start_ to compilation start time (avoids double-counting)
        total_start_ = now;
        // Reset last_print_time_ to avoid printing stale progress
        last_print_time_ = now;
    }

    ++stageCount_;
    current_stage_ = stageName;
    stage_start_ = now;
    timeoutThrown_.store(false);

    ALOG_INFO("[Compiler Monitor] Entering stage: %s", stageName.c_str());
}

void MonitorImpl::ResetStartTime() {
    std::lock_guard<std::mutex> lock(mutex_);
    total_start_ = std::chrono::steady_clock::now();
}

void MonitorImpl::EndStage() {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - stage_start_).count();

    if (elapsed < 0) {
        elapsed = 0;
    }

    std::string elapsedStr = FormatTime(elapsed);

    // Calculate total elapsed time: Python time (if any) + (now - total_start_)
    // This calculation is consistent with MonitorLoop to ensure consistency
    int64_t totalElapsed = python_elapsed_time_.count() +
                           std::chrono::duration_cast<std::chrono::seconds>(
                               now - total_start_).count();

    std::string totalElapsedStr = FormatTime(totalElapsed);

    // Print completion message for coarse-grained stages only
    if (IsCoarseStage(current_stage_)) {
        std::cout << "[Compiler Monitor] " << current_stage_ << " completed | "
                  << "Stage elapsed: " << elapsedStr << " | "
                  << "Total elapsed: " << totalElapsedStr << " ("
                  << totalElapsed << "s)" << std::endl;
        ALOG_INFO("[Compiler Monitor] %s completed | Stage elapsed: %s | Total elapsed: %s (%lds)",
                  current_stage_.c_str(), elapsedStr.c_str(), totalElapsedStr.c_str(), totalElapsed);
    }
}

void MonitorImpl::Stop() {
    if (running_.load()) {
        running_.store(false);
        cv_.notify_all();
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }

        // Only print total elapsed time if there was actual monitoring activity
        auto now = std::chrono::steady_clock::now();
        int64_t totalElapsed = python_elapsed_time_.count() +
                               std::chrono::duration_cast<std::chrono::seconds>(
                                   now - total_start_).count();

        // Print only if there was actual monitoring time (more than just initialization overhead)
        if (totalElapsed > 0 || stageCount_ > 0) {
            std::string totalElapsedStr = FormatTime(totalElapsed);
            std::cout << "[Compiler Monitor] Monitoring stopped | Total elapsed: "
                      << totalElapsedStr << " (" << totalElapsed << "s)" << std::endl;
            ALOG_INFO("MonitorImpl: Stopped | Total elapsed: %s (%lds)",
                      totalElapsedStr.c_str(), totalElapsed);
        }
    }
}

void MonitorImpl::MonitorLoop() {
    ALOG_INFO("MonitorImpl: Monitoring thread started, interval=%lds", interval_.count());

    while (running_.load()) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for the specified interval or until stopped
        if (cv_.wait_for(lock, interval_, [this]() { return !running_.load(); })) {
            break;  // Stopping
        }

        // Get current time and calculate total elapsed while holding the lock
        // This ensures total_start_ is not read while StartStage() is modifying it
        auto now = std::chrono::steady_clock::now();
        int64_t elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - stage_start_).count();
        auto time_since_last_print = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_print_time_);

        // Calculate total elapsed: Python time (if any) + (now - total_start_)
        // Note: total_start_ is reset to compilation start time after Python phase completes,
        // so there's no double-counting of Python execution time.
        int64_t total_elapsed = python_elapsed_time_.count() +
                               std::chrono::duration_cast<std::chrono::seconds>(
                                   now - total_start_).count();

        // Copy data that won't change during lock release
        std::string stage_name = current_stage_;
        int64_t timeout_value = timeout_.count();
        int64_t total_timeout_value = totalTimeout_.count();

        // Release lock before any I/O operations
        lock.unlock();

        if (elapsed < 0) {
            elapsed = 0;
        }
        if (total_elapsed < 0) {
            total_elapsed = 0;
        }

        // Check for per-stage timeout
        if (!timeoutThrown_.load() && elapsed > timeout_value) {
            timeoutThrown_.store(true);

            int64_t timeoutMin = timeout_value / 60;
            int64_t elapsedMin = elapsed / 60;
            std::string elapsedStr = FormatTime(elapsed);

            std::ostringstream msg;
            msg << "[Compiler Timeout] Stage '" << stage_name
                << "' exceeded timeout (";
            if (timeoutMin > 0) {
                msg << timeoutMin << "min " << (timeout_value % 60) << "s";
            } else {
                msg << timeout_value << "s";
            }
            msg << "). Elapsed: ";
            if (elapsedMin > 0) {
                msg << elapsedMin << "min " << (elapsed % 60) << "s";
            } else {
                msg << elapsed << "s";
            }

            std::string msgStr = msg.str();
            std::cerr << msgStr << std::endl;
            ALOG_ERROR("%s", msgStr.c_str());

            if (timeoutAction_ == TimeoutAction::THROW_EXCEPTION) {
                throw CompilationTimeoutException(stage_name, timeout_value, elapsed);
            }
        }

        // Check for total compilation timeout (if enabled, i.e., totalTimeout_ > 0)
        if (total_timeout_value > 0 && !timeoutThrown_.load() && total_elapsed > total_timeout_value) {
            timeoutThrown_.store(true);

            int64_t totalTimeoutMin = total_timeout_value / 60;
            int64_t totalElapsedMin = total_elapsed / 60;
            std::string totalElapsedStr = FormatTime(total_elapsed);

            std::ostringstream msg;
            msg << "[Compiler Timeout] Total compilation time exceeded timeout (";
            if (totalTimeoutMin > 0) {
                msg << totalTimeoutMin << "min " << (total_timeout_value % 60) << "s";
            } else {
                msg << total_timeout_value << "s";
            }
            msg << "). Total elapsed: ";
            if (totalElapsedMin > 0) {
                msg << totalElapsedMin << "min " << (total_elapsed % 60) << "s";
            } else {
                msg << total_elapsed << "s";
            }
            msg << ". Current stage: " << stage_name << ".";

            std::string msgStr = msg.str();
            std::cerr << msgStr << std::endl;
            ALOG_ERROR("%s", msgStr.c_str());

            if (timeoutAction_ == TimeoutAction::THROW_EXCEPTION) {
                throw CompilationTimeoutException("Total Compilation", total_timeout_value, total_elapsed);
            }
        }

        // Print progress if it's been at least interval seconds since last print
        if (time_since_last_print.count() >= interval_.count()) {
            // Update last_print_time
            {
                std::lock_guard<std::mutex> print_lock(mutex_);
                last_print_time_ = now;
            }

            std::string elapsedStr = FormatTime(elapsed);
            std::string totalElapsedStr = FormatTime(total_elapsed);

            std::cout << "[Compiler Monitor] Stage: " << stage_name
                      << " | Stage elapsed: " << elapsedStr << " | "
                      << "Total elapsed: " << totalElapsedStr
                      << std::endl;
            ALOG_INFO("[Compiler Monitor] Stage: %s | Stage elapsed: %s | Total elapsed: %s",
                      stage_name.c_str(), elapsedStr.c_str(), totalElapsedStr.c_str());
        }
    }

    ALOG_INFO("MonitorImpl: Monitoring thread stopped");
}

void MonitorImpl::PrintProgress() {
    // Not used in the new implementation (MonitorLoop handles printing directly)
}

bool MonitorImpl::CheckTimeout() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (timeoutThrown_.load()) {
        return false;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - stage_start_);

    if (elapsed > timeout_) {
        timeoutThrown_.store(true);
        return true;
    }

    return false;
}

void MonitorImpl::HandleTimeout(const std::string& stageName, int64_t timeoutSec, int64_t elapsedSec) {
    // Not used in the new implementation (handled in MonitorLoop)
}

std::string MonitorImpl::FormatElapsedTime(int64_t elapsedSec) const {
    return FormatTime(elapsedSec);
}

} // namespace npu::tile_fwk
