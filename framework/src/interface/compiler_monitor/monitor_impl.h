/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file monitor_impl.h
 * \brief Implementation class for compiler monitoring
 */

#ifndef SRC_INTERFACE_COMPILER_MONITOR_MONITOR_IMPL_H
#define SRC_INTERFACE_COMPILER_MONITOR_MONITOR_IMPL_H

#include <atomic>
#include <thread>
#include <string>
#include <chrono>
#include <condition_variable>

#include "monitor_manager.h"

namespace npu::tile_fwk {

/**
 * @brief Internal implementation of compiler monitoring
 *
 * This class manages:
 * - A background monitoring thread that periodically checks progress
 * - Tracking of current stage and elapsed time
 * - Timeout detection and handling
 *
 * The monitoring thread runs independently and can be safely controlled
 * through atomic flags and condition variables.
 */
class MonitorImpl {
public:
    /**
     * @brief Construct a MonitorImpl
     * @param interval Progress reporting interval in seconds
     * @param timeout Per-stage timeout threshold in seconds
     * @param totalTimeout Total compilation timeout threshold in seconds (0 = disabled)
     * @param timeoutAction What to do when timeout occurs
     */
    MonitorImpl(std::chrono::seconds interval,
                std::chrono::seconds timeout,
                std::chrono::seconds totalTimeout,
                TimeoutAction timeoutAction);

    ~MonitorImpl();

    /**
     * @brief Start tracking a new stage
     * @param stageName Name of the stage being entered
     */
    void StartStage(const std::string& stageName);

    /**
     * @brief End the current stage and print elapsed time if it's a coarse-grained stage
     */
    void EndStage();

    /**
     * @brief Reset the total start time (call when actual compilation starts)
     */
    void ResetStartTime();

    /**
     * @brief Start the monitoring thread
     */
    void Start();

    /**
     * @brief Stop the monitoring thread
     */
    void Stop();

    /**
     * @brief Get the stage count (number of stages tracked so far)
     */
    int64_t GetStageCount() const { return stageCount_; }

private:
    /**
     * @brief Main monitoring thread loop
     */
    void MonitorLoop();

    /**
     * @brief Print current progress information
     */
    void PrintProgress();

    /**
     * @brief Check for timeout condition
     * @return true if timeout has occurred
     */
    bool CheckTimeout();

    /**
     * @brief Handle timeout detected
     * @param stageName Name of the stage that timed out
     * @param timeoutSec Timeout threshold in seconds
     * @param elapsedSec Actual elapsed time in seconds
     *
     * Instead of throwing an exception in the monitor thread (which won't stop
     * the compilation), this sets a flag that the main thread can check.
     */
    void HandleTimeout(const std::string& stageName, int64_t timeoutSec, int64_t elapsedSec);

    /**
     * @brief Format elapsed time for display
     * @param elapsedSec Elapsed time in seconds
     * @return Formatted string like "2min 35s"
     */
    std::string FormatElapsedTime(int64_t elapsedSec) const;

    // Thread control
    std::atomic<bool> running_;
    std::thread monitor_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;

    // Timing information
    std::chrono::steady_clock::time_point total_start_;  // Current monitoring start time (resets at compilation start)
    std::chrono::steady_clock::time_point stage_start_;
    std::string current_stage_;
    int64_t stageCount_;
    std::chrono::steady_clock::time_point last_print_time_;  // Last progress print
    std::chrono::seconds python_elapsed_time_;  // Time spent in Python phase before compilation starts
    bool python_elapsed_captured_;  // Whether Python elapsed time has been captured

    // Configuration
    std::chrono::seconds interval_;
    std::chrono::seconds timeout_;
    std::chrono::seconds totalTimeout_;  // Total compilation timeout (0 = disabled)
    TimeoutAction timeoutAction_;

    // Flag to prevent multiple timeout throws
    std::atomic<bool> timeoutThrown_;
};

} // namespace npu::tile_fwk

#endif // SRC_INTERFACE_COMPILER_MONITOR_MONITOR_IMPL_H
