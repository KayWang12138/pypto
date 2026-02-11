/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file monitor_manager.h
 * \brief Compiler monitor manager for tracking compilation progress and detecting timeouts
 */

#ifndef SRC_INTERFACE_COMPILER_MONITOR_MONITOR_MANAGER_H
#define SRC_INTERFACE_COMPILER_MONITOR_MONITOR_MANAGER_H

#include <memory>
#include <string>
#include <mutex>
#include <chrono>
#include <atomic>

#include "compiler_monitor/monitor_exception.h"

namespace npu::tile_fwk {

// Forward declarations
class MonitorImpl;

/**
 * @brief Action to take when a timeout is detected
 */
enum class TimeoutAction {
    THROW_EXCEPTION = 0,  // Throw CompilationTimeoutException
    WARN_ONLY = 1        // Log warning but continue execution
};

/**
 * @brief Configuration options for the compiler monitor
 */
struct MonitorConfig {
    bool enable;                 // Whether monitoring is enabled (default: true)
    int intervalSec;             // Progress print interval in seconds (default: 30s)
    int timeoutSec;              // Per-stage timeout threshold in seconds (default: 600s)
    int totalTimeoutSec;         // Total compilation timeout threshold in seconds (default: 0 = disabled)
    TimeoutAction timeoutAction; // What to do on timeout (default: THROW_EXCEPTION)

    MonitorConfig()
        : enable(true),           // Default enabled
          intervalSec(30),        // Default interval: 30 seconds
          timeoutSec(600),        // Default per-stage timeout: 10 minutes
          totalTimeoutSec(0),     // Default total timeout: disabled (0 = no limit)
          timeoutAction(TimeoutAction::THROW_EXCEPTION) {}

    void SetEnable(bool v) { enable = v; }
    void SetIntervalSec(int v) { intervalSec = (v > 0) ? v : 30; }
    void SetTimeoutSec(int v) { timeoutSec = (v > 0) ? v : 600; }
    void SetTotalTimeoutSec(int v) { totalTimeoutSec = (v >= 0) ? v : 0; }
    void SetTimeoutAction(TimeoutAction v) { timeoutAction = v; }
};

/**
 * @brief Singleton manager for compiler progress monitoring
 *
 * The MonitorManager tracks compilation progress by stages:
 * - Automatically prints progress at configurable intervals
 * - Detects and handles stage timeouts
 * - Outputs to both console and log system
 *
 * Usage:
 * @code
 * // Initialize (usually at program start)
 * MonitorManager::Instance().Initialize();
 *
 * // Track a compilation stage
 * MonitorManager::Instance().StartStage("TensorGraphPass");
 * // ... perform compilation work ...
 * MonitorManager::Instance().EndStage();
 *
 * // Clean up (usually at program shutdown)
 * MonitorManager::Instance().Shutdown();
 * @endcode
 */
class MonitorManager {
public:
    /**
     * @brief Get singleton instance
     */
    static MonitorManager& Instance();

    /**
     * @brief Initialize the monitor
     *
     * Must be called before using StartStage/EndStage.
     * Creates the monitoring thread if enabled.
     */
    void Initialize();

    /**
     * @brief Shutdown the monitor
     *
     * Stops the monitoring thread and cleans up resources.
     */
    void Shutdown();

    /**
     * @brief Start tracking a new compilation stage
     * @param stageName Name of the stage being entered
     *
     * Starts timing for this stage and updates the current stage name
     * for progress reporting.
     */
    void StartStage(const std::string& stageName);

    /**
     * @brief End the current compilation stage
     *
     * Stops timing for the current stage. Note that timing continues
     * to be tracked in the background for monitoring purposes.
     */
    void EndStage();

    /**
     * @brief Get the current configuration
     * @return Reference to the current MonitorConfig
     */
    MonitorConfig& GetConfig() { return config_; }

    /**
     * @brief Get the current configuration (const version)
     * @return Const reference to the current MonitorConfig
     */
    const MonitorConfig& GetConfig() const { return config_; }

    /**
     * @brief Set configuration options
     * @param enable Whether to enable monitoring
     * @param intervalSec Progress print interval (seconds)
     * @param timeoutSec Per-stage timeout threshold (seconds)
     * @param totalTimeoutSec Total compilation timeout threshold (seconds, 0 = disabled)
     * @param timeoutAction What to do on timeout
     */
    void SetOptions(bool enable, int intervalSec, int timeoutSec, int totalTimeoutSec, TimeoutAction timeoutAction);

    /**
     * @brief Check if a timeout has occurred
     * @return true if timeout has been detected (for main thread to check)
     *
     * When timeout is detected in THROW_EXCEPTION mode, this flag is set
     * and the main thread should regularly check it to interrupt compilation.
     */
    bool HasTimeoutOccurred() const { return timeoutOccurred_.load(); }

    /**
     * @brief Clear the timeout flag
     */
    void ClearTimeoutFlag() { timeoutOccurred_.store(false); }

    /**
     * @brief Set timeout flag (called by MonitorImpl on timeout detection)
     */
    void SetTimeoutFlag() { timeoutOccurred_.store(true); }

private:
    MonitorManager();
    ~MonitorManager();

    MonitorManager(const MonitorManager&) = delete;
    MonitorManager& operator=(const MonitorManager&) = delete;

    // Function for MonitorImpl to call when timeout is detected
    friend class MonitorImpl;
    void HandleTimeout(const std::string& stageName, int64_t timeoutSec, int64_t elapsedSec);

    bool enable_;
    bool initialized_;
    MonitorConfig config_;
    std::unique_ptr<MonitorImpl> impl_;
    std::mutex mutex_;

    // Atomic flag for cross-thread timeout notification
    std::atomic<bool> timeoutOccurred_;
};

/**
 * @brief RAII helper for automatic stage tracking
 *
 * Automatically calls EndStage when going out of scope.
 *
 * @code
 * {
 *     MonitorStageScope scope("TensorGraphPass");
 *     // ... do work ...
 * } // EndStage() called automatically
 * @endcode
 */
class MonitorStageScope {
public:
    explicit MonitorStageScope(const std::string& stageName);
    ~MonitorStageScope();

    MonitorStageScope(const MonitorStageScope&) = delete;
    MonitorStageScope& operator=(const MonitorStageScope&) = delete;

    MonitorStageScope(MonitorStageScope&&) noexcept;
    MonitorStageScope& operator=(MonitorStageScope&&) noexcept;

private:
    std::string stageName_;
    std::atomic<bool> active_;  // Prevent double EndStage calls
    std::atomic<int> stageId_;  // Track which stage we're ending
};

} // namespace npu::tile_fwk

#endif // SRC_INTERFACE_COMPILER_MONITOR_MONITOR_MANAGER_H
