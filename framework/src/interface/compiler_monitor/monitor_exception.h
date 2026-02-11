/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file monitor_exception.h
 * \brief Custom exceptions for compiler monitoring
 */

#ifndef SRC_INTERFACE_COMPILER_MONITOR_MONITOR_EXCEPTION_H
#define SRC_INTERFACE_COMPILER_MONITOR_MONITOR_EXCEPTION_H

#include <stdexcept>
#include <string>
#include <sstream>

namespace npu::tile_fwk {

/**
 * @brief Exception thrown when a compilation stage times out
 *
 * This exception is raised when a compilation stage exceeds the configured
 * timeout threshold. The exception message contains:
 * - The name of the timed-out stage
 * - The timeout threshold
 * - The actual elapsed time
 *
 * Example:
 * @code
 * try {
 *     MonitorManager::Instance().StartStage("TensorGraphPass");
 *     // ... long running compilation ...
 *     MonitorManager::Instance().EndStage();
 * } catch (const CompilationTimeoutException& e) {
 *     std::cerr << "Compilation timeout: " << e.what() << std::endl;
 * }
 * @endcode
 */
class CompilationTimeoutException : public std::runtime_error {
public:
    /**
     * @brief Construct a CompilationTimeoutException
     * @param stageName The name of the stage that timed out
     * @param timeoutSec The timeout threshold in seconds
     * @param elapsedSec The actual elapsed time in seconds
     */
    CompilationTimeoutException(const std::string& stageName, int64_t timeoutSec, int64_t elapsedSec)
        : std::runtime_error(BuildMessage(stageName, timeoutSec, elapsedSec)),
          stageName_(stageName),
          timeoutSec_(timeoutSec),
          elapsedSec_(elapsedSec) {}

    /**
     * @brief Get the name of the timed-out stage
     */
    const std::string& GetStageName() const { return stageName_; }

    /**
     * @brief Get the timeout threshold in seconds
     */
    int64_t GetTimeoutSec() const { return timeoutSec_; }

    /**
     * @brief Get the actual elapsed time in seconds
     */
    int64_t GetElapsedSec() const { return elapsedSec_; }

private:
    /**
     * @brief Build the exception message
     */
    static std::string BuildMessage(const std::string& stageName, int64_t timeoutSec, int64_t elapsedSec) {
        int64_t timeoutMin = timeoutSec / 60;
        int64_t elapsedMin = elapsedSec / 60;

        std::ostringstream oss;
        oss << "[Compiler Timeout] Stage '" << stageName << "' exceeded timeout (";
        if (timeoutMin > 0) {
            oss << timeoutMin << " min / " << timeoutSec << " sec";
        } else {
            oss << timeoutSec << " sec";
        }
        oss << "). Elapsed: ";
        if (elapsedMin > 0) {
            oss << elapsedMin << " min / " << elapsedSec << " sec";
        } else {
            oss << elapsedSec << " sec";
        }
        return oss.str();
    }

    std::string stageName_;
    int64_t timeoutSec_;
    int64_t elapsedSec_;
};

/**
 * @brief Exception thrown when monitor configuration is invalid
 */
class MonitorConfigurationException : public std::runtime_error {
public:
    explicit MonitorConfigurationException(const std::string& msg)
        : std::runtime_error("[Monitor Config Error] " + msg) {}
};

} // namespace npu::tile_fwk

#endif // SRC_INTERFACE_COMPILER_MONITOR_MONITOR_EXCEPTION_H
