/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file monitor_manager.cpp
 * \brief Implementation of compiler monitor manager
 */

#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_config.h"
#include "interface/utils/log.h"
#include "interface/compiler_monitor/monitor_exception.h"
#include <atomic>

namespace npu::tile_fwk {

// Static singleton instance
MonitorManager& MonitorManager::Instance() {
    static MonitorManager instance;
    static std::once_flag initFlag;
    std::call_once(initFlag, []() {
        instance.Initialize();
    });
    return instance;
}

MonitorManager::MonitorManager()
    : enable_(true),  // Default enabled
      initialized_(false) {
    ALOG_INFO("MonitorManager: Created (default enable=%d)", enable_);
}

MonitorManager::~MonitorManager() {
    if (initialized_) {
        Shutdown();
    }
    ALOG_INFO("MonitorManager: Destroyed");
}

void MonitorManager::SetOptions(bool enable, int intervalSec, int timeoutSec, int totalTimeoutSec, TimeoutAction timeoutAction) {
    std::unique_ptr<MonitorImpl> newImpl;
    bool needStart = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        bool wasEnabled = enable_;
        bool willBeEnabled = enable;

        config_.enable = enable;
        config_.intervalSec = intervalSec;
        config_.timeoutSec = timeoutSec;
        config_.totalTimeoutSec = totalTimeoutSec;
        config_.timeoutAction = timeoutAction;

        ALOG_INFO("MonitorManager::SetOptions: enable=%d, interval=%ds, timeout=%ds, total_timeout=%ds, action=%d",
                  enable, intervalSec, timeoutSec, totalTimeoutSec, static_cast<int>(timeoutAction));

        // If already initialized and was enabled, stop old implementation
        if (initialized_) {
            if (impl_) {
                impl_->Stop();
                impl_.reset();
            }
            // Update enable flag based on config
            enable_ = config_.enable;

            // If still enabled, restart with new config
            if (config_.enable) {
                newImpl = std::make_unique<MonitorImpl>(
                    std::chrono::seconds(config_.intervalSec),
                    std::chrono::seconds(config_.timeoutSec),
                    std::chrono::seconds(config_.totalTimeoutSec),
                    config_.timeoutAction
                );
                needStart = true;
            }
        }
    }

    // Start the monitoring thread outside the lock to avoid deadlock
    if (needStart && newImpl) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            impl_ = std::move(newImpl);
        }
        impl_->Start();  // Start the monitoring thread
    }
}

void MonitorManager::Initialize() {
    std::unique_ptr<MonitorImpl> newImpl;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (initialized_) {
            return;  // Already initialized
        }

        ALOG_INFO("MonitorManager::Initialize: enable=%d, interval=%ds, timeout=%ds, total_timeout=%ds",
                  config_.enable, config_.intervalSec, config_.timeoutSec, config_.totalTimeoutSec);

        if (config_.enable) {
            newImpl = std::make_unique<MonitorImpl>(
                std::chrono::seconds(config_.intervalSec),
                std::chrono::seconds(config_.timeoutSec),
                std::chrono::seconds(config_.totalTimeoutSec),
                config_.timeoutAction
            );
            enable_ = true;
        } else {
            enable_ = false;
        }

        initialized_ = true;
    }

    // Start the monitoring thread outside the lock to avoid deadlock
    if (newImpl) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            impl_ = std::move(newImpl);
        }
        impl_->Start();  // Start the monitoring thread
    }
}

void MonitorManager::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        ALOG_WARN("MonitorManager not initialized, nothing to shutdown");
        return;
    }

    ALOG_INFO("MonitorManager::Shutdown");

    if (impl_) {
        impl_->Stop();
        impl_.reset();
    }

    enable_ = false;
    initialized_ = false;

    ALOG_INFO("MonitorManager shutdown complete");
}

void MonitorManager::StartStage(const std::string& stageName) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !enable_ || !impl_) {
        return;
    }

    impl_->StartStage(stageName);
}

void MonitorManager::EndStage() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || !enable_ || !impl_) {
        return;
    }

    // Call impl_->EndStage() to print elapsed time for coarse-grained stages
    impl_->EndStage();

    // Note: We don't stop timing here - the monitor continues
    // to track the last stage until a new one starts or Shutdown is called
    ALOG_INFO("[Compiler Monitor] EndStage called (monitoring continues)");
}

// RAII helper implementation
static std::atomic<int> gNextStageId(0);

MonitorStageScope::MonitorStageScope(const std::string& stageName)
    : stageName_(stageName),
      active_(false),
      stageId_(gNextStageId.fetch_add(1)) {
    active_.store(true);
    MonitorManager::Instance().StartStage(stageName_);
}

MonitorStageScope::~MonitorStageScope() {
    if (active_.exchange(false)) {
        MonitorManager::Instance().EndStage();
    }
}

MonitorStageScope::MonitorStageScope(MonitorStageScope&& other) noexcept
    : stageName_(std::move(other.stageName_)),
      active_(false),  // Always false for moved-from objects
      stageId_(other.stageId_.load()) {}

MonitorStageScope& MonitorStageScope::operator=(MonitorStageScope&& other) noexcept {
    if (this != &other) {
        if (active_.load()) {
            active_.store(false);
            MonitorManager::Instance().EndStage();
        }
        stageName_ = std::move(other.stageName_);
        active_ = false;  // After move, this takes ownership
        stageId_ = other.stageId_.load();
    }
    return *this;
}

} // namespace npu::tile_fwk
