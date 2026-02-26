/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_util.h"

#include <chrono>
#include <cstdio>
#include <sstream>
#include <iostream>

namespace npu::tile_fwk {

MonitorImpl::MonitorImpl(MonitorManager* manager) : manager_(manager) {}

MonitorImpl::~MonitorImpl() {
    Stop();
}

void MonitorImpl::Start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_ && thread_->joinable()) {
        return;
    }
    stop_.store(false);
    thread_ = std::make_unique<std::thread>(&MonitorImpl::MonitorLoop, this);
}

void MonitorImpl::Stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
        cv_.notify_all();
    }
    if (thread_ && thread_->joinable()) {
        thread_->join();
    }
    thread_.reset();
}

bool IsEnabledImmediate(MonitorManager* manager_) {
    return manager_->IsEnabled();
}

int GetTimeoutSecImmediate(MonitorManager* manager_) {
    int timeout_sec = manager_->GetTimeoutSec();
    if (timeout_sec <= 0) {
        if (timeout_sec == 0) {
            manager_->SetStageTimeoutFlag("Prepare");
            manager_->SetStageTimeoutFlag("Pass");
            manager_->SetStageTimeoutFlag("CodeGen");
        } else {
            timeout_sec = std::numeric_limits<int>::max();
        }
    }
    return timeout_sec;
}

int GetTotalTimeoutSecImmediate(MonitorManager* manager_) {
    int total_timeout_sec = manager_->GetTotalTimeoutSec();
    if (total_timeout_sec <= 0) {
        if (total_timeout_sec < 0) {
            total_timeout_sec = 60;
        } else {
            total_timeout_sec = 0;
            manager_->SetStageTimeoutFlag("Total");
        }
    }
    return total_timeout_sec;
}

int GetIntervalSecImmediate(MonitorManager* manager_) {
    int interval_sec = manager_->GetIntervalSec();
    if (interval_sec <= 0) {
        interval_sec = 2;
    }
    return interval_sec;
}

void MonitorImpl::MonitorLoop() {
    while (!stop_.load()) {
        // std::cout<<"ZYT ########### enable:"<<IsEnabledImmediate(manager_)<<" interval_sec:"<<GetIntervalSecImmediate(manager_)<<" timeout_sec:"<<GetTimeoutSecImmediate(manager_)<<" total_timeout_sec:"<<GetTotalTimeoutSecImmediate(manager_)<<std::endl;
        int interval_sec = GetIntervalSecImmediate(manager_);

        auto wait_duration = std::chrono::seconds(interval_sec);
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, wait_duration, [this] { return stop_.load(); });
        if (stop_.load()) {
            break;
        }
        lock.unlock();
        
        // update config options
        interval_sec = GetIntervalSecImmediate(manager_);
        int timeout_sec = GetTimeoutSecImmediate(manager_);
        int total_timeout_sec = GetTotalTimeoutSecImmediate(manager_);

        if (!IsEnabledImmediate(manager_)) {
            continue;
        }
        auto now = std::chrono::steady_clock::now();

        std::string stage = manager_->GetCurrentStageName();
        if (stage.empty()) {
            continue;
        }
        auto stage_start = manager_->GetStageStartTime();
        auto total_start = manager_->GetTotalStartTime();
        double stage_elapsed = std::chrono::duration<double>(now - stage_start).count();
        double total_elapsed = std::chrono::duration<double>(now - total_start).count();

        int total_n = manager_->GetTotalFunctionCount();
        int current_k = manager_->GetCurrentFunctionIndex();

        // check every stage cost total timeout
        std::unordered_map<std::string, double> stage_elapsed_total = manager_->GetStageElapsedTotals();
        for (const auto& [stage_tmp, sec] : stage_elapsed_total) {
            if (stage == stage_tmp) {
                if (manager_->GetStageTimeoutFlag(stage) == false) {
                    double stage_doing_elapsed_total = sec + manager_->GetCurrentStageElapsed(stage);
                    if (stage_doing_elapsed_total >= timeout_sec) {
                        manager_->SetStageTimeoutFlag(stage);
                        std::string warm_msg;
                        if (total_n > 1 && current_k > 0) {
                            warm_msg = "[Compiler Monitor] | [** WARNING **] Functions (completed): " +
                                std::to_string(current_k) + "/" + std::to_string(total_n) + " | Stage [" + stage_tmp +
                                "] total elapsed [" + FormatElapsed(stage_doing_elapsed_total) +
                                "] exceeded the current stage total time threshold [" +
                                FormatElapsed(static_cast<double>(timeout_sec)) +
                                "], you can terminate the process by pressing Ctrl+C !!!";
                        } else {
                            warm_msg = "[Compiler Monitor] | [** WARNING **] Stage [" + stage_tmp +
                                "] total elapsed [" + FormatElapsed(stage_doing_elapsed_total) +
                                "] exceeded the current stage total time threshold [" +
                                FormatElapsed(static_cast<double>(timeout_sec)) +
                                "], you can terminate the process by pressing Ctrl+C !!!";
                        }
                        (void)fprintf(stderr, "%s\n", warm_msg.c_str());
                        (void)fflush(stderr);
                    }
                }
            }
        }

        // check all stage cost total timeout
        if (total_elapsed >= static_cast<double>(total_timeout_sec) &&
            manager_->GetStageTimeoutFlag("Total") == false) {
            manager_->SetStageTimeoutFlag("Total");
            std::string warm_msg;
            warm_msg = "[Compiler Monitor] | [== WARNING ==] Total elapsed [" + FormatElapsed(total_elapsed) +
                "] exceeded the total time threshold [" + FormatElapsed(static_cast<double>(total_timeout_sec)) +
                "], you can terminate the process by pressing Ctrl+C !!!";
            (void)fprintf(stderr, "%s\n", warm_msg.c_str());
            (void)fflush(stderr);
        }

        std::string msg;
        if (total_n > 1 && current_k > 0) {
            msg = "[Compiler Monitor] Functions (completed): " + std::to_string(current_k) + "/" +
                  std::to_string(total_n) + " | Stage: " + stage + " | Stage elapsed: " +
                  FormatElapsed(stage_elapsed) + " | Total elapsed: " + FormatElapsed(total_elapsed);
        } else {
            msg = "[Compiler Monitor] Stage: " + stage + " | Number of stashed function: " + std::to_string(total_n) +
                  " | Stage elapsed: " + FormatElapsed(stage_elapsed) + " | Total elapsed: " +
                  FormatElapsed(total_elapsed);
        }
        (void)fprintf(stderr, "%s\n", msg.c_str());
        (void)fflush(stderr);
    }
}

}  // namespace npu::tile_fwk
