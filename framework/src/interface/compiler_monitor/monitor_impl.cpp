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

void MonitorImpl::MonitorLoop() {
    bool checkEnable = manager_->IsEnabled();
    int timeout_sec = manager_->GetTimeoutSec();
    if (timeout_sec <= 0) {
        timeout_sec = 60;
    }
    int total_timeout_sec = manager_->GetTotalTimeoutSec();
    if (total_timeout_sec <= 0) {
        total_timeout_sec = 120;
    }
    int interval_sec = manager_->GetIntervalSec();
    if (interval_sec <= 0) {
        interval_sec = 30;
    }
    // std::cout<<"ZYT ########### [pypto.set_compiler_monitor_options] enable:"<<checkEnable<<" interval_sec:"<<interval_sec<<" timeout_sec:"<<timeout_sec<<" total_timeout_sec:"<<total_timeout_sec<<std::endl;
    auto wait_duration = std::chrono::seconds(interval_sec);
    while (!stop_.load()) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, wait_duration, [this] { return stop_.load(); });
        if (stop_.load()) {
            break;
        }
        lock.unlock();

        if (!checkEnable) {
            continue;
        }
        auto now = std::chrono::steady_clock::now();
        auto last_print = manager_->GetLastPrintTime();
        auto elapsed_since_print = std::chrono::duration<double>(now - last_print).count();
        if (elapsed_since_print < static_cast<double>(interval_sec)) {
            // continue;
        }

        std::string stage = manager_->GetCurrentStageName();
        if (stage.empty()) {
            continue;
        }
        auto stage_start = manager_->GetStageStartTime();
        auto total_start = manager_->GetTotalStartTime();
        double stage_elapsed = std::chrono::duration<double>(now - stage_start).count();
        double total_elapsed = std::chrono::duration<double>(now - total_start).count();

        std::map<std::string, double> stage_elapsed_total = manager_->GetStageElapsedTotals();
        for (const auto& [stage_tmp, sec] : stage_elapsed_total) {
            if (stage == stage_tmp) {
                if (manager_->GetStageTimeoutFlag(stage) == false) {
                    double stage_doing_elapsed_total = sec + manager_->GetCurrentStageElapsed(stage);
                    if (stage_doing_elapsed_total >= timeout_sec) {
                        manager_->SetStageTimeoutFlag(stage);
                        std::string warm_msg;
                        warm_msg = "[** WARNING **] Stage [" + stage_tmp + "] total elapsed [" +
                            FormatElapsed(stage_doing_elapsed_total) +
                            "] exceeded the current stage total time threshold [" +
                            FormatElapsed(static_cast<double>(timeout_sec)) +
                            "], you can terminate the process by pressing Ctrl+C !!!";
                        (void)fprintf(stderr, "%s\n", warm_msg.c_str());
                        (void)fflush(stderr);
                    }
                }
            }
        }

        if (total_elapsed >= static_cast<double>(total_timeout_sec) && manager_->GetStageTimeoutFlag("Total") == false) {
            manager_->SetStageTimeoutFlag("Total");
            std::string warm_msg;
            warm_msg = "[== WARNING ==] Total elapsed [" + FormatElapsed(total_elapsed) +
                "] exceeded the total time threshold [" + FormatElapsed(static_cast<double>(total_timeout_sec)) +
                "], you can terminate the process by pressing Ctrl+C !!!";
            (void)fprintf(stderr, "%s\n", warm_msg.c_str());
            (void)fflush(stderr);
        }

        manager_->SetLastPrintTime(now);

        int total_n = manager_->GetTotalFunctionCount();
        int current_k = manager_->GetCurrentFunctionIndex();
        std::string msg;
        if (total_n > 1 && current_k > 0) {
            msg = "[Compiler Monitor] Functions (completed): " + std::to_string(current_k) + "/" +
                  std::to_string(total_n) + " | Stage: " + stage + " | Stage elapsed: " +
                  FormatElapsed(stage_elapsed) + " | Total elapsed: " + FormatElapsed(total_elapsed);
        } else {
            msg = "[Compiler Monitor] Stage: " + stage + " | Stage elapsed: " +
                  FormatElapsed(stage_elapsed) + " | Total elapsed: " + FormatElapsed(total_elapsed);
        }
        (void)fprintf(stderr, "%s\n", msg.c_str());
        (void)fflush(stderr);
    }
}

}  // namespace npu::tile_fwk
