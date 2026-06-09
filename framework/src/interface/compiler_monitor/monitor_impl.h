/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace npu::tile_fwk {

class MonitorManager;

class MonitorImpl {
public:
    explicit MonitorImpl(MonitorManager* manager);
    ~MonitorImpl();

    void Start();
    void Stop();
    void StartMonitoring();
    void StopMonitoring();

private:
    void MonitorLoop();
    void PrintTotalTimeOut(double total_elapsed, int total_timeout_sec);

    MonitorManager* manager_;
    std::unique_ptr<std::thread> thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> stage_start_flag_{false};
};

} // namespace npu::tile_fwk
