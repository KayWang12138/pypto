/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file host_machine.cpp
 * \brief
 */

#include "host_machine.h"
#include <functional>
#include <unistd.h>
#include <dlfcn.h>
#include "interface/configs/config_manager.h"
#include "passes/pass_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
namespace {
constexpr int64_t MACHINE_OK = 0;
enum class StashType {
    Function = 0,
    CurrFunc,
    ProgramConfig,
    TileShape,
    InternalConfig,
    ConfigJson
};
}

/* 支持模式转换 */
int HostMachine::Init(HostMachineMode mode) {
    if (initialized_.load() && mode == mode_) {
        ALOG_DEBUG("HostMachine is already initialized.");
        return MACHINE_OK;
    }
    if (mode_ == HostMachineMode::SERVER && mode == HostMachineMode::API) {
        DestroyThread();
    }
    mode_ = mode;

    if (mode == HostMachineMode::SERVER) {
        InitThread();
    }

    int32_t ret = this->InitBackend();
    if (ret != MACHINE_OK) {
        return ret;
    }

    initialized_.store(true);
    return MACHINE_OK;
}

int HostMachine::Destroy() {
    this->DestroyBackend();

    if (mode_ == HostMachineMode::SERVER) {
        WaitTaskFinish();
        DestroyThread();
    }

    ALOG_DEBUG("HostMachine is destroying...");
    return MACHINE_OK;
}

bool HostMachine::ForceEnableBackend() {
    return InitBackend(true) == MACHINE_OK;
}

int32_t HostMachine::InitBackend(const bool forceEnableBackend) {
    if (config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true) || forceEnableBackend) {
        if (mNpuBackendHandle != nullptr) {
            return MACHINE_OK;
        }
        std::string binPath = "libtile_fwk_compiler.so";
        std::string funcName = "Execute";
        mNpuBackendHandle = dlopen(binPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (mNpuBackendHandle == nullptr) {
            ALOG_ERROR("AIHAC Backend enable, can't get backend binary, ", dlerror());
            return 1;
        }
        mNpuExecuteFunc = (ExecuteFuncPtr)dlsym(mNpuBackendHandle, funcName.c_str());
        if (mNpuExecuteFunc == nullptr) {
            ALOG_ERROR("AIHAC Backend enable, can't get backend Execute function, ", dlerror());
            return 1;
        }
        mNpuMatchCacheFunc = (MatchCacheFuncPtr)dlsym(mNpuBackendHandle, "MatchCache");
        if (mNpuMatchCacheFunc == nullptr) {
            ALOG_ERROR("AIHAC Backend enable, can't get backend MatchCache function, ", dlerror());
            return 1;
        }
        mNpuInitFunc = (InitFuncPtr)dlsym(mNpuBackendHandle, "Initialize");
        if (mNpuInitFunc == nullptr) {
            ALOG_ERROR("AIHAC Backend enable, can't get backend Initialize function, ", dlerror());
            return 1;
        }
        mNpuInitFunc();
        ALOG_INFO("Init AIHAC Backend success.");
    } else {
        ALOG_INFO("Disable AIHAC Backend.");
    }
    if (config::GetPlatformConfig(KEY_ENABLE_COST_MODEL, true)) {
        std::string binPath = "libtile_fwk_simulation.so";
        std::string funcName = "ExecuteSimulation";
        mSimulationBackendHandle = dlopen(binPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (mSimulationBackendHandle == nullptr) {
            ALOG_ERROR("Simulation Backend enable, can't get backend binary, ", dlerror());
            return 1;
        }
        mSimulationExecuteFunc = (ExecuteFuncPtr)dlsym(mSimulationBackendHandle, funcName.c_str());
        if (mSimulationExecuteFunc == nullptr) {
            ALOG_ERROR("Simulation Backend enable, can't get backend function, ", dlerror());
            return 1;
        }
        ALOG_INFO("Init Simulation Backend success.");
    } else {
        ALOG_INFO("Disable Simulation Backend.");
    }
    return MACHINE_OK;
}

void HostMachine::DestroyBackend() {
    if (mSimulationBackendHandle != nullptr) {
        (void)dlclose(mSimulationBackendHandle);
    }
    if (mNpuBackendHandle != nullptr) {
        (void)dlclose(mNpuBackendHandle);
    }
}

void HostMachine::InitThread() {
    stopFlag_.store(false);
    for (int i = 0; i < compileThreadCount_; ++i) {
        compileThreads_.emplace_back(&HostMachine::CompileThreadFunc, this);
    }
    for (int i = 0; i < agentThreadCount_; ++i) {
        agentThreads_.emplace_back(&HostMachine::AgentThreadFunc, this);
    }
}

void HostMachine::DestroyThread() {
    auto notifyThread = [this](std::mutex &mutex, std::condition_variable &cv) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.notify_all();
    };
    stopFlag_.store(true);
    notifyThread(compileQueueMutex_, compileQueueCv_);
    for (auto &thread : compileThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    notifyThread(agentQueueMutex_, agentQueueCv_);
    for (auto &thread : agentThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    compileThreads_.clear();
    agentThreads_.clear();
}

void HostMachine::CompileFunction(Function* func) const {
    if (!func->HasCallOperation()) {
        auto &pm = PassManager::Instance();
        ASSERT(pm.RunPass(Program::GetInstance(), *func, config::GetPassStrategy()) == SUCCESS) << "Run pass failed.";
    }

    if (func->IsFunctionType(FunctionType::DYNAMIC) || func->IsFunctionTypeAndGraphType({FunctionType::STATIC}, {GraphType::TILE_GRAPH})) {
        Program::GetInstance().DumpJsonFile(config::LogTopFolder() + "/program.json");
    }
    if (func->rootFunc_ != nullptr) {
        func->rootFunc_->DumpTopoFile(config::LogTopFolder() + "/topo.json");
    }
}

void HostMachine::SubTask(Function *function) {
    if (mode_ == HostMachineMode::API) {
        MACHINE_ASSERT(curTask == nullptr);
        curTask = new MachineTask(curTaskId_++, function);
        return;
    }

    std::lock_guard<std::mutex> lock(compileQueueMutex_);
    auto task = std::make_unique<MachineTask>(curTaskId_++, function);
    compileQueue_.Push(std::move(task));
    compileQueueCv_.notify_one(); // 通知编译线程
}

void HostMachine::WaitTaskFinish() {
    while (curTaskId_ != finishQueue_.Size()) {
    } // wait all task finish
    ALOG_DEBUG("Finish all host machine task count: %lu", curTaskId_);

    /* reset counter */
    curTaskId_ = 0;
    finishQueue_.Clear();
}

void HostMachine::StashTask(Function* function) {
    if (function == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(stashQueueMutex_);
    stashedFuncQueue_.Push(std::make_tuple(function, Program::GetInstance().GetCurrentFunction(),
        Program::GetInstance().GetConfig(), Program::GetInstance().GetTileShape(),
        ConfigManager::Instance().GetInternalConfig(), ConfigManager::Instance().GetJsonData()));
}

void HostMachine::SubAllStashedTask() {
    std::lock_guard<std::mutex> lock(stashQueueMutex_);
    while (!stashedFuncQueue_.Empty()) {
        auto funcData = stashedFuncQueue_.Pop();
        Program::GetInstance().SetCurrentFunction(std::get<static_cast<size_t>(StashType::CurrFunc)>(funcData));
        Program::GetInstance().SetConfig(std::get<static_cast<size_t>(StashType::ProgramConfig)>(funcData));
        Program::GetInstance().SetTileShape(std::get<static_cast<size_t>(StashType::TileShape)>(funcData));
        ConfigManager::Instance().SetInternalConfig(std::get<static_cast<size_t>(StashType::InternalConfig)>(funcData));
        ConfigManager::Instance().SetJsonData(std::get<static_cast<size_t>(StashType::ConfigJson)>(funcData));
        SubTask(std::get<0>(funcData));
        WaitTaskFinish();
    }
}

std::string HostMachine::GetCacheKeyFromFunction(Function *function) {
    std::string cacheKey;
    if (function == nullptr) {
        return cacheKey;
    }
    if (function->BelongTo().GetLastFunction() != nullptr &&
        function->BelongTo().GetLastFunction()->GetFunctionType() == FunctionType::DYNAMIC) {
        cacheKey = function->BelongTo().GetLastFunction()->GetFunctionHash().Data();
    } else {
        cacheKey = function->GetFunctionHash().Data();
    }
    return cacheKey;
}

MachineTask *HostMachine::Compile(MachineTask *task) const {
    MachineTask *compileTask = task;
    if (compileTask == nullptr) {
        MACHINE_ASSERT(curTask != nullptr);
        compileTask = curTask;
    }

    auto &pm = PassManager::Instance();
    auto jsonPath = pm.GetResumePath(config::GetPassStrategy());
    bool existResumeFile = false;
    if (jsonPath != "") {
        if (access(jsonPath.c_str(), F_OK) == 0) {
            existResumeFile = true;
        }
    }
    if (existResumeFile) {
        std::ifstream file(jsonPath);
        ASSERT(file.good()) << "Json file: " << jsonPath << " open failed!!!";
        Json jsonData;
        try {
            file >> jsonData;
        } catch (const std::exception &e) {
            ASSERT(false) << "Json file: " << jsonPath << " parsing error: " << e.what();
        }
        Program::GetInstance().LoadJson(jsonData);
        Function *func = Program::GetInstance().GetCurrentFunction();

        CompileFunction(func);
        compileTask->SetFunction(func);
    } else {
        auto function = compileTask->GetFunction();
        compileTask->SetCacheKey(GetCacheKeyFromFunction(function));
        if (mNpuMatchCacheFunc != nullptr && mNpuMatchCacheFunc(compileTask->GetCacheKey())) {
            compileTask->SetCacheReuseType(CacheReuseType::Bin);
        } else {
            CompileFunction(function);
        }
    }

    return compileTask;
}

void HostMachine::PushAgentQueue(std::unique_ptr<MachineTask> task) {
    std::lock_guard<std::mutex> lock(agentQueueMutex_);
    agentQueue_.Push(std::move(task));
    agentQueueCv_.notify_one(); // 通知代理线程
}

void HostMachine::CompileThreadFunc() {
    while (!stopFlag_.load()) {
        std::unique_ptr<MachineTask> task;
        std::unique_lock<std::mutex> lock(compileQueueMutex_);
        compileQueueCv_.wait(lock, [this] { return !compileQueue_.Empty() || stopFlag_.load(); });
        if (stopFlag_.load()) {
            break;
        }
        task = compileQueue_.Pop();
        lock.unlock();
        (void)Compile(task.get());
        PushAgentQueue(std::move(task));
    }
}

void HostMachine::PushFinishQueue(std::unique_ptr<MachineTask> task) {
    finishQueue_.Push(std::move(task));
}

void HostMachine::AgentThreadFunc() {
    while (!stopFlag_.load()) {
        std::unique_ptr<MachineTask> task;
        std::unique_lock<std::mutex> lock(agentQueueMutex_);
        agentQueueCv_.wait(lock, [this] { return !agentQueue_.Empty() || stopFlag_.load(); });
        if (stopFlag_.load()) {
            break;
        }
        task = agentQueue_.Pop();
        lock.unlock();
        auto &cache = this->GetFunctionCache();
        if (this->mSimulationExecuteFunc != nullptr) {
            this->mSimulationExecuteFunc(task.get(), cache);
        }
        if (this->mNpuExecuteFunc != nullptr) {
            this->mNpuExecuteFunc(task.get(), cache);
        }
        PushFinishQueue(std::move(task));
    }
}
} // namespace npu::tile_fwk
