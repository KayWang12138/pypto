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
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
namespace {
enum class StashType {
    Function = 0,
    ProgramConfig,
    InternalConfig,
    ConfigJson
};
}

HostMachine& HostMachine::GetInstance() {
    static HostMachine sHostMachine;
    return sHostMachine;
}

/* 支持模式转换 */
bool HostMachine::Init(const HostMachineMode mode) {
    if (initialized_.load() && mode == mode_) {
        ALOG_DEBUG("HostMachine is already initialized.");
        return true;
    }
    if (mode_ == HostMachineMode::SERVER && mode == HostMachineMode::API) {
        DestroyThread();
    }
    mode_ = mode;

    if (mode == HostMachineMode::SERVER) {
        InitThread();
    }

    if (!InitBackend()) {
        return false;
    }

    initialized_.store(true);
    return true;
}

void HostMachine::Destroy() {
    this->DestroyBackend();

    if (mode_ == HostMachineMode::SERVER) {
        WaitTaskFinish();
        DestroyThread();
    }

    ALOG_DEBUG("HostMachine is destroying...");
}

bool HostMachine::ForceEnableBackend() {
    return InitBackend(true);
}

bool HostMachine::InitPassHandle() {
    if (mPassBackendHandle != nullptr) {
        return true;
    }
#ifdef ENABLE_FEATURE_PYTHON_FRONT_END
    mPassBackendHandle = dlopen(nullptr, RTLD_LAZY | RTLD_NOLOAD);
#else
    std::string passBinPath = "libtile_fwk_passes.so";
    mPassBackendHandle = dlopen(passBinPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif
    if (mPassBackendHandle == nullptr) {
        ALOG_ERROR("Fail to load pass handle, ", dlerror());
        return false;
    }

    std::string runPassFuncName = "RunPass";
    mPassRunFunc = (RunPassFuncPtr)dlsym(mPassBackendHandle, runPassFuncName.c_str());
    if (mPassRunFunc == nullptr) {
        ALOG_ERROR("Fail to get RunPass function, ", dlerror());
        return false;
    }

    std::string resumePathFuncName = "GetResumePath";
    mResumePathGetFunc = (ResumePathGetFuncPtr)dlsym(mPassBackendHandle, resumePathFuncName.c_str());
    if (mResumePathGetFunc == nullptr) {
        ALOG_ERROR("Fail to get GetResumePath function, ", dlerror());
        return false;
    }
    ALOG_INFO("Init pass handle success.");
    return true;
}

bool HostMachine::InitBackend(const bool forceEnableBackend) {
    if (!InitPassHandle()) {
        return false;
    }
    if (config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true) || forceEnableBackend) {
        if (mNpuBackendHandle != nullptr) {
            return true;
        }
        std::string binPath = "libtile_fwk_compiler.so";
        std::string funcName = "Execute";
        mNpuBackendHandle = dlopen(binPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (mNpuBackendHandle == nullptr) {
            ALOG_ERROR("AIHAC Backend enable, can't get backend binary, ", dlerror());
            return false;
        }
        mNpuExecuteFunc = (ExecuteFuncPtr)dlsym(mNpuBackendHandle, funcName.c_str());
        if (mNpuExecuteFunc == nullptr) {
            ALOG_ERROR("AIHAC Backend enable, can't get backend Execute function, ", dlerror());
            return false;
        }
        mNpuMatchCacheFunc = (MatchCacheFuncPtr)dlsym(mNpuBackendHandle, "MatchCache");
        if (mNpuMatchCacheFunc == nullptr) {
            ALOG_ERROR("AIHAC Backend enable, can't get backend MatchCache function, ", dlerror());
            return false;
        }
        mNpuInitFunc = (InitFuncPtr)dlsym(mNpuBackendHandle, "Initialize");
        if (mNpuInitFunc == nullptr) {
            ALOG_ERROR("AIHAC Backend enable, can't get backend Initialize function, ", dlerror());
            return false;
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
            return false;
        }
        mSimulationExecuteFunc = (ExecuteFuncPtr)dlsym(mSimulationBackendHandle, funcName.c_str());
        if (mSimulationExecuteFunc == nullptr) {
            ALOG_ERROR("Simulation Backend enable, can't get backend function, ", dlerror());
            return false;
        }
        ALOG_INFO("Init Simulation Backend success.");
    } else {
        ALOG_INFO("Disable Simulation Backend.");
    }
    return true;
}

void HostMachine::DestroyBackend() {
    if (mPassBackendHandle != nullptr) {
        (void)dlclose(mPassBackendHandle);
    }
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
    if (!func->HasCallOperation() && this->mPassRunFunc != nullptr) {
        ASSERT(this->mPassRunFunc(Program::GetInstance(), *func, config::GetPassStrategy())) << "Run pass failed.";
    }

    if (func->IsFunctionType(FunctionType::DYNAMIC) || func->IsFunctionTypeAndGraphType({FunctionType::STATIC}, {GraphType::TILE_GRAPH})) {
        auto path = config::GetAbsoluteTopFolder() + "/program.json";
        Program::GetInstance().DumpJsonFile(path);
        config::SetRundataOption(KEY_PROGRAM_PATH, path);
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
    stashedFuncQueue_.Push(std::make_tuple(function,
        config::Duplicate(),
        ConfigManager::Instance().GetInternalConfig(),
        ConfigManager::Instance().GetJsonData()));
}

void HostMachine::SubAllStashedTask() {
    std::lock_guard<std::mutex> lock(stashQueueMutex_);
    while (!stashedFuncQueue_.Empty()) {
        auto funcData = stashedFuncQueue_.Pop();
        config::Restore(std::get<static_cast<size_t>(StashType::ProgramConfig)>(funcData));
        ConfigManager::Instance().SetInternalConfig(std::get<static_cast<size_t>(StashType::InternalConfig)>(funcData));
        ConfigManager::Instance().SetJsonData(std::get<static_cast<size_t>(StashType::ConfigJson)>(funcData));
        SubTask(std::get<0>(funcData));
        WaitTaskFinish();
    }
}

void HostMachine::ClearStashFuncQueue() {
    stashedFuncQueue_.Clear();
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
    std::string jsonPath;
    if (this->mResumePathGetFunc != nullptr) {
        jsonPath = this->mResumePathGetFunc(config::GetPassStrategy());
    }
    bool existResumeFile = !jsonPath.empty() && (access(jsonPath.c_str(), F_OK) == 0);
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
        auto &cache = Program::GetInstance().GetFunctionCache();
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
