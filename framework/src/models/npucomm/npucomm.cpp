/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npucomm.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>

namespace npu {
namespace comm {

// 静态成员初始化
std::unique_ptr<NpuCommContext> NpuComm::context_;
std::unique_ptr<NpuCommEngine> NpuComm::engine_;
std::unique_ptr<NpuFineGrainComm> NpuComm::fineComm_;
std::unique_ptr<NpuMemoryConsistency> NpuComm::memory_;
std::unique_ptr<NpuCommTeam> NpuComm::team_;

// NpuCommContext实现
NpuCommContext::NpuCommContext(const NpuCommConfig& config) : config_(config) {
}

NpuCommContext::~NpuCommContext() {
    if (initialized_) {
        finalize();
    }
}

CommStatus NpuCommContext::init() {
    if (initialized_) {
        return COMM_SUCCESS;
    }

    std::cout << "[NPUCOMM] Initializing PGAS context for PE " << config_.myPe
              << " of " << config_.numPes << " PEs" << std::endl;
    std::cout << "[NPUCOMM] Symmetric heap size: " << config_.symmetricHeapSize << " bytes" << std::endl;
    std::cout << "[NPUCOMM] NPU optimization: " << (config_.enableNpuOptimization ? "enabled" : "disabled") << std::endl;
    std::cout << "[NPUCOMM] Fine-grain communication: " << (config_.enableFineGrainComm ? "enabled" : "disabled") << std::endl;
    std::cout << "[NPUCOMM] PGAS mode: " << (config_.enablePgasMode ? "enabled" : "disabled") << std::endl;

    // 初始化对称堆空间
    try {
        symmetricHeap_.reserve(1024); // 预分配空间
        initialized_ = true;
        std::cout << "[NPUCOMM] PGAS context initialized successfully" << std::endl;
        return COMM_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "[NPUCOMM] Failed to initialize context: " << e.what() << std::endl;
        return COMM_ERROR;
    }
}

CommStatus NpuCommContext::finalize() {
    if (!initialized_) {
        return COMM_SUCCESS;
    }

    std::cout << "[NPUCOMM] Finalizing PGAS context for PE " << config_.myPe << std::endl;

    // 清理对称堆
    for (auto ptr : symmetricHeap_) {
        if (ptr) {
            free(ptr);
        }
    }
    symmetricHeap_.clear();
    allocationMap_.clear();
    allocatedSize_ = 0;
    initialized_ = false;

    std::cout << "[NPUCOMM] PGAS context finalized successfully" << std::endl;
    return COMM_SUCCESS;
}

void* NpuCommContext::shmalloc(size_t size) {
    if (!initialized_) {
        std::cerr << "[NPUCOMM] Context not initialized" << std::endl;
        return nullptr;
    }

    if (allocatedSize_ + size > config_.symmetricHeapSize) {
        std::cerr << "[NPUCOMM] Symmetric heap exhausted" << std::endl;
        return nullptr;
    }

    void* ptr = nullptr;
    if (config_.enableNpuOptimization) {
        // 尝试从NPU内存分配
        ptr = allocNpuMemory(size);
    }

    if (!ptr) {
        // 回退到标准内存分配
        ptr = malloc(size);
    }

    if (ptr) {
        symmetricHeap_.push_back(ptr);
        allocationMap_[ptr] = size;
        allocatedSize_ += size;
        std::cout << "[NPUCOMM] Allocated " << size << " bytes in symmetric heap" << std::endl;
    }

    return ptr;
}

CommStatus NpuCommContext::shfree(void* ptr) {
    if (!ptr) return COMM_SUCCESS;

    auto it = std::find(symmetricHeap_.begin(), symmetricHeap_.end(), ptr);
    if (it != symmetricHeap_.end()) {
        auto alloc_it = allocationMap_.find(ptr);
        if (alloc_it != allocationMap_.end()) {
            allocatedSize_ -= alloc_it->second;
            allocationMap_.erase(alloc_it);
        }

        if (isNpuMemory(ptr)) {
            freeNpuMemory(ptr);
        } else {
            free(ptr);
        }

        symmetricHeap_.erase(it);
        std::cout << "[NPUCOMM] Freed symmetric heap allocation" << std::endl;
        return COMM_SUCCESS;
    }

    return COMM_INVALID_ADDR;
}

bool NpuCommContext::isSymmetricAddress(const void* addr) const {
    return std::find(symmetricHeap_.begin(), symmetricHeap_.end(), addr) != symmetricHeap_.end();
}

void* NpuCommContext::globalToLocal(void* global_addr, int pe) const {
    // 简化的全局到本地地址转换
    // 在真正的PGAS系统中，这需要更复杂的地址转换逻辑
    if (pe == config_.myPe) {
        return global_addr;
    }
    // 对于远程PE，返回nullptr表示需要通过通信访问
    return nullptr;
}

void* NpuCommContext::localToGlobal(void* local_addr, int pe) const {
    // 简化的本地到全局地址转换
    if (pe == config_.myPe && isSymmetricAddress(local_addr)) {
        return local_addr;
    }
    return nullptr;
}

// NPU内存管理辅助函数
void* NpuCommContext::allocNpuMemory(size_t size) {
    // 模拟NPU内存分配
    // 在实际实现中，应该调用NPU的内存分配API
    if (config_.enableNpuOptimization) {
        void* ptr = malloc(size);
        if (ptr) {
            std::cout << "[NPUCOMM] Allocated NPU memory: " << size << " bytes" << std::endl;
        }
        return ptr;
    }
    return nullptr;
}

void NpuCommContext::freeNpuMemory(void* ptr) {
    if (config_.enableNpuOptimization && ptr) {
        std::cout << "[NPUCOMM] Freed NPU memory" << std::endl;
        free(ptr);
    }
}

bool NpuCommContext::isNpuMemory(void* ptr) const {
    // 简化的NPU内存检测
    // 在实际实现中，需要更精确的检测机制
    return config_.enableNpuOptimization;
}

// NpuCommEngine实现
NpuCommEngine::NpuCommEngine(NpuCommContext* context) : context_(context) {
    sendBuffer_.resize(1024 * 1024); // 1MB buffer
    recvBuffer_.resize(1024 * 1024);
}

NpuCommEngine::~NpuCommEngine() {
}

CommStatus NpuCommEngine::initialize() {
    std::cout << "[NPUCOMM] Initializing NPU communication engine" << std::endl;

    if (!context_ || !context_->getConfig().enableNpuOptimization) {
        std::cout << "[NPUCOMM] Using standard communication (NPU optimization disabled)" << std::endl;
        return COMM_SUCCESS;
    }

    std::cout << "[NPUCOMM] NPU communication engine initialized with "
              << context_->getConfig().maxConcurrentOperations << " concurrent operations" << std::endl;

    return COMM_SUCCESS;
}

CommStatus NpuCommEngine::put(void* dest, const void* src, size_t size, int pe, SyncMode sync) {
    if (pe == context_->myPe()) {
        // 本地复制
        memcpy(dest, src, size);
        updateStats(size, 1);
        return COMM_SUCCESS;
    }

    CommStatus status;
    if (context_->getConfig().enableNpuOptimization && size <= context_->getConfig().maxMessageSize) {
        status = optimizeForNpu(dest, src, size, pe, true);
    } else {
        status = fallbackToStandard(dest, src, size, pe, true);
    }

    if (status == COMM_SUCCESS) {
        updateStats(size, 1);

        // 处理同步
        switch (sync) {
            case SYNC_FENCE:
                return fence();
            case SYNC_QUIET:
                return quiet();
            case SYNC_BARRIER:
                return barrier_all();
            default:
                break;
        }
    }

    return status;
}

CommStatus NpuCommEngine::get(void* dest, const void* src, size_t size, int pe, SyncMode sync) {
    if (pe == context_->myPe()) {
        // 本地复制
        memcpy(dest, src, size);
        updateStats(size, 1);
        return COMM_SUCCESS;
    }

    CommStatus status;
    if (context_->getConfig().enableNpuOptimization && size <= context_->getConfig().maxMessageSize) {
        status = optimizeForNpu(dest, src, size, pe, false);
    } else {
        status = fallbackToStandard(dest, src, size, pe, false);
    }

    if (status == COMM_SUCCESS) {
        updateStats(size, 1);

        // 处理同步
        switch (sync) {
            case SYNC_FENCE:
                return fence();
            case SYNC_QUIET:
                return quiet();
            case SYNC_BARRIER:
                return barrier_all();
            default:
                break;
        }
    }

    return status;
}

CommStatus NpuCommEngine::put_nbi(void* dest, const void* src, size_t size, int pe) {
    return put(dest, src, size, pe, SYNC_NONE);
}

CommStatus NpuCommEngine::get_nbi(void* dest, const void* src, size_t size, int pe) {
    return get(dest, src, size, pe, SYNC_NONE);
}

int64_t NpuCommEngine::atomic_fetch_add(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        int64_t old_value = *dest;
        *dest += value;
        return old_value;
    }

    // 远程原子操作 (模拟)
    std::cout << "[NPUCOMM] Atomic fetch-add: " << value << " to PE " << pe << std::endl;

    // 实际实现应该使用NPU的原子操作硬件支持
    int64_t old_value = 0; // 模拟返回值
    return old_value;
}

int64_t NpuCommEngine::atomic_compare_swap(int64_t* dest, int64_t compare, int64_t swap, int pe) {
    if (pe == context_->myPe()) {
        int64_t old_value = *dest;
        if (*dest == compare) {
            *dest = swap;
        }
        return old_value;
    }

    // 远程原子操作 (模拟)
    std::cout << "[NPUCOMM] Atomic compare-swap: " << compare << " -> " << swap << " on PE " << pe << std::endl;

    int64_t old_value = compare; // 模拟返回值
    return old_value;
}

int64_t NpuCommEngine::atomic_fetch_and(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        int64_t old_value = *dest;
        *dest &= value;
        return old_value;
    }

    std::cout << "[NPUCOMM] Atomic fetch-and: " << value << " to PE " << pe << std::endl;
    return 0; // 模拟返回值
}

int64_t NpuCommEngine::atomic_fetch_or(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        int64_t old_value = *dest;
        *dest |= value;
        return old_value;
    }

    std::cout << "[NPUCOMM] Atomic fetch-or: " << value << " to PE " << pe << std::endl;
    return 0; // 模拟返回值
}

void NpuCommEngine::atomic_add(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        *dest += value;
        return;
    }

    std::cout << "[NPUCOMM] Atomic add: " << value << " to PE " << pe << std::endl;
}

void NpuCommEngine::atomic_and(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        *dest &= value;
        return;
    }

    std::cout << "[NPUCOMM] Atomic and: " << value << " to PE " << pe << std::endl;
}

void NpuCommEngine::atomic_or(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        *dest |= value;
        return;
    }

    std::cout << "[NPUCOMM] Atomic or: " << value << " to PE " << pe << std::endl;
}

int64_t NpuCommEngine::atomic_fetch_xor(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        int64_t old_value = *dest;
        *dest ^= value;
        return old_value;
    }

    std::cout << "[NPUCOMM] Atomic fetch-xor: " << value << " to PE " << pe << std::endl;
    return 0; // 模拟返回值
}

void NpuCommEngine::atomic_xor(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        *dest ^= value;
        return;
    }

    std::cout << "[NPUCOMM] Atomic xor: " << value << " to PE " << pe << std::endl;
}

int64_t NpuCommEngine::atomic_swap(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        int64_t old_value = *dest;
        *dest = value;
        return old_value;
    }

    std::cout << "[NPUCOMM] Atomic swap: " << value << " to PE " << pe << std::endl;
    return 0; // 模拟返回值
}

int64_t NpuCommEngine::atomic_fetch_inc(int64_t* dest, int pe) {
    return atomic_fetch_add(dest, 1, pe);
}

void NpuCommEngine::atomic_inc(int64_t* dest, int pe) {
    atomic_add(dest, 1, pe);
}

int64_t NpuCommEngine::atomic_fetch_set(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        int64_t old_value = *dest;
        *dest = value;
        return old_value;
    }

    std::cout << "[NPUCOMM] Atomic fetch-set: " << value << " to PE " << pe << std::endl;
    return 0; // 模拟返回值
}

void NpuCommEngine::atomic_set(int64_t* dest, int64_t value, int pe) {
    if (pe == context_->myPe()) {
        *dest = value;
        return;
    }

    std::cout << "[NPUCOMM] Atomic set: " << value << " to PE " << pe << std::endl;
}

CommStatus NpuCommEngine::fence() {
    std::cout << "[NPUCOMM] Memory fence operation" << std::endl;
    // 实际实现应该执行内存栅栏
    std::atomic_thread_fence(std::memory_order_acq_rel);
    return COMM_SUCCESS;
}

CommStatus NpuCommEngine::quiet() {
    std::cout << "[NPUCOMM] Quiet operation - waiting for all communications to complete" << std::endl;
    // 等待所有未完成的通信操作
    while (activeOps_.load() > 0) {
        std::this_thread::yield();
    }
    return COMM_SUCCESS;
}

CommStatus NpuCommEngine::barrier(int PE_start, int logPE_stride, int PE_size) {
    std::cout << "[NPUCOMM] Barrier synchronization: PE_start=" << PE_start
              << ", stride=" << logPE_stride << ", size=" << PE_size << std::endl;

    // 实际实现应该使用NPU的硬件同步原语
    std::this_thread::sleep_for(std::chrono::microseconds(100)); // 模拟同步延迟

    return COMM_SUCCESS;
}

CommStatus NpuCommEngine::barrier_all() {
    return barrier(0, 0, context_->numPes());
}

CommStatus NpuCommEngine::device_put(void* dest, const void* src, size_t size, int pe) {
    std::cout << "[NPUCOMM] Device-initiated put: " << size << " bytes to PE " << pe << std::endl;

    // NPU设备端发起的通信
    // 实际实现应该在NPU内核中发起通信
    return put(dest, src, size, pe);
}

CommStatus NpuCommEngine::device_get(void* dest, const void* src, size_t size, int pe) {
    std::cout << "[NPUCOMM] Device-initiated get: " << size << " bytes from PE " << pe << std::endl;

    // NPU设备端发起的通信
    return get(dest, src, size, pe);
}

// 类型化put/get操作实现
void NpuCommEngine::put8(void* dest, const void* src, size_t nelems, int pe) {
    put(dest, src, nelems * sizeof(int8_t), pe);
}

void NpuCommEngine::put16(void* dest, const void* src, size_t nelems, int pe) {
    put(dest, src, nelems * sizeof(int16_t), pe);
}

void NpuCommEngine::put32(void* dest, const void* src, size_t nelems, int pe) {
    put(dest, src, nelems * sizeof(int32_t), pe);
}

void NpuCommEngine::put64(void* dest, const void* src, size_t nelems, int pe) {
    put(dest, src, nelems * sizeof(int64_t), pe);
}

void NpuCommEngine::put128(void* dest, const void* src, size_t nelems, int pe) {
    put(dest, src, nelems * 16, pe); // 128-bit = 16 bytes
}

void NpuCommEngine::get8(void* dest, const void* src, size_t nelems, int pe) {
    get(dest, src, nelems * sizeof(int8_t), pe);
}

void NpuCommEngine::get16(void* dest, const void* src, size_t nelems, int pe) {
    get(dest, src, nelems * sizeof(int16_t), pe);
}

void NpuCommEngine::get32(void* dest, const void* src, size_t nelems, int pe) {
    get(dest, src, nelems * sizeof(int32_t), pe);
}

void NpuCommEngine::get64(void* dest, const void* src, size_t nelems, int pe) {
    get(dest, src, nelems * sizeof(int64_t), pe);
}

void NpuCommEngine::get128(void* dest, const void* src, size_t nelems, int pe) {
    get(dest, src, nelems * 16, pe); // 128-bit = 16 bytes
}

// 单元素操作实现
void NpuCommEngine::p8(void* dest, int8_t value, int pe) {
    put(dest, &value, sizeof(int8_t), pe);
}

void NpuCommEngine::p16(void* dest, int16_t value, int pe) {
    put(dest, &value, sizeof(int16_t), pe);
}

void NpuCommEngine::p32(void* dest, int32_t value, int pe) {
    put(dest, &value, sizeof(int32_t), pe);
}

void NpuCommEngine::p64(void* dest, int64_t value, int pe) {
    put(dest, &value, sizeof(int64_t), pe);
}

int8_t NpuCommEngine::g8(const void* src, int pe) {
    int8_t value;
    get(&value, src, sizeof(int8_t), pe);
    return value;
}

int16_t NpuCommEngine::g16(const void* src, int pe) {
    int16_t value;
    get(&value, src, sizeof(int16_t), pe);
    return value;
}

int32_t NpuCommEngine::g32(const void* src, int pe) {
    int32_t value;
    get(&value, src, sizeof(int32_t), pe);
    return value;
}

int64_t NpuCommEngine::g64(const void* src, int pe) {
    int64_t value;
    get(&value, src, sizeof(int64_t), pe);
    return value;
}

// 非连续操作实现
void NpuCommEngine::iput32(void* dest, const void* src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {
    // 简化的非连续put实现
    // 实际应该处理步长(dst/sst)
    put(dest, src, nelems * sizeof(int32_t), pe);
    std::cout << "[NPUCOMM] Non-contiguous put32: " << nelems << " elements, dst_stride=" << dst << ", src_stride=" << sst << std::endl;
}

void NpuCommEngine::iget32(void* dest, const void* src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {
    // 简化的非连续get实现
    get(dest, src, nelems * sizeof(int32_t), pe);
    std::cout << "[NPUCOMM] Non-contiguous get32: " << nelems << " elements, dst_stride=" << dst << ", src_stride=" << sst << std::endl;
}

// 等待和测试操作实现
void NpuCommEngine::wait_until(void* addr, CmpOp cmp, int64_t value) {
    volatile int64_t* ptr = static_cast<volatile int64_t*>(addr);

    std::cout << "[NPUCOMM] Wait until operation, cmp=" << cmp << ", value=" << value << std::endl;

    // 简化的等待实现
    switch (cmp) {
        case CMP_EQ:
            while (*ptr != value) {
                std::this_thread::yield();
            }
            break;
        case CMP_NE:
            while (*ptr == value) {
                std::this_thread::yield();
            }
            break;
        case CMP_GT:
            while (*ptr <= value) {
                std::this_thread::yield();
            }
            break;
        case CMP_GE:
            while (*ptr < value) {
                std::this_thread::yield();
            }
            break;
        case CMP_LT:
            while (*ptr >= value) {
                std::this_thread::yield();
            }
            break;
        case CMP_LE:
            while (*ptr > value) {
                std::this_thread::yield();
            }
            break;
    }
}

int NpuCommEngine::test(void* addr, CmpOp cmp, int64_t value) {
    volatile int64_t* ptr = static_cast<volatile int64_t*>(addr);

    switch (cmp) {
        case CMP_EQ:
            return (*ptr == value) ? 1 : 0;
        case CMP_NE:
            return (*ptr != value) ? 1 : 0;
        case CMP_GT:
            return (*ptr > value) ? 1 : 0;
        case CMP_GE:
            return (*ptr >= value) ? 1 : 0;
        case CMP_LT:
            return (*ptr < value) ? 1 : 0;
        case CMP_LE:
            return (*ptr <= value) ? 1 : 0;
        default:
            return 0;
    }
}

// 集体通信操作实现
void NpuCommEngine::barrier(team_t team) {
    std::cout << "[NPUCOMM] Barrier operation on team " << team << std::endl;
    // 实际实现应该在团队范围内进行同步
    barrier_all();
}

void NpuCommEngine::barrier_team(team_t team) {
    std::cout << "[NPUCOMM] Team barrier operation on team " << team << std::endl;
    // 实际实现应该在指定团队范围内进行同步
    barrier_all();
}

void NpuCommEngine::sync(team_t team) {
    std::cout << "[NPUCOMM] Sync operation on team " << team << std::endl;
    // 团队同步
    quiet();
}

void NpuCommEngine::sync_all() {
    std::cout << "[NPUCOMM] Sync all operation" << std::endl;
    sync(0); // 默认团队
}

void NpuCommEngine::broadcast(void* dest, const void* src, size_t nelems, int root, team_t team) {
    std::cout << "[NPUCOMM] Broadcast: " << nelems << " elements from root " << root << " on team " << team << std::endl;
    // 简化的广播实现 - 复制到所有PE
    memcpy(dest, src, nelems * sizeof(int64_t)); // 假设int64_t数据
}

void NpuCommEngine::fcollect(void* dest, const void* src, size_t nelems, int contrib, team_t team) {
    std::cout << "[NPUCOMM] Fcollect: " << nelems << " elements, contrib=" << contrib << " on team " << team << std::endl;
    // Flat collect操作
    memcpy(dest, src, nelems * sizeof(int64_t));
}

void NpuCommEngine::alltoall(void* dest, const void* src, size_t nelems, team_t team) {
    std::cout << "[NPUCOMM] All-to-all: " << nelems << " elements on team " << team << std::endl;
    // All-to-all操作
    memcpy(dest, src, nelems * sizeof(int64_t));
}

// 归约操作实现
void NpuCommEngine::and_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    std::cout << "[NPUCOMM] AND reduce: " << nelems << " elements on team " << team << std::endl;
    const int64_t* src_ptr = static_cast<const int64_t*>(src);
    int64_t* dest_ptr = static_cast<int64_t*>(dest);
    for (size_t i = 0; i < nelems; ++i) {
        dest_ptr[i] &= src_ptr[i];
    }
}

void NpuCommEngine::or_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    std::cout << "[NPUCOMM] OR reduce: " << nelems << " elements on team " << team << std::endl;
    const int64_t* src_ptr = static_cast<const int64_t*>(src);
    int64_t* dest_ptr = static_cast<int64_t*>(dest);
    for (size_t i = 0; i < nelems; ++i) {
        dest_ptr[i] |= src_ptr[i];
    }
}

void NpuCommEngine::xor_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    std::cout << "[NPUCOMM] XOR reduce: " << nelems << " elements on team " << team << std::endl;
    const int64_t* src_ptr = static_cast<const int64_t*>(src);
    int64_t* dest_ptr = static_cast<int64_t*>(dest);
    for (size_t i = 0; i < nelems; ++i) {
        dest_ptr[i] ^= src_ptr[i];
    }
}

void NpuCommEngine::max_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    std::cout << "[NPUCOMM] MAX reduce: " << nelems << " elements on team " << team << std::endl;
    const int64_t* src_ptr = static_cast<const int64_t*>(src);
    int64_t* dest_ptr = static_cast<int64_t*>(dest);
    for (size_t i = 0; i < nelems; ++i) {
        if (src_ptr[i] > dest_ptr[i]) {
            dest_ptr[i] = src_ptr[i];
        }
    }
}

void NpuCommEngine::min_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    std::cout << "[NPUCOMM] MIN reduce: " << nelems << " elements on team " << team << std::endl;
    const int64_t* src_ptr = static_cast<const int64_t*>(src);
    int64_t* dest_ptr = static_cast<int64_t*>(dest);
    for (size_t i = 0; i < nelems; ++i) {
        if (src_ptr[i] < dest_ptr[i]) {
            dest_ptr[i] = src_ptr[i];
        }
    }
}

void NpuCommEngine::sum_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    std::cout << "[NPUCOMM] SUM reduce: " << nelems << " elements on team " << team << std::endl;
    const int64_t* src_ptr = static_cast<const int64_t*>(src);
    int64_t* dest_ptr = static_cast<int64_t*>(dest);
    for (size_t i = 0; i < nelems; ++i) {
        dest_ptr[i] += src_ptr[i];
    }
}

void NpuCommEngine::prod_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    std::cout << "[NPUCOMM] PROD reduce: " << nelems << " elements on team " << team << std::endl;
    const int64_t* src_ptr = static_cast<const int64_t*>(src);
    int64_t* dest_ptr = static_cast<int64_t*>(dest);
    for (size_t i = 0; i < nelems; ++i) {
        dest_ptr[i] *= src_ptr[i];
    }
}

NpuCommEngine::CommStats NpuCommEngine::getStats() const {
    return stats_;
}

void NpuCommEngine::updateStats(size_t bytes, uint64_t operations) {
    stats_.totalBytesTransferred += bytes;
    stats_.totalOperations += operations;

    // 简化的延迟估算
    uint64_t latency = 10000; // 10us 模拟延迟
    stats_.averageLatencyNs = (stats_.averageLatencyNs * (stats_.totalOperations - operations) + latency * operations) / stats_.totalOperations;
    stats_.maxLatencyNs = std::max(stats_.maxLatencyNs, latency);
    stats_.bandwidthGBps = (double)stats_.totalBytesTransferred / (double)stats_.averageLatencyNs * 1e9 / (1024.0 * 1024.0 * 1024.0);
}

CommStatus NpuCommEngine::optimizeForNpu(void* dest, const void* src, size_t size, int pe, bool isPut) {
    activeOps_.fetch_add(1);

    auto operation = [this, dest, src, size, pe, isPut]() {
        std::cout << "[NPUCOMM] NPU-optimized " << (isPut ? "put" : "get") << ": " << size << " bytes to PE " << pe << std::endl;

        // 模拟NPU传输延迟
        auto start = std::chrono::high_resolution_clock::now();

        // 使用NPU的RDMA或高速互连进行传输
        // 这里是模拟实现，实际应该调用NPU的硬件加速传输

        // 简化的内存复制 (实际应该使用NPU的硬件传输)
        size_t copy_size = std::min(size, sendBuffer_.size());
        if (isPut) {
            memcpy(sendBuffer_.data(), src, copy_size);
            // 模拟网络传输延迟
            std::this_thread::sleep_for(std::chrono::nanoseconds(500)); // 500ns latency
            memcpy(dest, sendBuffer_.data(), copy_size);
        } else {
            memcpy(recvBuffer_.data(), src, copy_size);
            std::this_thread::sleep_for(std::chrono::nanoseconds(500));
            memcpy(dest, recvBuffer_.data(), copy_size);
        }

        auto end = std::chrono::high_resolution_clock::now();
        uint64_t latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // 更新统计信息
        stats_.averageLatencyNs = (stats_.averageLatencyNs * (stats_.totalOperations) + latency) / (stats_.totalOperations + 1);
        stats_.maxLatencyNs = std::max(stats_.maxLatencyNs, latency);

        activeOps_.fetch_sub(1);
    };

    // 异步执行通信操作
    if (context_->getConfig().maxConcurrentOperations > 1) {
        std::thread(operation).detach();
        return COMM_SUCCESS;
    } else {
        operation();
        return COMM_SUCCESS;
    }
}

CommStatus NpuCommEngine::fallbackToStandard(void* dest, const void* src, size_t size, int pe, bool isPut) {
    std::cout << "[NPUCOMM] Standard " << (isPut ? "put" : "get") << ": " << size << " bytes to PE " << pe << std::endl;

    // 标准传输实现 (fallback)
    if (isPut) {
        memcpy(dest, src, size);
    } else {
        memcpy(dest, src, size);
    }

    return COMM_SUCCESS;
}

// NpuFineGrainComm实现
NpuFineGrainComm::NpuFineGrainComm(NpuCommEngine* engine) : engine_(engine) {
}

NpuFineGrainComm::~NpuFineGrainComm() {
}

CommStatus NpuFineGrainComm::fine_put(void* dest, const void* src, size_t size, int pe) {
    if (size <= 4096) { // 小消息优化
        return engine_->put(dest, src, size, pe);
    } else {
        return engine_->put_nbi(dest, src, size, pe);
    }
}

CommStatus NpuFineGrainComm::fine_get(void* dest, const void* src, size_t size, int pe) {
    if (size <= 4096) { // 小消息优化
        return engine_->get(dest, src, size, pe);
    } else {
        return engine_->get_nbi(dest, src, size, pe);
    }
}

CommStatus NpuFineGrainComm::vector_put(void* dest, const void* src, size_t element_size,
                                       size_t num_elements, int pe) {
    size_t total_size = element_size * num_elements;
    std::cout << "[NPUCOMM] Vector put: " << num_elements << " elements of " << element_size
              << " bytes to PE " << pe << std::endl;

    // NPU向量化传输优化
    return engine_->put(dest, src, total_size, pe);
}

CommStatus NpuFineGrainComm::vector_get(void* dest, const void* src, size_t element_size,
                                       size_t num_elements, int pe) {
    size_t total_size = element_size * num_elements;
    std::cout << "[NPUCOMM] Vector get: " << num_elements << " elements of " << element_size
              << " bytes from PE " << pe << std::endl;

    return engine_->get(dest, src, total_size, pe);
}

CommStatus NpuFineGrainComm::block_put(void* dest, const void* src, size_t rows, size_t cols,
                                      size_t element_size, int pe) {
    size_t total_size = rows * cols * element_size;
    std::cout << "[NPUCOMM] Block put: " << rows << "x" << cols << " matrix of " << element_size
              << " bytes to PE " << pe << std::endl;

    // 矩阵块传输，适合AI模型参数同步
    return engine_->put(dest, src, total_size, pe);
}

CommStatus NpuFineGrainComm::block_get(void* dest, const void* src, size_t rows, size_t cols,
                                      size_t element_size, int pe) {
    size_t total_size = rows * cols * element_size;
    std::cout << "[NPUCOMM] Block get: " << rows << "x" << cols << " matrix of " << element_size
              << " bytes from PE " << pe << std::endl;

    return engine_->get(dest, src, total_size, pe);
}

CommStatus NpuFineGrainComm::pipeline_put(void* dest, const void* src, size_t size, int pe) {
    std::cout << "[NPUCOMM] Pipeline put: " << size << " bytes to PE " << pe << std::endl;

    // 流水线通信，overlap compute and communication
    // 将大消息分成小块进行流水线传输
    const size_t chunk_size = 64 * 1024; // 64KB chunks
    size_t remaining = size;
    size_t offset = 0;

    while (remaining > 0) {
        size_t current_chunk = std::min(remaining, chunk_size);
        CommStatus status = engine_->put_nbi(
            static_cast<char*>(dest) + offset,
            static_cast<const char*>(src) + offset,
            current_chunk, pe);

        if (status != COMM_SUCCESS) {
            return status;
        }

        remaining -= current_chunk;
        offset += current_chunk;
    }

    return engine_->quiet(); // 等待所有块完成
}

CommStatus NpuFineGrainComm::pipeline_get(void* dest, const void* src, size_t size, int pe) {
    std::cout << "[NPUCOMM] Pipeline get: " << size << " bytes from PE " << pe << std::endl;

    // 流水线通信
    const size_t chunk_size = 64 * 1024; // 64KB chunks
    size_t remaining = size;
    size_t offset = 0;

    while (remaining > 0) {
        size_t current_chunk = std::min(remaining, chunk_size);
        CommStatus status = engine_->get_nbi(
            static_cast<char*>(dest) + offset,
            static_cast<const char*>(src) + offset,
            current_chunk, pe);

        if (status != COMM_SUCCESS) {
            return status;
        }

        remaining -= current_chunk;
        offset += current_chunk;
    }

    return engine_->quiet(); // 等待所有块完成
}

// NpuMemoryConsistency实现
NpuMemoryConsistency::NpuMemoryConsistency(NpuCommContext* context) : context_(context) {
}

NpuMemoryConsistency::~NpuMemoryConsistency() {
}

CommStatus NpuMemoryConsistency::acquire() {
    std::cout << "[NPUCOMM] Memory acquire operation" << std::endl;
    // 内存获取操作，确保后续读取看到最新的数据
    std::atomic_thread_fence(std::memory_order_acquire);
    return COMM_SUCCESS;
}

CommStatus NpuMemoryConsistency::release() {
    std::cout << "[NPUCOMM] Memory release operation" << std::endl;
    // 内存释放操作，确保之前的写入对其他线程可见
    std::atomic_thread_fence(std::memory_order_release);
    return COMM_SUCCESS;
}

CommStatus NpuMemoryConsistency::flush(void* addr, size_t size) {
    std::cout << "[NPUCOMM] Memory flush: " << size << " bytes" << std::endl;
    // 刷新内存区域
    // 实际实现应该调用NPU的缓存刷新指令
    return COMM_SUCCESS;
}

CommStatus NpuMemoryConsistency::flush_all() {
    std::cout << "[NPUCOMM] Memory flush all" << std::endl;
    // 刷新所有缓存
    return COMM_SUCCESS;
}

CommStatus NpuMemoryConsistency::prefetch(void* addr, size_t size) {
    std::cout << "[NPUCOMM] Memory prefetch: " << size << " bytes" << std::endl;
    // 预取数据到缓存
    // 实际实现应该调用NPU的预取指令
    return COMM_SUCCESS;
}

CommStatus NpuMemoryConsistency::evict(void* addr, size_t size) {
    std::cout << "[NPUCOMM] Memory evict: " << size << " bytes" << std::endl;
    // 从缓存中驱逐数据
    return COMM_SUCCESS;
}

// NpuCommTeam实现
NpuCommTeam::NpuCommTeam(NpuCommContext* context) : context_(context) {
}

NpuCommTeam::~NpuCommTeam() {
    teams_.clear();
}

CommStatus NpuCommTeam::team_create(int start_pe, int stride, int size, int* team_id) {
    std::vector<int> team_members;
    for (int i = 0; i < size; ++i) {
        int pe = start_pe + i * (1 << stride);
        if (pe < context_->numPes()) {
            team_members.push_back(pe);
        }
    }

    *team_id = nextTeamId_++;
    teams_[*team_id] = team_members;

    std::cout << "[NPUCOMM] Created team " << *team_id << " with " << team_members.size() << " members" << std::endl;

    return COMM_SUCCESS;
}

CommStatus NpuCommTeam::team_destroy(int team_id) {
    auto it = teams_.find(team_id);
    if (it != teams_.end()) {
        teams_.erase(it);
        std::cout << "[NPUCOMM] Destroyed team " << team_id << std::endl;
        return COMM_SUCCESS;
    }
    return COMM_INVALID_PE; // Team not found
}

CommStatus NpuCommTeam::team_split(int parent_team, int color, int key, int* new_team) {
    // 简化的团队分割实现
    auto it = teams_.find(parent_team);
    if (it == teams_.end()) {
        return COMM_INVALID_PE;
    }

    // 根据color分割团队
    std::vector<int> new_team_members;
    for (int pe : it->second) {
        if (pe % 2 == color) { // 简化的分割逻辑
            new_team_members.push_back(pe);
        }
    }

    if (new_team_members.empty()) {
        return COMM_ERROR;
    }

    *new_team = nextTeamId_++;
    teams_[*new_team] = new_team_members;

    std::cout << "[NPUCOMM] Split team " << parent_team << " into team " << *new_team << std::endl;

    return COMM_SUCCESS;
}

int NpuCommTeam::team_npes(int team_id) const {
    auto it = teams_.find(team_id);
    if (it != teams_.end()) {
        return it->second.size();
    }
    return 0;
}

int NpuCommTeam::team_mype(int team_id) const {
    auto it = teams_.find(team_id);
    if (it != teams_.end()) {
        const auto& members = it->second;
        auto my_pe_it = std::find(members.begin(), members.end(), context_->myPe());
        if (my_pe_it != members.end()) {
            return std::distance(members.begin(), my_pe_it);
        }
    }
    return -1; // Not a member of this team
}

CommStatus NpuCommTeam::team_broadcast(void* dest, const void* src, size_t size, int root, int team_id) {
    std::cout << "[NPUCOMM] Team broadcast: team " << team_id << ", root " << root << ", size " << size << std::endl;

    auto it = teams_.find(team_id);
    if (it == teams_.end()) {
        return COMM_INVALID_PE;
    }

    // 简化的广播实现 - 从root复制到所有团队成员
    memcpy(dest, src, size);
    std::this_thread::sleep_for(std::chrono::microseconds(100)); // 模拟广播延迟

    return COMM_SUCCESS;
}

CommStatus NpuCommTeam::team_reduce(void* dest, const void* src, size_t size, int op, int team_id) {
    std::cout << "[NPUCOMM] Team reduce: team " << team_id << ", op=" << op << ", size=" << size << std::endl;

    // 简化的归约实现
    memcpy(dest, src, size);
    std::this_thread::sleep_for(std::chrono::microseconds(200));

    return COMM_SUCCESS;
}

CommStatus NpuCommTeam::team_allreduce(void* dest, const void* src, size_t size, int op, int team_id) {
    std::cout << "[NPUCOMM] Team allreduce: team " << team_id << ", op=" << op << ", size=" << size << std::endl;

    // 简化的全归约实现
    memcpy(dest, src, size);
    std::this_thread::sleep_for(std::chrono::microseconds(300));

    return COMM_SUCCESS;
}

CommStatus NpuCommTeam::team_gather(void* dest, const void* src, size_t size, int root, int team_id) {
    std::cout << "[NPUCOMM] Team gather: team " << team_id << ", root " << root << ", size=" << size << std::endl;

    // 简化的收集实现
    memcpy(dest, src, size);
    std::this_thread::sleep_for(std::chrono::microseconds(150));

    return COMM_SUCCESS;
}

CommStatus NpuCommTeam::team_scatter(void* dest, const void* src, size_t size, int root, int team_id) {
    std::cout << "[NPUCOMM] Team scatter: team " << team_id << ", root " << root << ", size=" << size << std::endl;

    // 简化的分散实现
    memcpy(dest, src, size);
    std::this_thread::sleep_for(std::chrono::microseconds(150));

    return COMM_SUCCESS;
}

// NpuComm单例实现
NpuComm& NpuComm::getInstance() {
    static NpuComm instance;
    return instance;
}

CommStatus NpuComm::init(const NpuCommConfig& config) {
    auto& instance = getInstance();

    instance.context_ = std::make_unique<NpuCommContext>(config);
    CommStatus status = instance.context_->init();
    if (status != COMM_SUCCESS) {
        return status;
    }

    instance.engine_ = std::make_unique<NpuCommEngine>(instance.context_.get());
    status = instance.engine_->initialize();
    if (status != COMM_SUCCESS) {
        return status;
    }

    instance.fineComm_ = std::make_unique<NpuFineGrainComm>(instance.engine_.get());
    instance.memory_ = std::make_unique<NpuMemoryConsistency>(instance.context_.get());
    instance.team_ = std::make_unique<NpuCommTeam>(instance.context_.get());

    std::cout << "[NPUCOMM] NPU Communication Library initialized successfully" << std::endl;
    return COMM_SUCCESS;
}

CommStatus NpuComm::init_thread(ThreadLevel requested, ThreadLevel* provided) {
    // 简化的线程初始化
    *provided = requested;
    std::cout << "[NPUCOMM] Thread level initialized: " << requested << std::endl;
    return COMM_SUCCESS;
}

CommStatus NpuComm::init_attr(const InitAttr& attr) {
    NpuCommConfig config;
    config.numPes = attr.numPes > 0 ? attr.numPes : 8;
    config.myPe = attr.myPe >= 0 ? attr.myPe : 0;
    return init(config);
}

CommStatus NpuComm::finalize() {
    auto& instance = getInstance();

    instance.team_.reset();
    instance.memory_.reset();
    instance.fineComm_.reset();
    instance.engine_.reset();

    if (instance.context_) {
        instance.context_->finalize();
        instance.context_.reset();
    }

    std::cout << "[NPUCOMM] NPU Communication Library finalized" << std::endl;
    return COMM_SUCCESS;
}

CommStatus NpuComm::global_exit(int status) {
    std::cout << "[NPUCOMM] Global exit with status: " << status << std::endl;
    // 实际实现应该终止所有PE
    finalize();
    exit(status);
    return COMM_SUCCESS;
}

// 查询函数实现
int NpuComm::my_pe() {
    auto& instance = getInstance();
    return instance.context_ ? instance.context_->myPe() : -1;
}

int NpuComm::num_pes() {
    auto& instance = getInstance();
    return instance.context_ ? instance.context_->numPes() : 0;
}

void* NpuComm::ptr(const void* ptr, int pe) {
    auto& instance = getInstance();
    if (!instance.context_) return nullptr;

    if (pe == instance.context_->myPe()) {
        return const_cast<void*>(ptr);
    }
    return nullptr; // 远程指针
}

const char* NpuComm::info_get_version() {
    return "NPUCOMM 1.0.0";
}

const char* NpuComm::info_get_name() {
    return "NPU Communication Library";
}

// 内存管理实现
void* NpuComm::malloc(size_t size) {
    return ::malloc(size);
}

void* NpuComm::calloc(size_t nmemb, size_t size) {
    return ::calloc(nmemb, size);
}

void* NpuComm::align(size_t alignment, size_t size) {
    void* ptr = nullptr;
    // posix_memalign returns 0 on success; on failure it returns an error number and leaves ptr unspecified.
    // Return nullptr on failure to avoid propagating an invalid pointer.
    int ret = posix_memalign(&ptr, alignment, size);
    return (ret == 0) ? ptr : nullptr;
}

void NpuComm::free(void* ptr) {
    ::free(ptr);
}

void* NpuComm::shmalloc(size_t size) {
    auto& instance = getInstance();
    return instance.context_ ? instance.context_->shmalloc(size) : nullptr;
}

CommStatus NpuComm::shfree(void* ptr) {
    auto& instance = getInstance();
    return instance.context_ ? instance.context_->shfree(ptr) : COMM_ERROR;
}

void* NpuComm::shmemalign(size_t alignment, size_t size) {
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, alignment, size);
    return (ret == 0) ? ptr : nullptr;
}

void* NpuComm::shrealloc(void* ptr, size_t size) {
    // 简化的realloc实现
    return ::realloc(ptr, size);
}

// 远程内存访问实现
CommStatus NpuComm::put(void* dest, const void* src, size_t size, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        return instance.engine_->put(dest, src, size, pe, SYNC_NONE);
    }
    return COMM_SUCCESS;
}

CommStatus NpuComm::get(void* dest, const void* src, size_t size, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        return instance.engine_->get(dest, src, size, pe, SYNC_NONE);
    }
    return COMM_SUCCESS;
}

void NpuComm::put_nbi(void* dest, const void* src, size_t size, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->put_nbi(dest, src, size, pe);
    }
}

void NpuComm::get_nbi(void* dest, const void* src, size_t size, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->get_nbi(dest, src, size, pe);
    }
}

// 类型化操作实现
void NpuComm::put8(void* dest, const void* src, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->put8(dest, src, nelems, pe);
    }
}

void NpuComm::put16(void* dest, const void* src, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->put16(dest, src, nelems, pe);
    }
}

void NpuComm::put32(void* dest, const void* src, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->put32(dest, src, nelems, pe);
    }
}

void NpuComm::put64(void* dest, const void* src, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->put64(dest, src, nelems, pe);
    }
}

void NpuComm::put128(void* dest, const void* src, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->put128(dest, src, nelems, pe);
    }
}

void NpuComm::get8(void* dest, const void* src, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->get8(dest, src, nelems, pe);
    }
}

void NpuComm::get16(void* dest, const void* src, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->get16(dest, src, nelems, pe);
    }
}

void NpuComm::get32(void* dest, const void* src, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->get32(dest, src, nelems, pe);
    }
}

void NpuComm::get64(void* dest, const void* src, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->get64(dest, src, nelems, pe);
    }
}

void NpuComm::get128(void* dest, const void* src, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->get128(dest, src, nelems, pe);
    }
}

// 单元素操作实现
void NpuComm::p8(void* dest, int8_t value, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->p8(dest, value, pe);
    }
}

void NpuComm::p16(void* dest, int16_t value, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->p16(dest, value, pe);
    }
}

void NpuComm::p32(void* dest, int32_t value, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->p32(dest, value, pe);
    }
}

void NpuComm::p64(void* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->p64(dest, value, pe);
    }
}

int8_t NpuComm::g8(const void* src, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->g8(src, pe) : 0;
}

int16_t NpuComm::g16(const void* src, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->g16(src, pe) : 0;
}

int32_t NpuComm::g32(const void* src, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->g32(src, pe) : 0;
}

int64_t NpuComm::g64(const void* src, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->g64(src, pe) : 0;
}

// 非连续操作实现
void NpuComm::iput32(void* dest, const void* src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->iput32(dest, src, dst, sst, nelems, pe);
    }
}

void NpuComm::iget32(void* dest, const void* src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->iget32(dest, src, dst, sst, nelems, pe);
    }
}

// 原子操作实现
int64_t NpuComm::atomic_fetch_add(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->atomic_fetch_add(dest, value, pe) : 0;
}

void NpuComm::atomic_add(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->atomic_add(dest, value, pe);
    }
}

int64_t NpuComm::atomic_fetch_and(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->atomic_fetch_and(dest, value, pe) : 0;
}

void NpuComm::atomic_and(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->atomic_and(dest, value, pe);
    }
}

int64_t NpuComm::atomic_fetch_or(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->atomic_fetch_or(dest, value, pe) : 0;
}

void NpuComm::atomic_or(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->atomic_or(dest, value, pe);
    }
}

int64_t NpuComm::atomic_fetch_xor(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->atomic_fetch_xor(dest, value, pe) : 0;
}

void NpuComm::atomic_xor(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->atomic_xor(dest, value, pe);
    }
}

int64_t NpuComm::atomic_compare_swap(int64_t* dest, int64_t compare, int64_t swap, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->atomic_compare_swap(dest, compare, swap, pe) : 0;
}

int64_t NpuComm::atomic_swap(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->atomic_swap(dest, value, pe) : 0;
}

int64_t NpuComm::atomic_fetch_inc(int64_t* dest, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->atomic_fetch_inc(dest, pe) : 0;
}

void NpuComm::atomic_inc(int64_t* dest, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->atomic_inc(dest, pe);
    }
}

int64_t NpuComm::atomic_fetch_set(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->atomic_fetch_set(dest, value, pe) : 0;
}

void NpuComm::atomic_set(int64_t* dest, int64_t value, int pe) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->atomic_set(dest, value, pe);
    }
}

// 内存排序实现
CommStatus NpuComm::fence() {
    auto& instance = getInstance();
    if (instance.engine_) {
        return instance.engine_->fence();
    }
    return COMM_SUCCESS;
}

CommStatus NpuComm::quiet() {
    auto& instance = getInstance();
    if (instance.engine_) {
        return instance.engine_->quiet();
    }
    return COMM_SUCCESS;
}

// 点对点同步实现
void NpuComm::wait_until(void* addr, CmpOp cmp, int64_t value) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->wait_until(addr, cmp, value);
    }
}

int NpuComm::test(void* addr, CmpOp cmp, int64_t value) {
    auto& instance = getInstance();
    return instance.engine_ ? instance.engine_->test(addr, cmp, value) : 0;
}

// 集体通信实现
void NpuComm::barrier(team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->barrier_team(team);
    }
}

CommStatus NpuComm::barrier_all() {
    auto& instance = getInstance();
    if (instance.engine_) {
        return instance.engine_->barrier_all();
    }
    return COMM_SUCCESS;
}

void NpuComm::sync(team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->sync(team);
    }
}

void NpuComm::sync_all() {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->sync_all();
    }
}

void NpuComm::broadcast(void* dest, const void* src, size_t nelems, int root, team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->broadcast(dest, src, nelems, root, team);
    }
}

void NpuComm::fcollect(void* dest, const void* src, size_t nelems, int contrib, team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->fcollect(dest, src, nelems, contrib, team);
    }
}

void NpuComm::alltoall(void* dest, const void* src, size_t nelems, team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->alltoall(dest, src, nelems, team);
    }
}

// 归约操作实现
void NpuComm::and_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->and_reduce(dest, src, nelems, team);
    }
}

void NpuComm::or_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->or_reduce(dest, src, nelems, team);
    }
}

void NpuComm::xor_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->xor_reduce(dest, src, nelems, team);
    }
}

void NpuComm::max_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->max_reduce(dest, src, nelems, team);
    }
}

void NpuComm::min_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->min_reduce(dest, src, nelems, team);
    }
}

void NpuComm::sum_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->sum_reduce(dest, src, nelems, team);
    }
}

void NpuComm::prod_reduce(void* dest, const void* src, size_t nelems, team_t team) {
    auto& instance = getInstance();
    if (instance.engine_) {
        instance.engine_->prod_reduce(dest, src, nelems, team);
    }
}


} // namespace comm
} // namespace npu
