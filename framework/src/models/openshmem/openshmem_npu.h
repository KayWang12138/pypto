/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#ifndef OPENSHMEM_NPU_H
#define OPENSHMEM_NPU_H

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <string>

namespace npu {
namespace openshmem {

// OpenSHMEM配置
struct OpenSHMEMConfig {
    int numPes = 8;                    // 处理单元数量 (NPU数量)
    int myPe = 0;                      // 当前PE ID
    size_t symmetricHeapSize = 1024 * 1024 * 1024; // 对称堆大小 (1GB)
    bool enableNpuOptimization = true; // 启用NPU优化
    bool enablePtoIntegration = true;  // 启用PTO通信集成
    int ptoAccuracyLevel = 1;          // PTO精度级别
    std::string ptoArchType = "A2A3";  // PTO架构类型

    // NPU特定配置
    int npuCoreCount = 32;            // NPU核心数量
    size_t hbmMemorySize = 32ULL * 1024 * 1024 * 1024; // HBM内存大小 (32GB)
    bool enableHbmOptimization = true; // 启用HBM优化
    int maxConcurrentOperations = 16;  // 最大并发操作数
};

// OpenSHMEM上下文
class OpenSHMEMContext {
public:
    explicit OpenSHMEMContext(const OpenSHMEMConfig& config);
    ~OpenSHMEMContext();

    // 初始化和清理
    int init();
    int finalize();

    // 获取配置
    const OpenSHMEMConfig& getConfig() const { return config_; }

    // 获取当前PE信息
    int myPe() const { return config_.myPe; }
    int numPes() const { return config_.numPes; }

    // 对称堆管理
    void* shmalloc(size_t size);
    void shfree(void* ptr);
    bool isSymmetricAddress(const void* addr) const;

private:
    OpenSHMEMConfig config_;
    bool initialized_ = false;
    std::vector<void*> symmetricHeap_;
    size_t allocatedSize_ = 0;
};

// OpenSHMEM传输引擎 (NPU优化)
class OpenSHMEMTransport {
public:
    explicit OpenSHMEMTransport(OpenSHMEMContext* context);
    ~OpenSHMEMTransport();

    // 初始化传输层
    int initialize();

    // 单边通信操作 (核心OpenSHMEM API)
    int put(void* dest, const void* src, size_t size, int pe);
    int get(void* dest, const void* src, size_t size, int pe);

    // 原子操作
    int64_t atomic_fetch_add(int64_t* dest, int64_t value, int pe);
    int64_t atomic_compare_swap(int64_t* dest, int64_t compare, int64_t swap, int pe);

    // 栅栏同步
    int barrier(int PE_start, int logPE_stride, int PE_size);
    int barrier_all();

    // 归约操作
    int allreduce(void* dest, const void* src, size_t size,
                  int op, int PE_start, int logPE_stride, int PE_size);

    // 性能监控
    struct TransportStats {
        uint64_t totalBytesTransferred = 0;
        uint64_t totalOperations = 0;
        uint64_t averageLatencyNs = 0;
        uint64_t maxLatencyNs = 0;
        double bandwidthGBps = 0.0;
    };
    TransportStats getStats() const;

private:
    OpenSHMEMContext* context_;
    TransportStats stats_;

    // NPU特定的传输优化
    int optimizeForNpu(void* dest, const void* src, size_t size, int pe);
    int fallbackToStandard(void* dest, const void* src, size_t size, int pe);

    // 内部缓冲管理
    std::vector<uint8_t> sendBuffer_;
    std::vector<uint8_t> recvBuffer_;
};

// OpenSHMEM内存管理器 (NPU-aware)
class OpenSHMEMMemoryManager {
public:
    explicit OpenSHMEMMemoryManager(OpenSHMEMContext* context);
    ~OpenSHMEMMemoryManager();

    // 内存分配 (对称堆)
    void* alloc(size_t size, int alignment = 64);
    void free(void* ptr);

    // 内存查询
    bool isValidAddress(const void* addr) const;
    size_t getAllocationSize(const void* addr) const;

    // NPU内存优化
    void* allocNpuOptimized(size_t size);
    void prefetchToNpu(void* addr, size_t size);
    void flushFromNpu(void* addr, size_t size);

    // 内存统计
    struct MemoryStats {
        size_t totalAllocated = 0;
        size_t peakAllocated = 0;
        size_t numAllocations = 0;
        double memoryEfficiency = 0.0;
    };
    MemoryStats getStats() const;

private:
    OpenSHMEMContext* context_;
    MemoryStats stats_;
    std::unordered_map<const void*, size_t> allocationMap_;

    // NPU内存池管理
    struct NpuMemoryPool {
        void* baseAddress = nullptr;
        size_t poolSize = 0;
        size_t allocatedSize = 0;
        std::vector<std::pair<void*, size_t>> allocations;
    };
    NpuMemoryPool npuPool_;
};

// OpenSHMEM同步管理器
class OpenSHMEMSynchronization {
public:
    explicit OpenSHMEMSynchronization(OpenSHMEMContext* context);
    ~OpenSHMEMSynchronization();

    // 同步原语
    int quiet();  // 等待所有单边通信完成
    int fence();  // 内存栅栏

    // 事件同步 (扩展)
    int wait_until(void* addr, int cmp, int64_t value);
    int test(void* addr, int cmp, int64_t value);

    // 团队同步
    int team_sync(int team_id);
    int team_barrier(int team_id);

private:
    OpenSHMEMContext* context_;
    std::unordered_map<int, std::vector<int>> teams_;
};

// OpenSHMEM团队管理器
class OpenSHMEMTeamManager {
public:
    explicit OpenSHMEMTeamManager(OpenSHMEMContext* context);
    ~OpenSHMEMTeamManager();

    // 团队创建和管理
    int team_create(int start_pe, int stride, int size, int* team_id);
    int team_destroy(int team_id);
    int team_translate_pe(int team_id, int pe, int team_pe);

    // 团队查询
    int team_npes(int team_id);
    int team_mype(int team_id);

    // 团队通信
    int team_broadcast(void* dest, const void* src, size_t size, int root, int team_id);
    int team_collect(void* dest, const void* src, size_t size, int team_id);
    int team_alltoall(void* dest, const void* src, size_t size, int team_id);

private:
    OpenSHMEMContext* context_;
    std::unordered_map<int, std::vector<int>> teams_;
    int nextTeamId_ = 1;
};

} // namespace openshmem
} // namespace npu

#endif // OPENSHMEM_NPU_H
