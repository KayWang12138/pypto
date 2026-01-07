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
#ifndef NPUCOMM_H
#define NPUCOMM_H

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <string>
#include <atomic>

namespace npu {
namespace comm {

// NPU通信配置
struct NpuCommConfig {
    int numPes = 8;                          // 处理单元数量 (NPU数量)
    int myPe = 0;                            // 当前PE ID
    size_t symmetricHeapSize = 1024 * 1024 * 1024ULL; // 对称堆大小 (1GB)
    bool enableNpuOptimization = true;       // 启用NPU优化
    bool enableFineGrainComm = true;         // 启用细粒度通信
    int npuCoreCount = 32;                   // NPU核心数量
    size_t hbmMemorySize = 32ULL * 1024 * 1024 * 1024; // HBM内存大小 (32GB)
    bool enableDeviceInitiated = true;       // 启用设备端发起通信
    int maxConcurrentOperations = 64;        // 最大并发操作数
    size_t maxMessageSize = 64 * 1024;       // 最大消息大小 (64KB)
    bool enablePgasMode = true;              // 启用PGAS模式
};

// 通信状态
enum CommStatus {
    COMM_SUCCESS = 0,
    COMM_ERROR = -1,
    COMM_TIMEOUT = -2,
    COMM_INVALID_PE = -3,
    COMM_INVALID_ADDR = -4,
    COMM_OUT_OF_MEMORY = -5
};

// 同步模式
enum SyncMode {
    SYNC_NONE = 0,        // 无同步
    SYNC_FENCE = 1,       // 内存栅栏
    SYNC_QUIET = 2,       // 等待所有通信完成
    SYNC_BARRIER = 3      // 全局栅栏
};

// 数据类型枚举 (NVSHMEM兼容)
enum DataType {
    TYPE_SIGNED_BYTE = 0,
    TYPE_INT32 = 1,
    TYPE_INT64 = 2,
    TYPE_FLOAT32 = 3,
    TYPE_FLOAT64 = 4
};

// 归约操作类型
enum ReduceOp {
    OP_AND = 0,
    OP_OR = 1,
    OP_XOR = 2,
    OP_MAX = 3,
    OP_MIN = 4,
    OP_SUM = 5,
    OP_PROD = 6
};

// 比较操作类型
enum CmpOp {
    CMP_EQ = 0,
    CMP_NE = 1,
    CMP_GT = 2,
    CMP_GE = 3,
    CMP_LT = 4,
    CMP_LE = 5
};

// 线程级别
enum ThreadLevel {
    THREAD_SINGLE = 0,
    THREAD_FUNNELED = 1,
    THREAD_SERIALIZED = 2,
    THREAD_MULTIPLE = 3
};

// 初始化属性
struct InitAttr {
    int numPes = -1;
    int myPe = -1;
    ThreadLevel threadLevel = THREAD_SINGLE;
    void* bootstrap = nullptr;
    void* transport = nullptr;
};

// 团队配置
struct TeamConfig {
    int numContexts = 1;
    bool isSharedContext = false;
    bool isDeviceTeam = false;
};

// 团队句柄类型
typedef int team_t;

// NPU通信上下文
class NpuCommContext {
public:
    explicit NpuCommContext(const NpuCommConfig& config);
    ~NpuCommContext();

    // 初始化和清理
    CommStatus init();
    CommStatus finalize();

    // 获取配置和状态
    const NpuCommConfig& getConfig() const { return config_; }
    int myPe() const { return config_.myPe; }
    int numPes() const { return config_.numPes; }
    bool isInitialized() const { return initialized_; }

    // 对称堆管理 (PGAS核心)
    void* shmalloc(size_t size);
    CommStatus shfree(void* ptr);
    bool isSymmetricAddress(const void* addr) const;

    // 全局地址转换
    void* globalToLocal(void* global_addr, int pe) const;
    void* localToGlobal(void* local_addr, int pe) const;

private:
    NpuCommConfig config_;
    bool initialized_ = false;
    std::vector<void*> symmetricHeap_;
    size_t allocatedSize_ = 0;
    std::unordered_map<const void*, size_t> allocationMap_;

    // NPU-specific memory management
    void* allocNpuMemory(size_t size);
    void freeNpuMemory(void* ptr);
    bool isNpuMemory(void* ptr) const;
};

// 单边通信引擎 (核心API)
class NpuCommEngine {
public:
    explicit NpuCommEngine(NpuCommContext* context);
    ~NpuCommEngine();

    CommStatus initialize();

    // 核心单边通信API (put/get)
    CommStatus put(void* dest, const void* src, size_t size, int pe, SyncMode sync = SYNC_NONE);
    CommStatus get(void* dest, const void* src, size_t size, int pe, SyncMode sync = SYNC_NONE);

    // 批量通信操作
    CommStatus put_nbi(void* dest, const void* src, size_t size, int pe);
    CommStatus get_nbi(void* dest, const void* src, size_t size, int pe);

    // 原子操作 (细粒度同步) - NVSHMEM兼容
    int64_t atomic_fetch_add(int64_t* dest, int64_t value, int pe);
    void atomic_add(int64_t* dest, int64_t value, int pe);
    int64_t atomic_fetch_and(int64_t* dest, int64_t value, int pe);
    void atomic_and(int64_t* dest, int64_t value, int pe);
    int64_t atomic_fetch_or(int64_t* dest, int64_t value, int pe);
    void atomic_or(int64_t* dest, int64_t value, int pe);
    int64_t atomic_fetch_xor(int64_t* dest, int64_t value, int pe);
    void atomic_xor(int64_t* dest, int64_t value, int pe);
    int64_t atomic_compare_swap(int64_t* dest, int64_t compare, int64_t swap, int pe);
    int64_t atomic_swap(int64_t* dest, int64_t value, int pe);
    int64_t atomic_fetch_inc(int64_t* dest, int pe);
    void atomic_inc(int64_t* dest, int pe);
    int64_t atomic_fetch_set(int64_t* dest, int64_t value, int pe);
    void atomic_set(int64_t* dest, int64_t value, int pe);

    // 同步原语
    CommStatus fence();
    CommStatus quiet();
    CommStatus barrier(int PE_start = 0, int logPE_stride = 0, int PE_size = -1);
    CommStatus barrier_all();

    // 设备端发起的通信 (NPU优化)
    CommStatus device_put(void* dest, const void* src, size_t size, int pe);
    CommStatus device_get(void* dest, const void* src, size_t size, int pe);

    // 类型化put/get操作 (NVSHMEM兼容)
    void put8(void* dest, const void* src, size_t nelems, int pe);
    void put16(void* dest, const void* src, size_t nelems, int pe);
    void put32(void* dest, const void* src, size_t nelems, int pe);
    void put64(void* dest, const void* src, size_t nelems, int pe);
    void put128(void* dest, const void* src, size_t nelems, int pe);

    void get8(void* dest, const void* src, size_t nelems, int pe);
    void get16(void* dest, const void* src, size_t nelems, int pe);
    void get32(void* dest, const void* src, size_t nelems, int pe);
    void get64(void* dest, const void* src, size_t nelems, int pe);
    void get128(void* dest, const void* src, size_t nelems, int pe);

    // 单元素操作 (NVSHMEM P接口)
    void p8(void* dest, int8_t value, int pe);
    void p16(void* dest, int16_t value, int pe);
    void p32(void* dest, int32_t value, int pe);
    void p64(void* dest, int64_t value, int pe);

    int8_t g8(const void* src, int pe);
    int16_t g16(const void* src, int pe);
    int32_t g32(const void* src, int pe);
    int64_t g64(const void* src, int pe);

    // 非连续put/get (NVSHMEM IPUT/IGET)
    void iput32(void* dest, const void* src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);
    void iget32(void* dest, const void* src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

    // 等待和测试操作 (NVSHMEM同步)
    void wait_until(void* addr, CmpOp cmp, int64_t value);
    int test(void* addr, CmpOp cmp, int64_t value);

    // 集体通信操作
    void barrier(team_t team = 0);
    void barrier_team(team_t team); // 避免与barrier(int,int,int)冲突
    void sync(team_t team = 0);
    void sync_all();

    void broadcast(void* dest, const void* src, size_t nelems, int root, team_t team = 0);
    void fcollect(void* dest, const void* src, size_t nelems, int contrib, team_t team = 0);
    void alltoall(void* dest, const void* src, size_t nelems, team_t team = 0);

    // 归约操作
    void and_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    void or_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    void xor_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    void max_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    void min_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    void sum_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    void prod_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);

    // 性能监控
    struct CommStats {
        uint64_t totalBytesTransferred = 0;
        uint64_t totalOperations = 0;
        uint64_t averageLatencyNs = 0;
        uint64_t maxLatencyNs = 0;
        double bandwidthGBps = 0.0;
        uint64_t activeOperations = 0;
    };
    CommStats getStats() const;

private:
    NpuCommContext* context_;
    CommStats stats_;
    std::atomic<uint64_t> activeOps_{0};

    // NPU优化实现
    CommStatus optimizeForNpu(void* dest, const void* src, size_t size, int pe, bool isPut);
    CommStatus fallbackToStandard(void* dest, const void* src, size_t size, int pe, bool isPut);

    // 内部缓冲管理
    std::vector<uint8_t> sendBuffer_;
    std::vector<uint8_t> recvBuffer_;

    // Statistics management
    void updateStats(size_t bytes, uint64_t operations);
};

// 细粒度通信接口 (针对NPU特点优化)
class NpuFineGrainComm {
public:
    explicit NpuFineGrainComm(NpuCommEngine* engine);
    ~NpuFineGrainComm();

    // 细粒度put/get (适合小消息)
    CommStatus fine_put(void* dest, const void* src, size_t size, int pe);
    CommStatus fine_get(void* dest, const void* src, size_t size, int pe);

    // 向量通信 (NPU向量化优化)
    CommStatus vector_put(void* dest, const void* src, size_t element_size,
                         size_t num_elements, int pe);
    CommStatus vector_get(void* dest, const void* src, size_t element_size,
                         size_t num_elements, int pe);

    // 矩阵块通信 (适合AI模型)
    CommStatus block_put(void* dest, const void* src, size_t rows, size_t cols,
                        size_t element_size, int pe);
    CommStatus block_get(void* dest, const void* src, size_t rows, size_t cols,
                        size_t element_size, int pe);

    // 流水线通信 (overlap compute and communication)
    CommStatus pipeline_put(void* dest, const void* src, size_t size, int pe);
    CommStatus pipeline_get(void* dest, const void* src, size_t size, int pe);

private:
    NpuCommEngine* engine_;
};

// 内存一致性管理器
class NpuMemoryConsistency {
public:
    explicit NpuMemoryConsistency(NpuCommContext* context);
    ~NpuMemoryConsistency();

    // 一致性操作
    CommStatus acquire();
    CommStatus release();
    CommStatus flush(void* addr, size_t size);
    CommStatus flush_all();

    // 缓存管理
    CommStatus prefetch(void* addr, size_t size);
    CommStatus evict(void* addr, size_t size);

private:
    NpuCommContext* context_;
};

// 通信团队管理器
class NpuCommTeam {
public:
    explicit NpuCommTeam(NpuCommContext* context);
    ~NpuCommTeam();

    // 团队创建和管理
    CommStatus team_create(int start_pe, int stride, int size, int* team_id);
    CommStatus team_destroy(int team_id);
    CommStatus team_split(int parent_team, int color, int key, int* new_team);

    // 团队查询
    int team_npes(int team_id) const;
    int team_mype(int team_id) const;

    // 团队通信操作
    CommStatus team_broadcast(void* dest, const void* src, size_t size, int root, int team_id);
    CommStatus team_reduce(void* dest, const void* src, size_t size, int op, int team_id);
    CommStatus team_allreduce(void* dest, const void* src, size_t size, int op, int team_id);
    CommStatus team_gather(void* dest, const void* src, size_t size, int root, int team_id);
    CommStatus team_scatter(void* dest, const void* src, size_t size, int root, int team_id);

private:
    NpuCommContext* context_;
    std::unordered_map<int, std::vector<int>> teams_;
    int nextTeamId_ = 1;
};

// NPU通信主接口 (NVSHMEM兼容API)
class NpuComm {
public:
    static NpuComm& getInstance();
    static CommStatus init(const NpuCommConfig& config = NpuCommConfig{});
    static CommStatus init_thread(ThreadLevel requested, ThreadLevel* provided);
    static CommStatus init_attr(const InitAttr& attr);
    static CommStatus finalize();
    static CommStatus global_exit(int status);

    // 查询函数
    static int my_pe();
    static int num_pes();
    static void* ptr(const void* ptr, int pe);
    static const char* info_get_version();
    static const char* info_get_name();

    // 内存管理 (NVSHMEM兼容)
    static void* malloc(size_t size);
    static void* calloc(size_t nmemb, size_t size);
    static void* align(size_t alignment, size_t size);
    static void free(void* ptr);
    static void* shmalloc(size_t size);
    static CommStatus shfree(void* ptr);
    static void* shmemalign(size_t alignment, size_t size);
    static void* shrealloc(void* ptr, size_t size);

    // 远程内存访问 (NVSHMEM兼容)
    static CommStatus put(void* dest, const void* src, size_t size, int pe);
    static CommStatus get(void* dest, const void* src, size_t size, int pe);
    static void put_nbi(void* dest, const void* src, size_t size, int pe);
    static void get_nbi(void* dest, const void* src, size_t size, int pe);

    // 类型化put/get操作
    static void put8(void* dest, const void* src, size_t nelems, int pe);
    static void put16(void* dest, const void* src, size_t nelems, int pe);
    static void put32(void* dest, const void* src, size_t nelems, int pe);
    static void put64(void* dest, const void* src, size_t nelems, int pe);
    static void put128(void* dest, const void* src, size_t nelems, int pe);

    static void get8(void* dest, const void* src, size_t nelems, int pe);
    static void get16(void* dest, const void* src, size_t nelems, int pe);
    static void get32(void* dest, const void* src, size_t nelems, int pe);
    static void get64(void* dest, const void* src, size_t nelems, int pe);
    static void get128(void* dest, const void* src, size_t nelems, int pe);

    // 单元素操作
    static void p8(void* dest, int8_t value, int pe);
    static void p16(void* dest, int16_t value, int pe);
    static void p32(void* dest, int32_t value, int pe);
    static void p64(void* dest, int64_t value, int pe);

    static int8_t g8(const void* src, int pe);
    static int16_t g16(const void* src, int pe);
    static int32_t g32(const void* src, int pe);
    static int64_t g64(const void* src, int pe);

    // 非连续操作
    static void iput32(void* dest, const void* src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);
    static void iget32(void* dest, const void* src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

    // 原子内存操作
    static int64_t atomic_fetch_add(int64_t* dest, int64_t value, int pe);
    static void atomic_add(int64_t* dest, int64_t value, int pe);
    static int64_t atomic_fetch_and(int64_t* dest, int64_t value, int pe);
    static void atomic_and(int64_t* dest, int64_t value, int pe);
    static int64_t atomic_fetch_or(int64_t* dest, int64_t value, int pe);
    static void atomic_or(int64_t* dest, int64_t value, int pe);
    static int64_t atomic_fetch_xor(int64_t* dest, int64_t value, int pe);
    static void atomic_xor(int64_t* dest, int64_t value, int pe);
    static int64_t atomic_compare_swap(int64_t* dest, int64_t compare, int64_t swap, int pe);
    static int64_t atomic_swap(int64_t* dest, int64_t value, int pe);
    static int64_t atomic_fetch_inc(int64_t* dest, int pe);
    static void atomic_inc(int64_t* dest, int pe);
    static int64_t atomic_fetch_set(int64_t* dest, int64_t value, int pe);
    static void atomic_set(int64_t* dest, int64_t value, int pe);

    // 内存排序
    static CommStatus fence();
    static CommStatus quiet();

    // 点对点同步
    static void wait_until(void* addr, CmpOp cmp, int64_t value);
    static int test(void* addr, CmpOp cmp, int64_t value);

    // 集体通信
    static void barrier(team_t team = 0);
    static CommStatus barrier_all();
    static void sync(team_t team = 0);
    static void sync_all();

    static void broadcast(void* dest, const void* src, size_t nelems, int root, team_t team = 0);
    static void fcollect(void* dest, const void* src, size_t nelems, int contrib, team_t team = 0);
    static void alltoall(void* dest, const void* src, size_t nelems, team_t team = 0);

    // 归约操作
    static void and_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    static void or_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    static void xor_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    static void max_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    static void min_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    static void sum_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);
    static void prod_reduce(void* dest, const void* src, size_t nelems, team_t team = 0);

private:
    NpuComm() = default;
    ~NpuComm() = default;
    NpuComm(const NpuComm&) = delete;
    NpuComm& operator=(const NpuComm&) = delete;

    static std::unique_ptr<NpuCommContext> context_;
    static std::unique_ptr<NpuCommEngine> engine_;
    static std::unique_ptr<NpuFineGrainComm> fineComm_;
    static std::unique_ptr<NpuMemoryConsistency> memory_;
    static std::unique_ptr<NpuCommTeam> team_;
};

} // namespace comm
} // namespace npu

#endif // NPUCOMM_H
