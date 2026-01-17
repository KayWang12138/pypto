#ifndef HIGH_PERF_MEMORY_POOL_H
#define HIGH_PERF_MEMORY_POOL_H

#include <vector>
#include <mutex>
#include <unordered_map>
#include <cstdint>

// 日志宏定义（根据实际项目调整）
#define ALOG_INFO_F(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define ALOG_WARN_F(fmt, ...) printf("[WARN] " fmt "\n", ##__VA_ARGS__)
#define ALOG_ERROR_F(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)

// 大页相关常量
#define ONT_GB_SIZE (1024UL * 1024UL * 1024UL)  // 1GB
#define TWO_MB_SIZE (2UL * 1024UL * 1024UL)     // 2MB

// 大页标志（根据实际项目调整）
#define ONT_GB_HUGE_PAGE_FLAGS 0x10000000
#define TWO_MB_HUGE_PAGE_FLAGS 0x20000000

// 内存对齐函数（根据实际项目实现）
uint64_t MemSizeAlign(uint64_t size);

// 尝试获取大页内存（根据实际项目实现）
bool TryGetHugePageMem(uint8_t** devAddr, uint64_t size);

// 运行时内存分配函数（根据实际项目实现）
int rtMalloc(void** devAddr, size_t size, int flags, int priority);

// 运行时内存释放函数（根据实际项目实现）
int rtFree(void* devAddr, size_t size);

// 大页描述结构
struct HugePageDesc {
    void* addr;      // 大页地址
    size_t size;     // 大页大小
    bool is_used;    // 是否被使用
    uint64_t last_used_time;  // 最后使用时间
};

// 内存块描述结构
struct MemoryBlock {
    void* base_addr;   // 块的基地址
    size_t block_size; // 块大小
    size_t used_size;  // 已使用大小
    bool is_huge_1g;   // 是否为1GB大页
    bool is_allocated; // 是否已分配
    
    // 空闲内存链表
    struct FreeNode {
        FreeNode* next; // 指向下一个空闲节点
        size_t size;    // 空闲区域大小
    }* free_list;       // 空闲列表头
    
    MemoryBlock(void* addr, size_t size, bool is_huge_1g) 
        : base_addr(addr), block_size(size), used_size(0), 
          is_huge_1g(is_huge_1g), is_allocated(false), free_list(nullptr) {}
};

// 高性能内存池类
class HighPerfMemoryPool {
public:
    HighPerfMemoryPool();
    ~HighPerfMemoryPool();
    
    // 分配内存
    void* Allocate(uint64_t size);
    
    // 释放内存
    void Free(void* ptr);
    
    // 动态回收内存（可定期调用）
    void DynamicRecycle();
    
    // 设置回收阈值（毫秒）
    void SetRecycleThreshold(uint64_t threshold_ms) {
        recycle_threshold_ms_ = threshold_ms;
    }
    
    // 获取当前内存使用统计
    void GetMemoryStats(size_t& total_alloc, size_t& used, size_t& free) const;
    
private:
    // 内存块对齐
    uint64_t AlignMemorySize(uint64_t size) const;
    
    // 尝试从现有内存块分配
    void* TryAllocateFromExistingBlocks(uint64_t align_size);
    
    // 申请新的1GB大页内存块
    MemoryBlock* Allocate1GBBlock(uint64_t align_size);
    
    // 申请新的2MB大页内存块
    MemoryBlock* Allocate2MBBlock(uint64_t align_size);
    
    // 初始化内存块的空闲列表
    void InitFreeList(MemoryBlock* block);
    
    // 从内存块分配
    void* AllocateFromBlock(MemoryBlock* block, uint64_t align_size);
    
    // 向内存块释放
    void FreeToBlock(MemoryBlock* block, void* ptr, uint64_t size);
    
    // 合并相邻空闲块
    void MergeFreeBlocks(MemoryBlock* block);
    
    // 释放内存块
    void ReleaseBlock(MemoryBlock* block);
    
    // 获取当前时间（毫秒）
    uint64_t GetCurrentTimeMs() const;
    
    // 内存块列表
    std::vector<MemoryBlock*> memory_blocks_;
    
    // 内存块查找表（用于快速定位内存所属的块）
    std::unordered_map<void*, MemoryBlock*> addr_to_block_;
    
    // 已分配的设备地址列表
    std::vector<void*> allocated_dev_addrs_;
    
    // 大页描述符向量
    std::vector<HugePageDesc> huge_page_vec_;
    
    // 线程安全锁
    mutable std::mutex mutex_;
    
    // 内存回收阈值（毫秒）
    uint64_t recycle_threshold_ms_;
    
    // 内存对齐最小单位
    static const size_t MIN_ALIGN_SIZE = 64;
    
    // 最大支持的内存块大小
    static const size_t MAX_BLOCK_SIZE = 1024UL * 1024UL * 1024UL * 4; // 4GB
};

#endif // HIGH_PERF_MEMORY_POOL_H
