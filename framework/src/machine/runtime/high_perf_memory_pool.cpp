#include "high_perf_memory_pool.h"
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>

// 内存对齐函数实现（示例）
uint64_t MemSizeAlign(uint64_t size) {
    const uint64_t ALIGNMENT = 64;
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

// 尝试获取大页内存实现（示例）
bool TryGetHugePageMem(uint8_t** devAddr, uint64_t size) {
    // 这里是示例实现，实际项目中需要替换为真实的大页获取逻辑
    *devAddr = malloc(size);
    return *devAddr != nullptr;
}

// 运行时内存分配函数实现（示例）
int rtMalloc(void** devAddr, size_t size, int flags, int priority) {
    // 这里是示例实现，实际项目中需要替换为真实的rtMalloc调用
    *devAddr = malloc(size);
    return *devAddr != nullptr ? 0 : -1;
}

// 运行时内存释放函数实现（示例）
int rtFree(void* devAddr, size_t size) {
    // 这里是示例实现，实际项目中需要替换为真实的rtFree调用
    free(devAddr);
    return 0;
}

// HighPerfMemoryPool 实现
HighPerfMemoryPool::HighPerfMemoryPool() 
    : recycle_threshold_ms_(30000) { // 默认30秒回收
}

HighPerfMemoryPool::~HighPerfMemoryPool() {
    // 释放所有内存块
    for (MemoryBlock* block : memory_blocks_) {
        ReleaseBlock(block);
        delete block;
    }
    
    // 释放所有设备地址
    for (void* addr : allocated_dev_addrs_) {
        free(addr);
    }
}

void* HighPerfMemoryPool::Allocate(uint64_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (size == 0) {
        return nullptr;
    }
    
    // 内存对齐
    uint64_t align_size = AlignMemorySize(size);
    ALOG_INFO_F("MemoryPool::Allocate size[%lu] with align size[%lu].", size, align_size);
    
    // 1. 尝试从现有内存块分配
    void* ptr = TryAllocateFromExistingBlocks(align_size);
    if (ptr != nullptr) {
        return ptr;
    }
    
    // 2. 尝试分配1GB大页
    MemoryBlock* block = Allocate1GBBlock(align_size);
    if (block != nullptr) {
        ptr = AllocateFromBlock(block, align_size);
        if (ptr != nullptr) {
            return ptr;
        }
    }
    
    // 3. 尝试分配2MB大页
    block = Allocate2MBBlock(align_size);
    if (block != nullptr) {
        ptr = AllocateFromBlock(block, align_size);
        if (ptr != nullptr) {
            return ptr;
        }
    }
    
    ALOG_ERROR_F("MemoryPool::Allocate failed for size %lu", size);
    return nullptr;
}

void HighPerfMemoryPool::Free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 查找ptr所属的内存块
    auto it = addr_to_block_.find(ptr);
    if (it == addr_to_block_.end()) {
        ALOG_ERROR_F("MemoryPool::Free invalid ptr %p", ptr);
        return;
    }
    
    MemoryBlock* block = it->second;
    
    // 从块中释放内存
    // 注意：这里简化实现，实际需要记录每个分配的大小
    // 可以通过在分配时记录大小，或者在内存块中添加大小前缀
    // 这里假设每个分配的大小都被记录在某个地方
    // 简化起见，我们假设释放时知道大小，或者通过其他方式获取
    // 实际实现中需要完善
    
    // 这里简化处理，实际需要实现正确的释放逻辑
    block->used_size -= block->block_size; // 简化处理
    
    // 将释放的内存添加到空闲列表
    FreeToBlock(block, ptr, block->block_size); // 简化处理
    
    // 移除地址映射
    addr_to_block_.erase(it);
    
    ALOG_INFO_F("MemoryPool::Free ptr %p", ptr);
}

void HighPerfMemoryPool::DynamicRecycle() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    uint64_t current_time = GetCurrentTimeMs();
    
    // 遍历所有内存块，释放长时间未使用的块
    auto it = memory_blocks_.begin();
    while (it != memory_blocks_.end()) {
        MemoryBlock* block = *it;
        
        // 检查块是否完全空闲
        if (block->used_size == 0) {
            // 释放块
            ReleaseBlock(block);
            delete block;
            it = memory_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    
    ALOG_INFO_F("MemoryPool::DynamicRecycle completed");
}

void HighPerfMemoryPool::GetMemoryStats(size_t& total_alloc, size_t& used, size_t& free) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    total_alloc = 0;
    used = 0;
    
    for (const MemoryBlock* block : memory_blocks_) {
        total_alloc += block->block_size;
        used += block->used_size;
    }
    
    free = total_alloc - used;
}

uint64_t HighPerfMemoryPool::AlignMemorySize(uint64_t size) const {
    return (size + MIN_ALIGN_SIZE - 1) & ~(MIN_ALIGN_SIZE - 1);
}

void* HighPerfMemoryPool::TryAllocateFromExistingBlocks(uint64_t align_size) {
    // 遍历所有内存块，尝试分配
    for (MemoryBlock* block : memory_blocks_) {
        void* ptr = AllocateFromBlock(block, align_size);
        if (ptr != nullptr) {
            return ptr;
        }
    }
    return nullptr;
}

MemoryBlock* HighPerfMemoryPool::Allocate1GBBlock(uint64_t align_size) {
    uint8_t* dev_addr = nullptr;
    
    // 1. 尝试获取1GB大页内存
    if (TryGetHugePageMem(&dev_addr, align_size)) {
        MemoryBlock* block = new MemoryBlock(dev_addr, ONT_GB_SIZE, true);
        InitFreeList(block);
        memory_blocks_.push_back(block);
        allocated_dev_addrs_.emplace_back(dev_addr);
        huge_page_vec_.emplace_back(HugePageDesc(dev_addr, ONT_GB_SIZE, true, GetCurrentTimeMs()));
        return block;
    }
    
    // 2. 尝试使用rtMalloc分配1GB大页
    size_t alloc_size = ((align_size - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;
    int res = rtMalloc((void**)&dev_addr, alloc_size, ONT_GB_HUGE_PAGE_FLAGS, 0);
    if (res == 0) {
        MemoryBlock* block = new MemoryBlock(dev_addr, alloc_size, true);
        InitFreeList(block);
        memory_blocks_.push_back(block);
        allocated_dev_addrs_.emplace_back(dev_addr);
        huge_page_vec_.emplace_back(HugePageDesc(dev_addr, alloc_size, true, GetCurrentTimeMs()));
        return block;
    }
    
    ALOG_WARN_F("1G page mem alloc failed, turn to 2M page.");
    return nullptr;
}

MemoryBlock* HighPerfMemoryPool::Allocate2MBBlock(uint64_t align_size) {
    uint8_t* dev_addr = nullptr;
    
    // 尝试使用rtMalloc分配2MB大页
    int res = rtMalloc((void**)&dev_addr, align_size, TWO_MB_HUGE_PAGE_FLAGS, 0);
    if (res == 0) {
        MemoryBlock* block = new MemoryBlock(dev_addr, align_size, false);
        InitFreeList(block);
        memory_blocks_.push_back(block);
        allocated_dev_addrs_.emplace_back(dev_addr);
        huge_page_vec_.emplace_back(HugePageDesc(dev_addr, align_size, true, GetCurrentTimeMs()));
        return block;
    }
    
    ALOG_ERROR_F("2M page mem alloc failed");
    return nullptr;
}

void HighPerfMemoryPool::InitFreeList(MemoryBlock* block) {
    // 初始化空闲列表，将整个块作为一个空闲节点
    block->free_list = reinterpret_cast<typename MemoryBlock::FreeNode*>(block->base_addr);
    block->free_list->size = block->block_size - sizeof(typename MemoryBlock::FreeNode);
    block->free_list->next = nullptr;
}

void* HighPerfMemoryPool::AllocateFromBlock(MemoryBlock* block, uint64_t align_size) {
    typename MemoryBlock::FreeNode* prev = nullptr;
    typename MemoryBlock::FreeNode* curr = block->free_list;
    
    // 查找足够大的空闲块
    while (curr != nullptr) {
        if (curr->size >= align_size) {
            // 找到足够大的块
            void* ptr = curr + 1; // 跳过FreeNode头部
            
            // 如果剩余空间足够大，分割成两个块
            size_t remaining_size = curr->size - align_size;
            if (remaining_size >= sizeof(typename MemoryBlock::FreeNode) + MIN_ALIGN_SIZE) {
                // 分割块
                typename MemoryBlock::FreeNode* new_node = reinterpret_cast<typename MemoryBlock::FreeNode*>(
                    static_cast<char*>(ptr) + align_size);
                new_node->size = remaining_size - sizeof(typename MemoryBlock::FreeNode);
                new_node->next = curr->next;
                
                if (prev == nullptr) {
                    block->free_list = new_node;
                } else {
                    prev->next = new_node;
                }
            } else {
                // 整个块都分配出去
                if (prev == nullptr) {
                    block->free_list = curr->next;
                } else {
                    prev->next = curr->next;
                }
            }
            
            // 更新块的使用情况
            block->used_size += align_size;
            block->is_allocated = true;
            
            // 记录地址到块的映射
            addr_to_block_[ptr] = block;
            
            ALOG_INFO_F("Allocate from block %p, ptr %p, size %lu", block->base_addr, ptr, align_size);
            return ptr;
        }
        
        prev = curr;
        curr = curr->next;
    }
    
    return nullptr; // 没有找到足够大的块
}

void HighPerfMemoryPool::FreeToBlock(MemoryBlock* block, void* ptr, uint64_t size) {
    // 创建新的空闲节点
    typename MemoryBlock::FreeNode* new_node = reinterpret_cast<typename MemoryBlock::FreeNode*>(
        static_cast<char*>(ptr) - sizeof(typename MemoryBlock::FreeNode));
    
    new_node->size = size + sizeof(typename MemoryBlock::FreeNode);
    
    // 将新节点插入到空闲列表的合适位置
    typename MemoryBlock::FreeNode* prev = nullptr;
    typename MemoryBlock::FreeNode* curr = block->free_list;
    
    // 找到插入位置（按地址排序）
    while (curr != nullptr && curr < new_node) {
        prev = curr;
        curr = curr->next;
    }
    
    // 插入新节点
    new_node->next = curr;
    if (prev == nullptr) {
        block->free_list = new_node;
    } else {
        prev->next = new_node;
    }
        
    // 更新块的使用情况
    block->used_size -= size;
    if (block->used_size == 0) {
        block->is_allocated = false;
    }

    // 合并相邻的空闲块
    MergeFreeBlocks(block);
}

void HighPerfMemoryPool::MergeFreeBlocks(MemoryBlock* block) {
    typename MemoryBlock::FreeNode* curr = block->free_list;
    while (curr != nullptr && curr->next != nullptr) {
        // 检查当前块和下一个块是否相邻
        char* curr_end = reinterpret_cast<char*>(curr + 1) + curr->size;
        if (curr_end == reinterpret_cast<char*>(curr->next)) {
            // 合并两个块
            curr->size += curr->next->size + sizeof(typename MemoryBlock::FreeNode);
            curr->next = curr->next->next;
        } else {
            curr = curr->next;
        }
    }
}

void HighPerfMemoryPool::ReleaseBlock(MemoryBlock* block) {
    // 释放块的内存
    if (block->is_huge_1g) {
        rtFree(block->base_addr, block->block_size);
    } else {
        rtFree(block->base_addr, block->block_size);
    }
    
    // 从设备地址列表中移除
    auto it = std::find(allocated_dev_addrs_.begin(), allocated_dev_addrs_.end(), block->base_addr);
    if (it != allocated_dev_addrs_.end()) {
        allocated_dev_addrs_.erase(it);
    }
    
    // 从大页描述符中移除
    auto hp_it = std::find_if(huge_page_vec_.begin(), huge_page_vec_.end(),
        [block](const HugePageDesc& desc) { return desc.addr == block->base_addr; });
    if (hp_it != huge_page_vec_.end()) {
        huge_page_vec_.erase(hp_it);
    }
    
    // 移除所有相关的地址映射
    for (auto it = addr_to_block_.begin(); it != addr_to_block_.end();) {
        if (it->second == block) {
            it = addr_to_block_.erase(it);
        } else {
            ++it;
        }
    }
}

uint64_t HighPerfMemoryPool::GetCurrentTimeMs() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}
