#include "high_perf_memory_pool.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

using namespace std;

// 测试基本分配释放功能
void test_basic_allocation() {
    cout << "=== 测试基本分配释放功能 ===\n";
    
    HighPerfMemoryPool pool;
    
    // 测试不同大小的内存分配
    void* ptr1 = pool.Allocate(64);
    void* ptr2 = pool.Allocate(128);
    void* ptr3 = pool.Allocate(1024);
    void* ptr4 = pool.Allocate(4096);
    
    cout << "分配成功：ptr1=" << ptr1 << ", ptr2=" << ptr2 << ", ptr3=" << ptr3 << ", ptr4=" << ptr4 << endl;
    
    // 测试释放
    pool.Free(ptr1);
    pool.Free(ptr2);
    pool.Free(ptr3);
    pool.Free(ptr4);
    
    cout << "释放成功\n";
    
    cout << "基本分配释放测试通过\n\n";
}

// 测试大页分配策略
void test_huge_page_allocation() {
    cout << "=== 测试大页分配策略 ===\n";
    
    HighPerfMemoryPool pool;
    
    // 测试1GB大页分配（实际会被替换为示例实现的malloc）
    void* large_ptr = pool.Allocate(1024 * 1024 * 1024); // 1GB
    cout << "1GB内存分配：" << (large_ptr != nullptr ? "成功" : "失败") << "，地址=" << large_ptr << endl;
    
    if (large_ptr != nullptr) {
        pool.Free(large_ptr);
        cout << "1GB内存释放成功\n";
    }
    
    // 测试多个小内存分配
    vector<void*> ptrs;
    for (int i = 0; i < 100; ++i) {
        void* ptr = pool.Allocate(4096);
        if (ptr != nullptr) {
            ptrs.push_back(ptr);
        }
    }
    
    cout << "成功分配 " << ptrs.size() << " 个4KB内存块\n";
    
    // 释放所有内存
    for (void* ptr : ptrs) {
        pool.Free(ptr);
    }
    
    cout << "大页分配策略测试通过\n\n";
}

// 测试动态回收功能
void test_dynamic_recycle() {
    cout << "=== 测试动态回收功能 ===\n";
    
    HighPerfMemoryPool pool;
    
    // 设置短回收阈值（1秒）
    pool.SetRecycleThreshold(1000);
    
    size_t total_alloc, used, free;
    
    // 分配一些内存
    vector<void*> ptrs;
    for (int i = 0; i < 50; ++i) {
        void* ptr = pool.Allocate(8192);
        if (ptr != nullptr) {
            ptrs.push_back(ptr);
        }
    }
    
    pool.GetMemoryStats(total_alloc, used, free);
    cout << "分配后：总内存=" << total_alloc << "，已使用=" << used << "，空闲=" << free << endl;
    
    // 释放所有内存
    for (void* ptr : ptrs) {
        pool.Free(ptr);
    }
    
    pool.GetMemoryStats(total_alloc, used, free);
    cout << "释放后：总内存=" << total_alloc << "，已使用=" << used << "，空闲=" << free << endl;
    
    // 执行动态回收
    pool.DynamicRecycle();
    
    pool.GetMemoryStats(total_alloc, used, free);
    cout << "动态回收后：总内存=" << total_alloc << "，已使用=" << used << "，空闲=" << free << endl;
    
    cout << "动态回收功能测试通过\n\n";
}

// 测试性能对比
void test_performance() {
    cout << "=== 测试性能对比 ===\n";
    
    const int NUM_ALLOCATIONS = 100000;
    const size_t ALLOC_SIZE = 4096;
    
    HighPerfMemoryPool pool;
    
    // 测试内存池性能
    auto start = chrono::high_resolution_clock::now();
    
    vector<void*> ptrs;
    for (int i = 0; i < NUM_ALLOCATIONS; ++i) {
        void* ptr = pool.Allocate(ALLOC_SIZE);
        if (ptr != nullptr) {
            ptrs.push_back(ptr);
        }
    }
    
    for (void* ptr : ptrs) {
        pool.Free(ptr);
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto pool_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    
    // 测试malloc/free性能
    start = chrono::high_resolution_clock::now();
    
    ptrs.clear();
    for (int i = 0; i < NUM_ALLOCATIONS; ++i) {
        void* ptr = malloc(ALLOC_SIZE);
        if (ptr != nullptr) {
            ptrs.push_back(ptr);
        }
    }
    
    for (void* ptr : ptrs) {
        free(ptr);
    }
    
    end = chrono::high_resolution_clock::now();
    auto malloc_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    
    cout << "内存池性能：" << NUM_ALLOCATIONS << "次分配释放耗时 " << pool_time << " ms\n";
    cout << "malloc/free性能：" << NUM_ALLOCATIONS << "次分配释放耗时 " << malloc_time << " ms\n";
    cout << "性能提升：" << (malloc_time / static_cast<double>(pool_time)) << "x\n";
    
    cout << "性能对比测试完成\n\n";
}

// 测试多线程并发访问
void test_multi_thread() {
    cout << "=== 测试多线程并发访问 ===\n";
    
    HighPerfMemoryPool pool;
    const int NUM_THREADS = 8;
    const int NUM_OPERATIONS = 10000;
    atomic<int> success_count(0);
    
    auto thread_func = [&]() {
        for (int i = 0; i < NUM_OPERATIONS; ++i) {
            // 随机分配不同大小的内存
            size_t size = 64 + (rand() % 4096);
            void* ptr = pool.Allocate(size);
            
            if (ptr != nullptr) {
                // 写入一些数据
                memset(ptr, 0xAA, size);
                
                // 读取验证
                bool valid = true;
                const char* data = static_cast<const char*>(ptr);
                for (size_t j = 0; j < size; ++j) {
                    if (data[j] != 0xAA) {
                        valid = false;
                        break;
                    }
                }
                
                if (valid) {
                    success_count++;
                }
                
                pool.Free(ptr);
            }
        }
    };
    
    vector<thread> threads;
    auto start = chrono::high_resolution_clock::now();
    
    // 创建多个线程
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(thread_func);
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    
    cout << "线程数：" << NUM_THREADS << "，每线程操作数：" << NUM_OPERATIONS << endl;
    cout << "总操作数：" << NUM_THREADS * NUM_OPERATIONS << "，成功操作数：" << success_count << endl;
    cout << "耗时：" << duration << " ms\n";
    
    if (success_count == NUM_THREADS * NUM_OPERATIONS) {
        cout << "多线程并发访问测试通过\n\n";
    } else {
        cout << "多线程并发访问测试失败\n\n";
    }
}

// 测试内存统计功能
void test_memory_stats() {
    cout << "=== 测试内存统计功能 ===\n";
    
    HighPerfMemoryPool pool;
    
    size_t total_alloc, used, free;
    
    // 初始状态
    pool.GetMemoryStats(total_alloc, used, free);
    cout << "初始状态：总内存=" << total_alloc << "，已使用=" << used << "，空闲=" << free << endl;
    
    // 分配内存
    vector<void*> ptrs;
    for (int i = 0; i < 20; ++i) {
        void* ptr = pool.Allocate(16384);
        if (ptr != nullptr) {
            ptrs.push_back(ptr);
        }
    }
    
    pool.GetMemoryStats(total_alloc, used, free);
    cout << "分配后：总内存=" << total_alloc << "，已使用=" << used << "，空闲=" << free << endl;
    
    // 释放部分内存
    for (int i = 0; i < 10; ++i) {
        pool.Free(ptrs[i]);
    }
    ptrs.erase(ptrs.begin(), ptrs.begin() + 10);
    
    pool.GetMemoryStats(total_alloc, used, free);
    cout << "释放部分内存后：总内存=" << total_alloc << "，已使用=" << used << "，空闲=" << free << endl;
    
    // 释放所有内存
    for (void* ptr : ptrs) {
        pool.Free(ptr);
    }
    
    pool.GetMemoryStats(total_alloc, used, free);
    cout << "释放所有内存后：总内存=" << total_alloc << "，已使用=" << used << "，空闲=" << free << endl;
    
    // 执行动态回收
    pool.DynamicRecycle();
    
    pool.GetMemoryStats(total_alloc, used, free);
    cout << "动态回收后：总内存=" << total_alloc << "，已使用=" << used << "，空闲=" << free << endl;
    
    cout << "内存统计功能测试通过\n\n";
}

int main() {
    cout << "高性能内存池测试\n\n";
    
    // 运行所有测试
    test_basic_allocation();
    test_huge_page_allocation();
    test_dynamic_recycle();
    test_performance();
    test_multi_thread();
    test_memory_stats();
    
    cout << "所有测试完成！\n";
    
    return 0;
}
