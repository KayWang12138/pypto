/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Basic NPU Communication Example
 * Demonstrates fundamental put/get operations and PGAS memory model
 */

#include <iostream>
#include <vector>
#include <chrono>
#include "../npucomm.h"

using namespace npu::comm;

int main(int argc, char* argv[]) {
    std::cout << "=== NPU Communication Library - Basic Example ===\n";

    // Initialize NPU communication library
    NpuCommConfig config;
    config.numPes = 4;  // Simulate 4 NPU devices
    config.myPe = 0;    // This is PE 0
    config.enableNpuOptimization = true;
    config.enableFineGrainComm = true;
    config.enablePgasMode = true;

    CommStatus status = NpuComm::init(config);
    if (status != COMM_SUCCESS) {
        std::cerr << "Failed to initialize NPU communication library\n";
        return 1;
    }

    std::cout << "Initialized as PE " << NpuComm::my_pe() << " of " << NpuComm::num_pes() << " PEs\n";

    // Allocate symmetric heap memory (PGAS model)
    const size_t BUFFER_SIZE = 1024 * 1024; // 1MB
    void* local_buffer = NpuComm::shmalloc(BUFFER_SIZE);
    void* remote_buffer = NpuComm::shmalloc(BUFFER_SIZE);

    if (!local_buffer || !remote_buffer) {
        std::cerr << "Failed to allocate symmetric heap memory\n";
        NpuComm::finalize();
        return 1;
    }

    std::cout << "Allocated " << BUFFER_SIZE << " bytes symmetric heap memory\n";

    // Initialize local buffer with test data
    uint32_t* local_data = static_cast<uint32_t*>(local_buffer);
    for (size_t i = 0; i < BUFFER_SIZE / sizeof(uint32_t); ++i) {
        local_data[i] = i + NpuComm::my_pe() * 1000; // PE-specific data
    }

    // Demonstrate basic put/get operations
    std::cout << "\n=== Basic Communication Operations ===\n";

    // Example 1: Put operation (single-sided)
    std::cout << "Performing put operation...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    status = NpuComm::put(remote_buffer, local_buffer, 1024, 1); // Put 1KB to PE 1
    if (status != COMM_SUCCESS) {
        std::cerr << "Put operation failed\n";
    } else {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Put operation completed in " << duration.count() << " microseconds\n";
    }

    // Example 2: Get operation (single-sided)
    std::cout << "Performing get operation...\n";
    auto start_time2 = std::chrono::high_resolution_clock::now();

    status = NpuComm::get(local_buffer, remote_buffer, 1024, 1); // Get 1KB from PE 1
    if (status != COMM_SUCCESS) {
        std::cerr << "Get operation failed\n";
    } else {
        auto end_time2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time2 - start_time2);
        std::cout << "Get operation completed in " << duration2.count() << " microseconds\n";
    }

    // Example 3: Atomic operations
    std::cout << "\n=== Atomic Operations ===\n";
    int64_t counter = 0;
    int64_t old_value = NpuComm::atomic_fetch_add(&counter, 42, 1);
    std::cout << "Atomic fetch-add: old_value=" << old_value << ", new_value=" << counter << "\n";

    old_value = NpuComm::atomic_compare_swap(&counter, 42, 100, 1);
    std::cout << "Atomic compare-swap: old_value=" << old_value << ", current_value=" << counter << "\n";

    // Example 4: Synchronization
    std::cout << "\n=== Synchronization Operations ===\n";

    std::cout << "Performing memory fence...\n";
    status = NpuComm::fence();
    if (status == COMM_SUCCESS) {
        std::cout << "Memory fence completed\n";
    }

    std::cout << "Performing barrier synchronization...\n";
    auto start_time3 = std::chrono::high_resolution_clock::now();
    status = NpuComm::barrier_all();
    if (status == COMM_SUCCESS) {
        auto end_time3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3 - start_time3);
        std::cout << "Barrier synchronization completed in " << duration3.count() << " milliseconds\n";
    }

    // Example 5: Fine-grain communication (using internal classes)
    std::cout << "\n=== Fine-Grain Communication ===\n";

    NpuComm& comm = NpuComm::getInstance();
    // Note: In a real implementation, we'd access the fine-grain interface
    // This is just a demonstration of the concept

    // Cleanup
    std::cout << "\n=== Cleanup ===\n";

    status = NpuComm::shfree(local_buffer);
    if (status == COMM_SUCCESS) {
        std::cout << "Freed local buffer\n";
    }

    status = NpuComm::shfree(remote_buffer);
    if (status == COMM_SUCCESS) {
        std::cout << "Freed remote buffer\n";
    }

    status = NpuComm::finalize();
    if (status == COMM_SUCCESS) {
        std::cout << "NPU communication library finalized successfully\n";
    }

    std::cout << "Example completed successfully!\n";
    return 0;
}
