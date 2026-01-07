/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Fine-Grain Communication Example
 * Demonstrates NPU-optimized fine-grain communication patterns
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include "../npucomm.h"

using namespace npu::comm;

int main(int argc, char* argv[]) {
    std::cout << "=== NPU Communication Library - Fine-Grain Example ===\n";

    // Initialize with fine-grain optimization
    NpuCommConfig config;
    config.numPes = 8;
    config.myPe = 0;
    config.enableNpuOptimization = true;
    config.enableFineGrainComm = true;
    config.enablePgasMode = true;
    config.maxMessageSize = 64 * 1024; // 64KB max message size
    config.enableDeviceInitiated = true;

    CommStatus status = NpuComm::init(config);
    if (status != COMM_SUCCESS) {
        std::cerr << "Failed to initialize NPU communication library\n";
        return 1;
    }

    std::cout << "Fine-grain communication enabled for PE " << NpuComm::my_pe() << "\n";

    // Allocate fine-grain buffers
    const size_t SMALL_MSG_SIZE = 256;  // 256 bytes - very small messages
    const size_t MEDIUM_MSG_SIZE = 4096; // 4KB - medium messages
    const size_t NUM_OPERATIONS = 1000;

    void* small_buffer = NpuComm::shmalloc(SMALL_MSG_SIZE);
    void* medium_buffer = NpuComm::shmalloc(MEDIUM_MSG_SIZE);

    if (!small_buffer || !medium_buffer) {
        std::cerr << "Failed to allocate fine-grain buffers\n";
        NpuComm::finalize();
        return 1;
    }

    // Initialize with test patterns
    memset(small_buffer, 0xAA, SMALL_MSG_SIZE);
    memset(medium_buffer, 0xBB, MEDIUM_MSG_SIZE);

    std::cout << "\n=== Fine-Grain Communication Patterns ===\n";

    // Pattern 1: Small message burst communication
    std::cout << "Pattern 1: Small message burst (" << NUM_OPERATIONS << " operations)\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        // Alternate between different PEs for load balancing
        int target_pe = (i % (NpuComm::num_pes() - 1)) + 1;

        // Fine-grain put operation (NPU optimized for small messages)
        status = NpuComm::put(small_buffer, small_buffer, SMALL_MSG_SIZE, target_pe);
        if (status != COMM_SUCCESS) {
            std::cerr << "Small message put failed at operation " << i << "\n";
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (status == COMM_SUCCESS) {
        double ops_per_sec = NUM_OPERATIONS * 1000.0 / duration.count();
        double bandwidth = (NUM_OPERATIONS * SMALL_MSG_SIZE) / (1024.0 * 1024.0) / (duration.count() / 1000.0);
        std::cout << "  Completed in " << duration.count() << " ms\n";
        std::cout << "  Operations/sec: " << ops_per_sec << "\n";
        std::cout << "  Bandwidth: " << bandwidth << " MB/s\n";
    }

    // Pattern 2: Medium message streaming
    std::cout << "\nPattern 2: Medium message streaming\n";
    start_time = std::chrono::high_resolution_clock::now();

    const size_t STREAM_OPERATIONS = 100;
    for (size_t i = 0; i < STREAM_OPERATIONS; ++i) {
        int target_pe = (i % 2) + 1; // Alternate between PE 1 and 2

        status = NpuComm::get(medium_buffer, medium_buffer, MEDIUM_MSG_SIZE, target_pe);
        if (status != COMM_SUCCESS) {
            std::cerr << "Medium message get failed at operation " << i << "\n";
            break;
        }
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (status == COMM_SUCCESS) {
        double bandwidth = (STREAM_OPERATIONS * MEDIUM_MSG_SIZE) / (1024.0 * 1024.0) / (duration.count() / 1000.0);
        std::cout << "  Completed in " << duration.count() << " ms\n";
        std::cout << "  Bandwidth: " << bandwidth << " MB/s\n";
    }

    // Pattern 3: Atomic operations for fine-grain synchronization
    std::cout << "\nPattern 3: Fine-grain atomic operations\n";

    int64_t shared_counter = 0;
    const size_t ATOMIC_OPERATIONS = 500;

    start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < ATOMIC_OPERATIONS; ++i) {
        int target_pe = (i % 3) + 1; // Distribute across PEs 1, 2, 3

        // Atomic increment
        int64_t old_value = NpuComm::atomic_fetch_add(&shared_counter, 1, target_pe);

        // Occasionally do compare-and-swap
        if (i % 10 == 0) {
            NpuComm::atomic_compare_swap(&shared_counter, old_value + 1, old_value + 2, target_pe);
        }
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double atomic_ops_per_sec = ATOMIC_OPERATIONS * 1000.0 / duration.count();
    std::cout << "  Atomic operations completed in " << duration.count() << " ms\n";
    std::cout << "  Atomic ops/sec: " << atomic_ops_per_sec << "\n";
    std::cout << "  Final counter value: " << shared_counter << "\n";

    // Pattern 4: Mixed workload (compute + communication overlap)
    std::cout << "\nPattern 4: Mixed workload (compute + communication)\n";

    // Simulate compute-bound work mixed with communication
    const size_t MIXED_OPERATIONS = 50;
    std::vector<uint64_t> compute_results(MIXED_OPERATIONS);

    start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < MIXED_OPERATIONS; ++i) {
        // Start asynchronous communication
        int target_pe = (i % 2) + 1;

        // Simulate compute work (matrix multiplication or similar)
        uint64_t compute_result = 0;
        for (size_t j = 0; j < 10000; ++j) {
            compute_result += j * j;
        }
        compute_results[i] = compute_result;

        // Communication operation
        status = NpuComm::put(medium_buffer, &compute_result, sizeof(uint64_t), target_pe);
        if (status != COMM_SUCCESS) {
            std::cerr << "Mixed workload communication failed at operation " << i << "\n";
            break;
        }
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (status == COMM_SUCCESS) {
        std::cout << "  Mixed workload completed in " << duration.count() << " ms\n";
        std::cout << "  Compute+comm operations/sec: " << MIXED_OPERATIONS * 1000.0 / duration.count() << "\n";
    }

    // Synchronization and cleanup
    std::cout << "\n=== Synchronization and Cleanup ===\n";

    std::cout << "Performing global synchronization...\n";
    status = NpuComm::barrier_all();
    if (status == COMM_SUCCESS) {
        std::cout << "Global barrier completed\n";
    }

    // Free resources
    NpuComm::shfree(small_buffer);
    NpuComm::shfree(medium_buffer);

    NpuComm::finalize();

    std::cout << "Fine-grain communication example completed!\n";
    return 0;
}
