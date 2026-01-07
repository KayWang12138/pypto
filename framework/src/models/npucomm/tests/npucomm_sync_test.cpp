/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NPUCOMM Synchronization Test
 */

#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <cstring>
#include "../npucomm.h"

using namespace npu::comm;

void test_synchronization_primitives() {
    std::cout << "Testing synchronization primitives...\n";

    // Test fence
    CommStatus status = NpuComm::fence();
    assert(status == COMM_SUCCESS);

    // Test quiet
    status = NpuComm::quiet();
    assert(status == COMM_SUCCESS);

    // Test barrier (single PE case)
    status = NpuComm::barrier_all();
    assert(status == COMM_SUCCESS);

    std::cout << "✓ Synchronization primitives test passed\n";
}

void test_memory_consistency() {
    std::cout << "Testing memory consistency operations...\n";

    const size_t BUF_SIZE = 1024;
    void* buffer = NpuComm::shmalloc(BUF_SIZE);
    assert(buffer != nullptr);

    // Initialize buffer
    memset(buffer, 0xAA, BUF_SIZE);

    // Test memory operations with consistency
    NpuComm::fence();  // Memory fence before access

    // Modify data
    memset(buffer, 0xBB, BUF_SIZE);

    NpuComm::fence();  // Memory fence after modification

    // Verify data consistency
    for (size_t i = 0; i < BUF_SIZE; ++i) {
        assert(static_cast<unsigned char*>(buffer)[i] == 0xBB);
    }

    NpuComm::shfree(buffer);

    std::cout << "✓ Memory consistency test passed\n";
}

void test_barrier_performance() {
    std::cout << "Testing barrier performance...\n";

    const int NUM_BARRIERS = 100;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_BARRIERS; ++i) {
        assert(NpuComm::barrier_all() == COMM_SUCCESS);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double barriers_per_sec = NUM_BARRIERS * 1000.0 / duration.count();
    std::cout << "  Completed " << NUM_BARRIERS << " barriers in " << duration.count() << " ms\n";
    std::cout << "  Barriers/sec: " << barriers_per_sec << "\n";

    std::cout << "✓ Barrier performance test passed\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== NPUCOMM Synchronization Test ===\n";

    try {
        test_synchronization_primitives();
        test_memory_consistency();
        test_barrier_performance();

        std::cout << "\n🎉 All synchronization tests passed!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Synchronization test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
