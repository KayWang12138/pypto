/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NPUCOMM Atomic Operations Test
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <atomic>
#include "../npucomm.h"

using namespace npu::comm;

void test_basic_atomic_operations() {
    std::cout << "Testing basic atomic operations...\n";

    int64_t counter = 0;
    int64_t value = 42;

    // Test atomic fetch-add (local)
    int64_t old_value = NpuComm::atomic_fetch_add(&counter, value, NpuComm::my_pe());
    assert(old_value == 0);
    assert(counter == value);

    // Test another fetch-add
    old_value = NpuComm::atomic_fetch_add(&counter, value, NpuComm::my_pe());
    assert(old_value == value);
    assert(counter == value * 2);

    // Test atomic compare-swap (local)
    old_value = NpuComm::atomic_compare_swap(&counter, value * 2, 100, NpuComm::my_pe());
    assert(old_value == value * 2);
    assert(counter == 100);

    // Test compare-swap failure case
    old_value = NpuComm::atomic_compare_swap(&counter, 50, 200, NpuComm::my_pe());
    assert(old_value == 100); // Should return current value
    assert(counter == 100);   // Should remain unchanged

    std::cout << "✓ Basic atomic operations test passed\n";
}

void test_atomic_fetch_and_or() {
    std::cout << "Testing atomic fetch-and/or operations...\n";

    int64_t value = 0xFFFF; // All bits set

    // Test fetch-and
    int64_t mask = 0x00FF;
    int64_t old_value = NpuComm::atomic_fetch_and(&value, mask, NpuComm::my_pe());
    assert(old_value == 0xFFFF);
    assert(value == (0xFFFF & mask));

    // Test fetch-or
    value = 0x000F;
    int64_t or_mask = 0xF000;
    old_value = NpuComm::atomic_fetch_or(&value, or_mask, NpuComm::my_pe());
    assert(old_value == 0x000F);
    assert(value == (0x000F | or_mask));

    std::cout << "✓ Atomic fetch-and/or test passed\n";
}

void test_atomic_stress() {
    std::cout << "Testing atomic operations stress test...\n";

    const int NUM_OPERATIONS = 10000;
    int64_t counter = 0;

    // Perform many atomic operations
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        NpuComm::atomic_fetch_add(&counter, 1, NpuComm::my_pe());
    }

    assert(counter == NUM_OPERATIONS);

    // Test concurrent-like operations (simulated)
    counter = 0;
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        int64_t expected = i;
        int64_t desired = i + 1;

        // Use compare-swap to implement atomic increment
        while (true) {
            int64_t current = NpuComm::atomic_compare_swap(&counter, expected, desired, NpuComm::my_pe());
            if (current == expected) {
                break; // Success
            }
            // Update expected and try again
            expected = current;
            desired = current + 1;
        }
    }

    assert(counter == NUM_OPERATIONS);

    std::cout << "✓ Atomic stress test passed\n";
}

void test_atomic_performance() {
    std::cout << "Testing atomic operations performance...\n";

    const int NUM_OPERATIONS = 100000;
    int64_t counter = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        NpuComm::atomic_fetch_add(&counter, 1, NpuComm::my_pe());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double ops_per_sec = NUM_OPERATIONS * 1000.0 / duration.count();
    std::cout << "  Completed " << NUM_OPERATIONS << " atomic operations in " << duration.count() << " ms\n";
    std::cout << "  Atomic ops/sec: " << ops_per_sec << "\n";
    assert(counter == NUM_OPERATIONS);

    std::cout << "✓ Atomic performance test passed\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== NPUCOMM Atomic Operations Test ===\n";

    try {
        test_basic_atomic_operations();
        test_atomic_fetch_and_or();
        test_atomic_stress();
        test_atomic_performance();

        std::cout << "\n🎉 All atomic tests passed!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Atomic test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
