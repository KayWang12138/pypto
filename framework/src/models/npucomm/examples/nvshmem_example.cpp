/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NVSHMEM-compatible Example Program
 * Demonstrates NVSHMEM API usage with NPU Communication Library
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <nvshmem.h>

int main(int argc, char* argv[]) {
    std::cout << "=== NVSHMEM-compatible Example with NPU Communication Library ===\n";

    // Initialize NVSHMEM (compatible API)
    int status = nvshmem_init();
    if (status != 0) {
        std::cerr << "Failed to initialize NVSHMEM\n";
        return 1;
    }

    std::cout << "Initialized as PE " << nvshmem_my_pe() << " of " << nvshmem_n_pes() << " PEs\n";
    std::cout << "NVSHMEM version: " << nvshmem_info_get_version() << "\n";
    std::cout << "Library name: " << nvshmem_info_get_name() << "\n";

    // Allocate symmetric memory
    const size_t BUF_SIZE = 1024;
    int64_t* symmetric_buffer = (int64_t*)nvshmem_shmalloc(BUF_SIZE * sizeof(int64_t));
    if (!symmetric_buffer) {
        std::cerr << "Failed to allocate symmetric memory\n";
        nvshmem_finalize();
        return 1;
    }

    std::cout << "Allocated " << BUF_SIZE * sizeof(int64_t) << " bytes of symmetric memory\n";

    // Initialize data
    for (size_t i = 0; i < BUF_SIZE; ++i) {
        symmetric_buffer[i] = nvshmem_my_pe() * 1000 + i;
    }

    // Example 1: Basic put/get operations
    std::cout << "\n=== Basic Put/Get Operations ===\n";

    if (nvshmem_n_pes() > 1) {
        int target_pe = (nvshmem_my_pe() + 1) % nvshmem_n_pes();

        // Put operation
        std::cout << "PE " << nvshmem_my_pe() << " putting data to PE " << target_pe << "\n";
        nvshmem_put64(symmetric_buffer, symmetric_buffer, 10, target_pe);

        // Get operation
        std::cout << "PE " << nvshmem_my_pe() << " getting data from PE " << target_pe << "\n";
        nvshmem_get64(symmetric_buffer + 10, symmetric_buffer, 10, target_pe);
    } else {
        std::cout << "Single PE - demonstrating local operations\n";
        nvshmem_put64(symmetric_buffer + 20, symmetric_buffer, 10, nvshmem_my_pe());
        nvshmem_get64(symmetric_buffer + 30, symmetric_buffer + 20, 10, nvshmem_my_pe());
    }

    // Example 2: Atomic operations
    std::cout << "\n=== Atomic Operations ===\n";

    int64_t counter = 0;
    int64_t old_value = nvshmem_atomic_fetch_add(&counter, 42, nvshmem_my_pe());
    std::cout << "Atomic fetch-add: old_value=" << old_value << ", counter=" << counter << "\n";

    old_value = nvshmem_atomic_compare_swap(&counter, 42, 100, nvshmem_my_pe());
    std::cout << "Atomic compare-swap: old_value=" << old_value << ", counter=" << counter << "\n";

    // Example 3: Single element operations
    std::cout << "\n=== Single Element Operations ===\n";

    int32_t single_value = 0;
    nvshmem_p(&single_value, 12345, nvshmem_my_pe());
    std::cout << "Put single value: " << single_value << "\n";

    int32_t read_value = nvshmem_g(&single_value, nvshmem_my_pe());
    std::cout << "Get single value: " << read_value << "\n";

    // Example 4: Collective operations
    std::cout << "\n=== Collective Operations ===\n";

    std::cout << "Performing global barrier...\n";
    nvshmem_barrier_all();

    // Broadcast example
    int64_t broadcast_value = nvshmem_my_pe();
    std::cout << "Before broadcast: PE " << nvshmem_my_pe() << " has value " << broadcast_value << "\n";

    nvshmem_broadcast(&broadcast_value, &broadcast_value, 1, 0, 0, 0, nvshmem_n_pes());
    std::cout << "After broadcast: PE " << nvshmem_my_pe() << " has value " << broadcast_value << "\n";

    // Example 5: Reduction operations
    std::cout << "\n=== Reduction Operations ===\n";

    int64_t local_sum = nvshmem_my_pe() + 1; // PE 0: 1, PE 1: 2, etc.
    int64_t global_sum = 0;

    std::cout << "Local sum for PE " << nvshmem_my_pe() << ": " << local_sum << "\n";

    nvshmem_sum_reduce(&global_sum, &local_sum, 1, 0, 0, nvshmem_n_pes());
    std::cout << "Global sum: " << global_sum << "\n";

    // Example 6: Memory ordering
    std::cout << "\n=== Memory Ordering ===\n";

    std::cout << "Performing fence operation...\n";
    nvshmem_fence();

    std::cout << "Performing quiet operation...\n";
    nvshmem_quiet();

    // Example 7: Wait/Test operations
    std::cout << "\n=== Synchronization Operations ===\n";

    int64_t sync_var = 0;
    std::cout << "Testing wait_until operation...\n";

    // Set sync_var to 1 on another thread or PE (simplified)
    sync_var = 1;

    int test_result = nvshmem_test(&sync_var, NVSHMEM_CMP_EQ, 1);
    std::cout << "Test result (should be 1): " << test_result << "\n";

    // Cleanup
    std::cout << "\n=== Cleanup ===\n";

    status = nvshmem_shfree(symmetric_buffer);
    if (status == 0) {
        std::cout << "Freed symmetric memory\n";
    }

    status = nvshmem_finalize();
    if (status == 0) {
        std::cout << "NVSHMEM finalized successfully\n";
    }

    std::cout << "\n🎉 NVSHMEM-compatible example completed!\n";
    return 0;
}
