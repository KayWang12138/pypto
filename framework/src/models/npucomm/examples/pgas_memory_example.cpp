/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * PGAS Memory Model Example
 * Demonstrates Partitioned Global Address Space memory management
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include "../npucomm.h"

using namespace npu::comm;

struct GlobalData {
    int32_t integers[100];
    double doubles[50];
    char strings[200];
};

// Simulate a distributed data structure
struct DistributedMatrix {
    float* data;
    size_t rows;
    size_t cols;
    size_t total_elements;

    DistributedMatrix(size_t r, size_t c) : rows(r), cols(c), total_elements(r * c) {
        data = nullptr; // Will be allocated in symmetric heap
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== NPU Communication Library - PGAS Memory Model Example ===\n";

    // Initialize with PGAS focus
    NpuCommConfig config;
    config.numPes = 4;
    config.myPe = 0;
    config.symmetricHeapSize = 2ULL * 1024 * 1024 * 1024; // 2GB symmetric heap
    config.enablePgasMode = true;
    config.enableNpuOptimization = true;

    CommStatus status = NpuComm::init(config);
    if (status != COMM_SUCCESS) {
        std::cerr << "Failed to initialize NPU communication library\n";
        return 1;
    }

    std::cout << "PGAS memory model initialized for PE " << NpuComm::my_pe() << "\n";
    std::cout << "Symmetric heap size: " << config.symmetricHeapSize / (1024*1024*1024) << " GB\n";

    // Example 1: Basic PGAS memory allocation and access
    std::cout << "\n=== Basic PGAS Memory Operations ===\n";

    // Allocate global data structure in symmetric heap
    GlobalData* global_data = static_cast<GlobalData*>(NpuComm::shmalloc(sizeof(GlobalData)));
    if (!global_data) {
        std::cerr << "Failed to allocate global data structure\n";
        NpuComm::finalize();
        return 1;
    }

    std::cout << "Allocated global data structure of " << sizeof(GlobalData) << " bytes\n";

    // Initialize local portion of global data
    for (size_t i = 0; i < 100; ++i) {
        global_data->integers[i] = NpuComm::my_pe() * 1000 + i;
    }

    for (size_t i = 0; i < 50; ++i) {
        global_data->doubles[i] = NpuComm::my_pe() * 1.5 + i * 0.1;
    }

    sprintf(global_data->strings, "Data from PE %d", NpuComm::my_pe());

    // Demonstrate global address operations
    std::cout << "Local data access:\n";
    std::cout << "  integers[0] = " << global_data->integers[0] << "\n";
    std::cout << "  doubles[0] = " << global_data->doubles[0] << "\n";
    std::cout << "  strings = " << global_data->strings << "\n";

    // Example 2: Distributed matrix operations
    std::cout << "\n=== Distributed Matrix Operations ===\n";

    const size_t MATRIX_ROWS = 1024;
    const size_t MATRIX_COLS = 1024;
    const size_t MATRIX_SIZE = MATRIX_ROWS * MATRIX_COLS * sizeof(float);

    // Allocate distributed matrix in symmetric heap
    DistributedMatrix* dist_matrix = static_cast<DistributedMatrix*>(
        NpuComm::shmalloc(sizeof(DistributedMatrix)));
    float* matrix_data = static_cast<float*>(NpuComm::shmalloc(MATRIX_SIZE));

    if (!dist_matrix || !matrix_data) {
        std::cerr << "Failed to allocate distributed matrix\n";
        NpuComm::shfree(global_data);
        NpuComm::finalize();
        return 1;
    }

    // Initialize matrix metadata
    dist_matrix->data = matrix_data;
    dist_matrix->rows = MATRIX_ROWS;
    dist_matrix->cols = MATRIX_COLS;
    dist_matrix->total_elements = MATRIX_ROWS * MATRIX_COLS;

    std::cout << "Allocated distributed matrix: " << MATRIX_ROWS << "x" << MATRIX_COLS
              << " (" << MATRIX_SIZE / (1024*1024) << " MB)\n";

    // Initialize matrix data (each PE initializes its portion)
    size_t elements_per_pe = dist_matrix->total_elements / NpuComm::num_pes();
    size_t start_idx = NpuComm::my_pe() * elements_per_pe;
    size_t end_idx = (NpuComm::my_pe() + 1) * elements_per_pe;

    for (size_t i = start_idx; i < end_idx; ++i) {
        size_t row = i / MATRIX_COLS;
        size_t col = i % MATRIX_COLS;
        matrix_data[i] = NpuComm::my_pe() * 100.0f + row * 0.1f + col * 0.01f;
    }

    std::cout << "Initialized matrix portion for PE " << NpuComm::my_pe()
              << " (elements " << start_idx << " to " << end_idx << ")\n";

    // Example 3: PGAS-style data exchange
    std::cout << "\n=== PGAS-Style Data Exchange ===\n";

    // Exchange matrix portions between PEs using global addresses
    for (int target_pe = 1; target_pe < NpuComm::num_pes(); ++target_pe) {
        if (target_pe == NpuComm::my_pe()) continue;

        std::cout << "Exchanging data with PE " << target_pe << "...\n";

        // Calculate exchange size (exchange 1/4 of local portion)
        size_t exchange_size = elements_per_pe / 4 * sizeof(float);
        size_t local_offset = start_idx + (target_pe * elements_per_pe / 4) % elements_per_pe;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Put operation to remote PE (using global address)
        status = NpuComm::put(matrix_data + local_offset, matrix_data + local_offset,
                             exchange_size, target_pe);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        if (status == COMM_SUCCESS) {
            std::cout << "  Data exchange completed in " << duration.count() << " microseconds\n";
        } else {
            std::cerr << "  Data exchange failed\n";
        }
    }

    // Example 4: Global reduction operations
    std::cout << "\n=== Global Reduction Operations ===\n";

    // Compute local sum
    double local_sum = 0.0;
    for (size_t i = start_idx; i < end_idx; ++i) {
        local_sum += matrix_data[i];
    }

    std::cout << "Local sum for PE " << NpuComm::my_pe() << ": " << local_sum << "\n";

    // In a full PGAS implementation, we would do global reductions here
    // For now, just demonstrate the concept with barriers
    status = NpuComm::barrier_all();
    if (status == COMM_SUCCESS) {
        std::cout << "Global barrier synchronization completed\n";
    }

    // Example 5: Memory consistency and synchronization
    std::cout << "\n=== Memory Consistency Operations ===\n";

    // Demonstrate fence operations
    std::cout << "Performing memory fence...\n";
    status = NpuComm::fence();
    if (status == COMM_SUCCESS) {
        std::cout << "Memory fence completed\n";
    }

    // Demonstrate quiet operations
    std::cout << "Performing quiet operation (wait for all communications)...\n";
    status = NpuComm::quiet();
    if (status == COMM_SUCCESS) {
        std::cout << "Quiet operation completed\n";
    }

    // Example 6: Dynamic memory management in PGAS
    std::cout << "\n=== Dynamic Memory Management ===\n";

    // Allocate and free memory dynamically
    std::vector<void*> dynamic_allocations;

    for (size_t i = 0; i < 10; ++i) {
        size_t alloc_size = (i + 1) * 1024; // 1KB to 10KB
        void* ptr = NpuComm::shmalloc(alloc_size);
        if (ptr) {
            dynamic_allocations.push_back(ptr);
            memset(ptr, i, alloc_size); // Fill with pattern
            std::cout << "Allocated " << alloc_size << " bytes dynamically\n";
        } else {
            std::cerr << "Failed to allocate " << alloc_size << " bytes\n";
            break;
        }
    }

    // Free dynamic allocations
    for (void* ptr : dynamic_allocations) {
        status = NpuComm::shfree(ptr);
        if (status == COMM_SUCCESS) {
            std::cout << "Freed dynamic allocation\n";
        }
    }

    // Cleanup
    std::cout << "\n=== Cleanup ===\n";

    NpuComm::shfree(global_data);
    NpuComm::shfree(dist_matrix);
    NpuComm::shfree(matrix_data);

    status = NpuComm::finalize();
    if (status == COMM_SUCCESS) {
        std::cout << "PGAS memory model example completed successfully!\n";
    }

    return 0;
}
