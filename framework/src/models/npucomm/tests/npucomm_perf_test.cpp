/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NPUCOMM Performance Regression Test
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <cassert>
#include <cstring>
#include "../npucomm.h"

using namespace npu::comm;

void test_put_performance() {
    std::cout << "Testing PUT operation performance...\n";

    const size_t MSG_SIZE = 4096;
    const int NUM_ITERATIONS = 1000;

    void* send_buf = NpuComm::shmalloc(MSG_SIZE);
    void* recv_buf = NpuComm::shmalloc(MSG_SIZE);

    assert(send_buf != nullptr && recv_buf != nullptr);

    // Initialize send buffer
    memset(send_buf, 0xAB, MSG_SIZE);

    std::vector<double> latencies;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        CommStatus status = NpuComm::put(recv_buf, send_buf, MSG_SIZE, NpuComm::my_pe());
        assert(status == COMM_SUCCESS);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        latencies.push_back(duration.count());
    }

    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());
    double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double min_latency = latencies.front();
    double max_latency = latencies.back();
    double p50_latency = latencies[latencies.size() / 2];
    double p95_latency = latencies[latencies.size() * 95 / 100];

    // Calculate bandwidth
    double total_bytes = MSG_SIZE * NUM_ITERATIONS;
    double total_time_sec = avg_latency * NUM_ITERATIONS / 1e6;
    double bandwidth_mbps = (total_bytes / (1024.0 * 1024.0)) / total_time_sec;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Message size: " << MSG_SIZE << " bytes\n";
    std::cout << "  Iterations: " << NUM_ITERATIONS << "\n";
    std::cout << "  Avg latency: " << avg_latency << " us\n";
    std::cout << "  Min latency: " << min_latency << " us\n";
    std::cout << "  Max latency: " << max_latency << " us\n";
    std::cout << "  P50 latency: " << p50_latency << " us\n";
    std::cout << "  P95 latency: " << p95_latency << " us\n";
    std::cout << "  Bandwidth: " << bandwidth_mbps << " MB/s\n";

    // Performance assertions (basic regression test)
    assert(avg_latency < 1000.0); // Should be less than 1ms
    assert(bandwidth_mbps > 100.0); // Should be at least 100 MB/s

    NpuComm::shfree(send_buf);
    NpuComm::shfree(recv_buf);

    std::cout << "✓ PUT performance test passed\n";
}

void test_get_performance() {
    std::cout << "Testing GET operation performance...\n";

    const size_t MSG_SIZE = 4096;
    const int NUM_ITERATIONS = 1000;

    void* send_buf = NpuComm::shmalloc(MSG_SIZE);
    void* recv_buf = NpuComm::shmalloc(MSG_SIZE);

    assert(send_buf != nullptr && recv_buf != nullptr);

    // Initialize send buffer
    memset(send_buf, 0xCD, MSG_SIZE);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        CommStatus status = NpuComm::get(recv_buf, send_buf, MSG_SIZE, NpuComm::my_pe());
        assert(status == COMM_SUCCESS);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double ops_per_sec = NUM_ITERATIONS * 1000.0 / duration.count();
    double bandwidth_mbps = (MSG_SIZE * NUM_ITERATIONS) / (1024.0 * 1024.0) / (duration.count() / 1000.0);

    std::cout << "  Completed " << NUM_ITERATIONS << " GET operations in " << duration.count() << " ms\n";
    std::cout << "  GET ops/sec: " << ops_per_sec << "\n";
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_mbps << " MB/s\n";

    // Verify data
    for (size_t i = 0; i < MSG_SIZE; ++i) {
        assert(static_cast<unsigned char*>(recv_buf)[i] == 0xCD);
    }

    NpuComm::shfree(send_buf);
    NpuComm::shfree(recv_buf);

    std::cout << "✓ GET performance test passed\n";
}

void test_atomic_performance() {
    std::cout << "Testing atomic operation performance...\n";

    const int NUM_OPERATIONS = 50000;
    int64_t counter = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        NpuComm::atomic_fetch_add(&counter, 1, NpuComm::my_pe());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double atomic_ops_per_sec = NUM_OPERATIONS * 1000.0 / duration.count();

    std::cout << "  Completed " << NUM_OPERATIONS << " atomic operations in " << duration.count() << " ms\n";
    std::cout << "  Atomic ops/sec: " << atomic_ops_per_sec << "\n";
    std::cout << "  Final counter value: " << counter << "\n";

    assert(counter == NUM_OPERATIONS);
    assert(atomic_ops_per_sec > 10000.0); // Should be at least 10K ops/sec

    std::cout << "✓ Atomic performance test passed\n";
}

void test_barrier_performance() {
    std::cout << "Testing barrier performance...\n";

    const int NUM_BARRIERS = 1000;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_BARRIERS; ++i) {
        assert(NpuComm::barrier_all() == COMM_SUCCESS);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double barriers_per_sec = NUM_BARRIERS * 1000.0 / duration.count();

    std::cout << "  Completed " << NUM_BARRIERS << " barriers in " << duration.count() << " ms\n";
    std::cout << "  Barriers/sec: " << barriers_per_sec << "\n";

    assert(barriers_per_sec > 100.0); // Should be at least 100 barriers/sec

    std::cout << "✓ Barrier performance test passed\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== NPUCOMM Performance Regression Test ===\n";

    try {
        test_put_performance();
        test_get_performance();
        test_atomic_performance();
        test_barrier_performance();

        std::cout << "\n🎉 All performance tests passed!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Performance test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
