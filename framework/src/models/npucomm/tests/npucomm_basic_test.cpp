/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NPUCOMM Basic Functionality Test
 */

#include <iostream>
#include <cassert>
#include <cstring>
#include "../npucomm.h"

using namespace npu::comm;

void test_initialization() {
    std::cout << "Testing initialization...\n";

    NpuCommConfig config;
    config.numPes = 4;
    config.myPe = 0;

    CommStatus status = NpuComm::init(config);
    assert(status == COMM_SUCCESS);
    assert(NpuComm::my_pe() == 0);
    assert(NpuComm::num_pes() == 4);

    std::cout << "✓ Initialization test passed\n";
}

void test_memory_management() {
    std::cout << "Testing memory management...\n";

    const size_t TEST_SIZE = 1024;
    void* ptr = NpuComm::shmalloc(TEST_SIZE);
    assert(ptr != nullptr);

    // Test that it's in symmetric heap
    // Note: In real implementation, this would check against symmetric heap addresses

    memset(ptr, 0xAA, TEST_SIZE);

    CommStatus status = NpuComm::shfree(ptr);
    assert(status == COMM_SUCCESS);

    std::cout << "✓ Memory management test passed\n";
}

void test_basic_communication() {
    std::cout << "Testing basic communication...\n";

    const size_t MSG_SIZE = 256;
    void* send_buf = NpuComm::shmalloc(MSG_SIZE);
    void* recv_buf = NpuComm::shmalloc(MSG_SIZE);

    assert(send_buf != nullptr && recv_buf != nullptr);

    // Initialize send buffer
    memset(send_buf, 0xBB, MSG_SIZE);
    memset(recv_buf, 0x00, MSG_SIZE);

    // Test local put (PE to itself)
    CommStatus status = NpuComm::put(recv_buf, send_buf, MSG_SIZE, NpuComm::my_pe());
    assert(status == COMM_SUCCESS);
    assert(memcmp(send_buf, recv_buf, MSG_SIZE) == 0);

    // Test local get
    memset(recv_buf, 0x00, MSG_SIZE);
    status = NpuComm::get(recv_buf, send_buf, MSG_SIZE, NpuComm::my_pe());
    assert(status == COMM_SUCCESS);
    assert(memcmp(send_buf, recv_buf, MSG_SIZE) == 0);

    NpuComm::shfree(send_buf);
    NpuComm::shfree(recv_buf);

    std::cout << "✓ Basic communication test passed\n";
}

void test_synchronization() {
    std::cout << "Testing synchronization...\n";

    CommStatus status;

    // Test fence
    status = NpuComm::fence();
    assert(status == COMM_SUCCESS);

    // Test quiet
    status = NpuComm::quiet();
    assert(status == COMM_SUCCESS);

    // Test barrier (single PE case)
    status = NpuComm::barrier_all();
    assert(status == COMM_SUCCESS);

    std::cout << "✓ Synchronization test passed\n";
}

void test_atomic_operations() {
    std::cout << "Testing atomic operations...\n";

    int64_t counter = 0;

    // Test atomic fetch-add (local)
    int64_t old_value = NpuComm::atomic_fetch_add(&counter, 42, NpuComm::my_pe());
    assert(old_value == 0);
    assert(counter == 42);

    // Test atomic compare-swap (local)
    old_value = NpuComm::atomic_compare_swap(&counter, 42, 100, NpuComm::my_pe());
    assert(old_value == 42);
    assert(counter == 100);

    std::cout << "✓ Atomic operations test passed\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== NPUCOMM Basic Functionality Test ===\n";

    try {
        test_initialization();
        test_memory_management();
        test_basic_communication();
        test_synchronization();
        test_atomic_operations();

        CommStatus status = NpuComm::finalize();
        assert(status == COMM_SUCCESS);

        std::cout << "\n🎉 All basic tests passed!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        NpuComm::finalize();
        return 1;
    }
}
