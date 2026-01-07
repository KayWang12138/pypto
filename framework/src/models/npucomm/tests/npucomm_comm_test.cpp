/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NPUCOMM Communication Test
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>
#include "../npucomm.h"

using namespace npu::comm;

void test_local_communication() {
    std::cout << "Testing local communication operations...\n";

    const size_t MSG_SIZE = 4096;
    void* send_buf = NpuComm::shmalloc(MSG_SIZE);
    void* recv_buf = NpuComm::shmalloc(MSG_SIZE);

    assert(send_buf != nullptr && recv_buf != nullptr);

    // Initialize send buffer with pattern
    for (size_t i = 0; i < MSG_SIZE; ++i) {
        static_cast<unsigned char*>(send_buf)[i] = i % 256;
    }

    // Clear receive buffer
    memset(recv_buf, 0, MSG_SIZE);

    // Test local put
    assert(NpuComm::put(recv_buf, send_buf, MSG_SIZE, NpuComm::my_pe()) == COMM_SUCCESS);

    // Verify data
    for (size_t i = 0; i < MSG_SIZE; ++i) {
        assert(static_cast<unsigned char*>(recv_buf)[i] == (i % 256));
    }

    // Test local get
    memset(recv_buf, 0, MSG_SIZE);
    assert(NpuComm::get(recv_buf, send_buf, MSG_SIZE, NpuComm::my_pe()) == COMM_SUCCESS);

    // Verify data
    for (size_t i = 0; i < MSG_SIZE; ++i) {
        assert(static_cast<unsigned char*>(recv_buf)[i] == (i % 256));
    }

    NpuComm::shfree(send_buf);
    NpuComm::shfree(recv_buf);

    std::cout << "✓ Local communication test passed\n";
}

void test_non_blocking_operations() {
    std::cout << "Testing non-blocking communication operations...\n";

    const size_t MSG_SIZE = 2048;
    void* send_buf = NpuComm::shmalloc(MSG_SIZE);
    void* recv_buf = NpuComm::shmalloc(MSG_SIZE);

    assert(send_buf != nullptr && recv_buf != nullptr);

    // Initialize send buffer
    memset(send_buf, 0xAB, MSG_SIZE);
    memset(recv_buf, 0x00, MSG_SIZE);

    // Test non-blocking put
    // Note: put_nbi returns void in current API, cannot check return values
    NpuComm::put_nbi(recv_buf, send_buf, MSG_SIZE, NpuComm::my_pe());

    // Test non-blocking get
    memset(recv_buf, 0x00, MSG_SIZE);
    // Note: get_nbi returns void in current API, cannot check return values
    NpuComm::get_nbi(recv_buf, send_buf, MSG_SIZE, NpuComm::my_pe());

    // Wait for completion
    assert(NpuComm::quiet() == COMM_SUCCESS);

    // Verify data
    for (size_t i = 0; i < MSG_SIZE; ++i) {
        assert(static_cast<unsigned char*>(recv_buf)[i] == 0xAB);
    }

    NpuComm::shfree(send_buf);
    NpuComm::shfree(recv_buf);

    std::cout << "✓ Non-blocking communication test passed\n";
}

void test_message_sizes() {
    std::cout << "Testing various message sizes...\n";

    std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384, 65536};

    for (size_t size : sizes) {
        void* send_buf = NpuComm::shmalloc(size);
        void* recv_buf = NpuComm::shmalloc(size);

        assert(send_buf != nullptr && recv_buf != nullptr);

        // Fill with pattern
        memset(send_buf, 0xCD, size);
        memset(recv_buf, 0x00, size);

        // Test communication
        assert(NpuComm::put(recv_buf, send_buf, size, NpuComm::my_pe()) == COMM_SUCCESS);

        // Verify
        for (size_t i = 0; i < size; ++i) {
            assert(static_cast<unsigned char*>(recv_buf)[i] == 0xCD);
        }

        NpuComm::shfree(send_buf);
        NpuComm::shfree(recv_buf);

        std::cout << "  ✓ Message size " << size << " bytes test passed\n";
    }

    std::cout << "✓ Message sizes test passed\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== NPUCOMM Communication Test ===\n";

    try {
        test_local_communication();
        test_non_blocking_operations();
        test_message_sizes();

        std::cout << "\n🎉 All communication tests passed!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Communication test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
