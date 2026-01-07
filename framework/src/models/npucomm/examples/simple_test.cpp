/**
 * Simple NPU Communication Test
 * Standalone test that doesn't depend on full framework
 */

#include <iostream>
#include <cstring>
#include <chrono>

// Simple mock implementation for testing
namespace npu {
namespace comm {

enum CommStatus {
    COMM_SUCCESS = 0,
    COMM_ERROR = -1
};

class NpuComm {
public:
    static CommStatus init() {
        std::cout << "[MOCK] NPU Communication initialized\n";
        return COMM_SUCCESS;
    }

    static CommStatus finalize() {
        std::cout << "[MOCK] NPU Communication finalized\n";
        return COMM_SUCCESS;
    }

    static void* shmalloc(size_t size) {
        void* ptr = malloc(size);
        std::cout << "[MOCK] Allocated " << size << " bytes\n";
        return ptr;
    }

    static CommStatus shfree(void* ptr) {
        if (ptr) {
            free(ptr);
            std::cout << "[MOCK] Freed memory\n";
        }
        return COMM_SUCCESS;
    }

    static CommStatus put(void* dest, const void* src, size_t size, int pe) {
        memcpy(dest, src, size);
        std::cout << "[MOCK] Put " << size << " bytes to PE " << pe << "\n";
        return COMM_SUCCESS;
    }

    static CommStatus get(void* dest, const void* src, size_t size, int pe) {
        memcpy(dest, src, size);
        std::cout << "[MOCK] Get " << size << " bytes from PE " << pe << "\n";
        return COMM_SUCCESS;
    }

    static CommStatus barrier_all() {
        std::cout << "[MOCK] Barrier all\n";
        return COMM_SUCCESS;
    }

    static int64_t atomic_fetch_add(int64_t* dest, int64_t value, int pe) {
        int64_t old = *dest;
        *dest += value;
        std::cout << "[MOCK] Atomic fetch-add " << value << " to PE " << pe << "\n";
        return old;
    }
};

} // namespace comm
} // namespace npu

int main(int argc, char* argv[]) {
    std::cout << "=== Simple NPU Communication Test ===\n";

    // Initialize
    auto status = npu::comm::NpuComm::init();
    if (status != npu::comm::COMM_SUCCESS) {
        std::cerr << "Failed to initialize\n";
        return 1;
    }

    // Allocate memory
    const size_t BUF_SIZE = 1024;
    void* buffer1 = npu::comm::NpuComm::shmalloc(BUF_SIZE);
    void* buffer2 = npu::comm::NpuComm::shmalloc(BUF_SIZE);

    if (!buffer1 || !buffer2) {
        std::cerr << "Failed to allocate memory\n";
        return 1;
    }

    // Initialize data
    memset(buffer1, 0xAA, BUF_SIZE);

    // Test put operation
    auto start_time = std::chrono::high_resolution_clock::now();
    status = npu::comm::NpuComm::put(buffer2, buffer1, BUF_SIZE, 1);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (status == npu::comm::COMM_SUCCESS) {
        std::cout << "Put operation completed in " << duration.count() << " microseconds\n";
    }

    // Test atomic operation
    int64_t counter = 0;
    int64_t old_value = npu::comm::NpuComm::atomic_fetch_add(&counter, 42, 1);
    std::cout << "Atomic operation: old=" << old_value << ", new=" << counter << "\n";

    // Test barrier
    status = npu::comm::NpuComm::barrier_all();
    if (status == npu::comm::COMM_SUCCESS) {
        std::cout << "Barrier completed\n";
    }

    // Cleanup
    npu::comm::NpuComm::shfree(buffer1);
    npu::comm::NpuComm::shfree(buffer2);

    status = npu::comm::NpuComm::finalize();
    if (status == npu::comm::COMM_SUCCESS) {
        std::cout << "Test completed successfully!\n";
    }

    return 0;
}
