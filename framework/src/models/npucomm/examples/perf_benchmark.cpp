/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NPU Communication Performance Benchmark
 * Comprehensive performance evaluation of NPUCOMM library
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <cstring>
#include "../npucomm.h"

using namespace npu::comm;

// Benchmark configuration
struct BenchmarkConfig {
    std::vector<size_t> message_sizes = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}; // 64B to 1MB
    std::vector<size_t> operation_counts = {100, 500, 1000, 5000};
    bool enable_warmup = true;
    size_t warmup_iterations = 50;
};

struct BenchmarkResult {
    size_t message_size;
    size_t num_operations;
    double avg_latency_us;
    double min_latency_us;
    double max_latency_us;
    double bandwidth_mbps;
    double operations_per_sec;
    double std_dev_us;
};

class NpuCommBenchmark {
public:
    NpuCommBenchmark(const BenchmarkConfig& config) : config_(config) {}

    std::vector<BenchmarkResult> run_put_benchmark() {
        return run_benchmark("PUT", [](void* dest, const void* src, size_t size, int pe) {
            return NpuComm::put(dest, src, size, pe);
        });
    }

    std::vector<BenchmarkResult> run_get_benchmark() {
        return run_benchmark("GET", [](void* dest, const void* src, size_t size, int pe) {
            return NpuComm::get(dest, src, size, pe);
        });
    }

    std::vector<BenchmarkResult> run_atomic_benchmark() {
        return run_benchmark("ATOMIC", [](void* dest, const void* src, size_t size, int pe) {
            int64_t* counter = static_cast<int64_t*>(dest);
            NpuComm::atomic_fetch_add(counter, 1, pe);
            return COMM_SUCCESS;
        });
    }

    void print_results(const std::vector<BenchmarkResult>& results, const std::string& title) {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << title << "\n";
        std::cout << std::string(80, '=') << "\n";

        std::cout << std::setw(10) << "Msg Size"
                  << std::setw(12) << "Operations"
                  << std::setw(12) << "Avg Lat(us)"
                  << std::setw(12) << "Min Lat(us)"
                  << std::setw(12) << "Max Lat(us)"
                  << std::setw(12) << "Bandwidth"
                  << std::setw(12) << "Ops/sec"
                  << "\n";

        std::cout << std::string(80, '-') << "\n";

        for (const auto& result : results) {
            std::cout << std::setw(10) << format_size(result.message_size)
                      << std::setw(12) << result.num_operations
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.avg_latency_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.min_latency_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.max_latency_us
                      << std::setw(12) << format_bandwidth(result.bandwidth_mbps)
                      << std::setw(12) << std::fixed << std::setprecision(0) << result.operations_per_sec
                      << "\n";
        }
    }

private:
    BenchmarkConfig config_;
    void* buffer_;

    using CommFunction = std::function<CommStatus(void*, const void*, size_t, int)>;

    std::vector<BenchmarkResult> run_benchmark(const std::string& op_name,
                                              CommFunction comm_func) {
        std::vector<BenchmarkResult> results;

        // Allocate benchmark buffer
        size_t max_size = *std::max_element(config_.message_sizes.begin(),
                                          config_.message_sizes.end());
        buffer_ = NpuComm::shmalloc(max_size);
        if (!buffer_) {
            std::cerr << "Failed to allocate benchmark buffer\n";
            return results;
        }

        // Initialize buffer with test data
        memset(buffer_, 0xAB, max_size);

        // Warmup phase
        if (config_.enable_warmup) {
            std::cout << "Performing warmup (" << config_.warmup_iterations << " iterations)...\n";
            for (size_t i = 0; i < config_.warmup_iterations; ++i) {
                comm_func(buffer_, buffer_, 1024, 1);
            }
            NpuComm::quiet(); // Wait for all operations to complete
        }

        // Run benchmarks for different message sizes and operation counts
        for (size_t msg_size : config_.message_sizes) {
            for (size_t num_ops : config_.operation_counts) {
                BenchmarkResult result = run_single_benchmark(msg_size, num_ops, comm_func);
                results.push_back(result);

                std::cout << op_name << " benchmark: " << format_size(msg_size)
                          << ", " << num_ops << " ops, "
                          << std::fixed << std::setprecision(2) << result.avg_latency_us << " us avg, "
                          << format_bandwidth(result.bandwidth_mbps) << "\n";
            }
        }

        // Cleanup
        NpuComm::shfree(buffer_);
        return results;
    }

    BenchmarkResult run_single_benchmark(size_t msg_size, size_t num_ops,
                                       CommFunction comm_func) {
        std::vector<double> latencies;

        for (size_t i = 0; i < num_ops; ++i) {
            // Vary target PE to simulate different communication patterns
            int target_pe = (i % (NpuComm::num_pes() - 1)) + 1;

            auto start = std::chrono::high_resolution_clock::now();
            CommStatus status = comm_func(buffer_, buffer_, msg_size, target_pe);
            auto end = std::chrono::high_resolution_clock::now();

            if (status == COMM_SUCCESS) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                latencies.push_back(duration.count());
            } else {
                std::cerr << "Communication failed in benchmark\n";
                latencies.push_back(0.0);
            }
        }

        // Wait for all operations to complete
        NpuComm::quiet();

        // Calculate statistics
        BenchmarkResult result;
        result.message_size = msg_size;
        result.num_operations = num_ops;

        if (!latencies.empty()) {
            result.min_latency_us = *std::min_element(latencies.begin(), latencies.end());
            result.max_latency_us = *std::max_element(latencies.begin(), latencies.end());
            result.avg_latency_us = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

            // Calculate standard deviation
            double sum_sq = 0.0;
            for (double lat : latencies) {
                double diff = lat - result.avg_latency_us;
                sum_sq += diff * diff;
            }
            result.std_dev_us = std::sqrt(sum_sq / latencies.size());

            // Calculate bandwidth (MB/s)
            double total_bytes = msg_size * num_ops;
            double total_time_sec = result.avg_latency_us * num_ops / 1e6;
            result.bandwidth_mbps = (total_bytes / (1024.0 * 1024.0)) / total_time_sec;

            // Calculate operations per second
            result.operations_per_sec = num_ops / (total_time_sec);
        }

        return result;
    }

    std::string format_size(size_t bytes) {
        if (bytes >= 1024 * 1024) {
            return std::to_string(bytes / (1024 * 1024)) + "MB";
        } else if (bytes >= 1024) {
            return std::to_string(bytes / 1024) + "KB";
        } else {
            return std::to_string(bytes) + "B";
        }
    }

    std::string format_bandwidth(double mbps) {
        if (mbps >= 1024) {
            return std::to_string(mbps / 1024) + "GB/s";
        } else {
            return std::to_string(mbps) + "MB/s";
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== NPU Communication Library - Performance Benchmark ===\n";

    // Initialize NPU communication library with performance-oriented config
    NpuCommConfig config;
    config.numPes = 4;
    config.myPe = 0;
    config.enableNpuOptimization = true;
    config.enableFineGrainComm = true;
    config.enablePgasMode = true;
    config.maxConcurrentOperations = 128;
    config.symmetricHeapSize = 1ULL * 1024 * 1024 * 1024; // 1GB

    CommStatus status = NpuComm::init(config);
    if (status != COMM_SUCCESS) {
        std::cerr << "Failed to initialize NPU communication library\n";
        return 1;
    }

    std::cout << "Performance benchmark initialized for PE " << NpuComm::my_pe()
              << " of " << NpuComm::num_pes() << " PEs\n";

    // Configure benchmark
    BenchmarkConfig bench_config;
    bench_config.message_sizes = {256, 1024, 4096, 16384, 65536}; // Focus on practical sizes
    bench_config.operation_counts = {100, 1000};

    NpuCommBenchmark benchmark(bench_config);

    // Run PUT benchmark
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RUNNING PUT BENCHMARKS\n";
    std::cout << std::string(80, '=') << "\n";

    auto put_results = benchmark.run_put_benchmark();
    benchmark.print_results(put_results, "PUT Operation Results");

    // Run GET benchmark
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RUNNING GET BENCHMARKS\n";
    std::cout << std::string(80, '=') << "\n";

    auto get_results = benchmark.run_get_benchmark();
    benchmark.print_results(get_results, "GET Operation Results");

    // Run atomic benchmark
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RUNNING ATOMIC BENCHMARKS\n";
    std::cout << std::string(80, '=') << "\n";

    auto atomic_results = benchmark.run_atomic_benchmark();
    benchmark.print_results(atomic_results, "Atomic Operation Results");

    // Summary
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "BENCHMARK SUMMARY\n";
    std::cout << std::string(80, '=') << "\n";

    if (!put_results.empty()) {
        auto best_put = *std::min_element(put_results.begin(), put_results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.avg_latency_us < b.avg_latency_us;
            });

        auto format_size = [](size_t bytes) -> std::string {
            if (bytes >= 1024 * 1024) {
                return std::to_string(bytes / (1024 * 1024)) + "MB";
            } else if (bytes >= 1024) {
                return std::to_string(bytes / 1024) + "KB";
            } else {
                return std::to_string(bytes) + "B";
            }
        };

        auto format_bandwidth = [](double mbps) -> std::string {
            if (mbps >= 1024) {
                return std::to_string(mbps / 1024) + "GB/s";
            } else {
                return std::to_string(mbps) + "MB/s";
            }
        };

        std::cout << "Best PUT performance: " << format_size(best_put.message_size)
                  << " messages, " << std::fixed << std::setprecision(2) << best_put.avg_latency_us
                  << " us avg latency, " << format_bandwidth(best_put.bandwidth_mbps) << "\n";
    }

    // Cleanup
    status = NpuComm::finalize();
    if (status == COMM_SUCCESS) {
        std::cout << "\nPerformance benchmark completed successfully!\n";
    }

    return 0;
}
