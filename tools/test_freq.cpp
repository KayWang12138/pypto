/**
 * 测试系统计数器频率
 * - ARM64: 直接读取 cntfrq_el0 寄存器
 * - x86:   使用 clock_gettime 测量实时频率
 * 
 * 编译: g++ -o test_freq test_freq.cpp
 * 运行: ./test_freq
 */

#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <time.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#define HAS_RDTSC 1
#else
#define HAS_RDTSC 0
#endif

inline uint64_t GetFreq()
{
    uint64_t freq;
#if defined(__aarch64__)
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
#elif defined(__x86_64__) || defined(_M_X64)
    freq = 0;  // x86 无固定频率寄存器，需实测
#else
    freq = 0;
#endif
    return freq;
}

inline uint64_t GetCycles()
{
    uint64_t cycles;
#if defined(__aarch64__)
    asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
#elif HAS_RDTSC
    cycles = __rdtsc();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    cycles = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
#endif
    return cycles;
}

inline uint64_t GetTimeNs()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "System Counter Frequency Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // 编译诊断
    std::cout << "\n[编译诊断]" << std::endl;
#if defined(__aarch64__)
    std::cout << "架构: ARM64 (aarch64) ✅" << std::endl;
    std::cout << "计数器类型: cntvct_el0 (系统计数器)" << std::endl;
#elif defined(__x86_64__) || defined(_M_X64)
    std::cout << "架构: x86_64 ✅" << std::endl;
    std::cout << "计数器类型: RDTSC (CPU TSC)" << std::endl;
#else
    std::cout << "架构: 其他/未知" << std::endl;
    std::cout << "计数器类型: clock_gettime (模拟)" << std::endl;
#endif

    uint64_t freq = GetFreq();
    
#if defined(__aarch64__)
    std::cout << "\n[寄存器读取结果]" << std::endl;
    std::cout << "cntfrq_el0 (Hz):       " << freq << std::endl;
    std::cout << "cntfrq_el0 (MHz):      " << freq / 1000000 << std::endl;
    std::cout << "cntfrq_el0 (cycles/ns): " << freq / 1000000000.0 << std::endl;
#elif defined(__x86_64__) || defined(_M_X64)
    std::cout << "\n[x86 TSC 测量]" << std::endl;
    std::cout << "注意: x86 无固定频率寄存器，需要实测 TSC 频率" << std::endl;
#else
    std::cout << "\n[模拟模式]" << std::endl;
#endif

    std::cout << "\n[实测验证] 等待 1 秒..." << std::endl;
    
    uint64_t ns1 = GetTimeNs();
    uint64_t cycles1 = GetCycles();
    sleep(1);
    uint64_t ns2 = GetTimeNs();
    uint64_t cycles2 = GetCycles();
    
    uint64_t ns_elapsed = ns2 - ns1;
    uint64_t cycles_elapsed = cycles2 - cycles1;
    double measured_freq = (double)cycles_elapsed / ((double)ns_elapsed / 1000000000.0);
    
    std::cout << "\n[实测结果]" << std::endl;
    std::cout << "经过时间 (ns):         " << ns_elapsed << std::endl;
    std::cout << "cycles 差值:           " << cycles_elapsed << std::endl;
    std::cout << "实测频率 (Hz):         " << (uint64_t)measured_freq << std::endl;
    std::cout << "实测频率 (MHz):        " << (uint64_t)measured_freq / 1000000 << std::endl;

#if defined(__aarch64__)
    double ratio = measured_freq / (double)freq;
    std::cout << "\n[一致性验证]" << std::endl;
    std::cout << "实测/寄存器比值:       " << ratio << std::endl;
    if (ratio > 0.99 && ratio < 1.01) {
        std::cout << "结果: ✅ 验证成功！" << std::endl;
    } else {
        std::cout << "结果: ⚠️ 存在偏差" << std::endl;
    }
#elif defined(__x86_64__) || defined(_M_X64)
    std::cout << "\n[x86 TSC 说明]" << std::endl;
    std::cout << "TSC 频率通常等于 CPU 标称频率或固定值" << std::endl;
    std::cout << "注意: 这是 Host CPU 的 TSC，不是 NPU 芯片的频率" << std::endl;
#endif

    std::cout << "\n========================================" << std::endl;
#if defined(__aarch64__)
    std::cout << "总结: ARM64 系统计数器频率 = " << freq / 1000000 << " MHz" << std::endl;
#elif defined(__x86_64__) || defined(_M_X64)
    std::cout << "总结: x86 TSC 频率 ≈ " << (uint64_t)measured_freq / 1000000 << " MHz" << std::endl;
    std::cout << "      (这是 Host CPU 频率，非 NPU 芯片频率)" << std::endl;
#endif
    std::cout << "========================================" << std::endl;
    
    return 0;
}