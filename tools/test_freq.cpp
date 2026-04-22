/**
 * 测试 ARM64 系统计数器频率 (cntfrq_el0)
 * 用于验证 GetFreq() 函数的准确性
 * 
 * 编译: g++ -o test_freq test_freq.cpp
 * 运行: ./test_freq
 */

#include <cstdint>
#include <iostream>
#include <unistd.h>

inline uint64_t GetFreq()
{
    uint64_t freq;
#ifdef __aarch64__
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
#else
    freq = 1000000000;  // 非 ARM64 平台返回 1 GHz（模拟）
#endif
    return freq;
}

inline uint64_t GetCycles()
{
    uint64_t cycles;
#ifdef __aarch64__
    asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
#else
    cycles = 0;  // 非 ARM64 平台无法读取
#endif
    return cycles;
}

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "ARM64 System Counter Frequency Test" << std::endl;
    std::cout << "========================================" << std::endl;

#ifdef __aarch64__
    std::cout << "Platform: aarch64 (ARM64)" << std::endl;
#else
    std::cout << "Platform: non-ARM64 (模拟环境)" << std::endl;
#endif

    uint64_t freq = GetFreq();
    uint64_t cycles1 = GetCycles();
    
    std::cout << "\n[寄存器读取结果]" << std::endl;
    std::cout << "cntfrq_el0 (频率):     " << freq << " Hz" << std::endl;
    std::cout << "cntfrq_el0 (MHz):      " << freq / 1000000 << " MHz" << std::endl;
    std::cout << "cntfrq_el0 (cycles/ns): " << freq / 1000000000.0 << std::endl;
    
#ifdef __aarch64__
    std::cout << "\n[实测验证] 等待 1 秒后再次读取 cycles..." << std::endl;
    sleep(1);
    
    uint64_t cycles2 = GetCycles();
    uint64_t delta = cycles2 - cycles1;
    
    std::cout << "cycles 差值:           " << delta << std::endl;
    std::cout << "实测频率 (Hz):         " << delta << " Hz" << std::endl;
    std::cout << "实测频率 (MHz):        " << delta / 1000000 << " MHz" << std::endl;
    
    // 验证一致性
    double ratio = (double)delta / (double)freq;
    std::cout << "\n[一致性验证]" << std::endl;
    std::cout << "实测频率 / 声称频率 比值: " << ratio << std::endl;
    
    if (ratio > 0.99 && ratio < 1.01) {
        std::cout << "结果: ✅ 验证成功，频率测量一致！" << std::endl;
    } else if (ratio > 0.9 && ratio < 1.1) {
        std::cout << "结果: ⚠️ 基本一致，可能有轻微偏差" << std::endl;
    } else {
        std::cout << "结果: ❌ 验证失败，频率不一致" << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "总结: 当前芯片系统计数器频率 = " << freq / 1000000 << " MHz" << std::endl;
    std::cout << "========================================" << std::endl;
#else
    std::cout << "\n非 ARM64 平台，无法进行实测验证" << std::endl;
#endif
    
    return 0;
}