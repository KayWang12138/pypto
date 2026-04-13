/*!
 * \file aicore_manager.h
 * \brief
 */

#pragma once
#include <cstdint>
#include <sys/ioctl.h>
#include <functional>
#include <vector>
#include <atomic>
#include <array>
#include <semaphore.h>

struct AicpuProfilingPoint {
    uint64_t execStart;
    uint64_t execEnd;
    uint32_t taskId;
    std::string label;

    bool isAllZero() const {
        return execStart == 0 && execEnd == 0 && taskId == 0;
    }
};

class Utils
{
public:
constexpr static size_t PROFILING_LEN = 173UL;

public:
    void AsmCntvc(uint64_t &cntvct) const
{
    asm volatile("mrs %0, cntvct_el0" : "=r"(cntvct));
}
};

