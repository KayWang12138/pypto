#ifndef MACHINE_UTILS_CHECKCINJECT_H
#define MACHINE_UTILS_CHECKCINJECT_H

#include <cstddef>

namespace npu::tile_fwk {
inline int Checkinject(const char cmdStr[], size_t strLen)
{
    if (cmdStr == nullptr || strLen == 0) {
        return -1;
    }
    const char cmdIllegalChar[] = {';', '|', '<', '>', '`'};
    for (size_t i = 0; i < strLen; i++) {
        for (const auto& c : cmdIllegalChar) {
            if (cmdStr[i] == c) {
                return -1;
            }
        }
    }
    return 0;
}
} // namespace npu::tile_fwk

#endif
