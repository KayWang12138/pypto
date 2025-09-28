/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>

namespace npu::tile_fwk::Distributed {
constexpr uint64_t VECTOR_PRE_SIZE = 1024;

struct TensorInfo {
    uint64_t rawAddr;
    uint32_t dim;
    uint64_t rawIndex;
    std::vector<uint32_t> offset;
    std::vector<uint32_t> shape;
    std::vector<uint32_t> rawShape;
    std::vector<uint32_t> dynValidShape;
};

inline uint64_t GetVirtualAddrBist(uint64_t val, uint64_t start, uint64_t end)
{
    return (((val) >> (start)) & ((1UL << ((end) - (start) + 1UL)) - 1UL));
}

inline uint64_t GetVirtaulAddrOffset(uint64_t val)
{
    constexpr uint64_t offsetStart = 0UL; 
    constexpr uint64_t offsetEnd = 57UL; 
    return GetVirtualAddrBist(val, offsetStart, offsetEnd);
}

inline uint64_t GetVirtaulAddrGroupIndex(uint64_t val)
{
    constexpr uint64_t groupIndexStart = 58UL; 
    constexpr uint64_t groupIndexEnd = 59UL; 
    return GetVirtualAddrBist(val, groupIndexStart, groupIndexEnd);
}

inline uint64_t GetVirtaulAddrMemType(uint64_t val)
{
    constexpr uint64_t memTypeStart = 60UL; 
    constexpr uint64_t memTypeEnd = 61UL; 
    return GetVirtualAddrBist(val, memTypeStart, memTypeEnd);
}
}
