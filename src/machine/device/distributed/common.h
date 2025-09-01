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
struct TensorInfo {
    uint64_t rawAddr;
    uint32_t dim;
    std::vector<uint32_t> offset;
    std::vector<uint32_t> shape;
    std::vector<uint32_t> rawShape;
    std::vector<uint32_t> dynValidShape;
};
}
