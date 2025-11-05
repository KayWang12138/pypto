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
 * \file L2CacheImplA2A3.cpp
 * \brief
 */

#include <unordered_map>
#include "L2CacheImplA2A3.h"

namespace CostModel
{
    uint64_t L2CacheImplA2A3::Simulate(const CostModel::CachePacket &packet) {
        if (packet.size == 0) {
            return 0;
        }
        return 0;
    }
} // namespace CostModel
