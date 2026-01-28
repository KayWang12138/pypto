/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file distributed_context.h
 * \brief
 */

#pragma once
#include <vector>
#include <string>

namespace npu::tile_fwk::dynamic {
enum class ResType {
    MESH,
    RING,
    UNKNOWN
};

class DistributedContext {
public:
    DistributedContext(){};
    ~DistributedContext(){};
    static std::vector<uint64_t> GetCommContext(const std::vector<std::string> &groupNames);
    static std::vector<uint64_t> GetCommContextToHost(const std::vector<std::string> &groupNames);
    template<ResType T>
    static uint64_t AllocCommContext(uint64_t ctxAddr);
};
} // namespace npu::tile_fwk::dynamic