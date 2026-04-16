/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file boundary_utils.h
 * \brief 边界 tensor 集中管理工具类
 */

#pragma once

#include <unordered_set>
#include <mutex>

namespace npu::tile_fwk {

class BoundaryUtils {
public:
    static BoundaryUtils& GetInstance()
    {
        static BoundaryUtils instance;
        return instance;
    }

    void AddBoundary(int tensorMagic);
    void RemoveBoundary(int tensorMagic);
    bool IsBoundary(int tensorMagic) const;
    void Clear();
    const std::unordered_set<int>& GetBoundaries() const;

private:
    BoundaryUtils() = default;
    ~BoundaryUtils() = default;
    BoundaryUtils(const BoundaryUtils&) = delete;
    BoundaryUtils& operator=(const BoundaryUtils&) = delete;

    std::unordered_set<int> boundaryTensors_;
    mutable std::mutex mutex_;
};

} // namespace npu::tile_fwk