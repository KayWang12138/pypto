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
 * \file boundary_utils.cpp
 * \brief 边界 tensor 集中管理工具类实现
 */

#include "boundary_utils.h"

namespace npu::tile_fwk {

void BoundaryUtils::AddBoundary(int tensorMagic)
{
    std::lock_guard<std::mutex> lock(mutex_);
    boundaryTensors_.insert(tensorMagic);
}

void BoundaryUtils::RemoveBoundary(int tensorMagic)
{
    std::lock_guard<std::mutex> lock(mutex_);
    boundaryTensors_.erase(tensorMagic);
}

bool BoundaryUtils::IsBoundary(int tensorMagic) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return boundaryTensors_.count(tensorMagic) > 0;
}

void BoundaryUtils::Clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    boundaryTensors_.clear();
}

const std::unordered_set<int>& BoundaryUtils::GetBoundaries() const { return boundaryTensors_; }

} // namespace npu::tile_fwk