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
 * \file rank_info.h
 * \brief Rank information manager for distributed precision tool
 */

#ifndef RANK_INFO_H
#define RANK_INFO_H

#include <string>
#include <cstdlib>

namespace npu::tile_fwk {

class RankInfo {
public:
    static RankInfo* GetInstance();

    void Initialize();

    int GetRankId() const { return rankId_; }
    int GetWorldSize() const { return worldSize_; }

    std::string GenerateShmemName(const std::string& groupName) const;

private:
    RankInfo() = default;
    ~RankInfo() = default;

    int rankId_ = -1;
    int worldSize_ = 1;
    std::string jobId_;
};

} // namespace npu::tile_fwk

#endif // RANK_INFO_H
