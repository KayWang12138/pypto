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
 * \file comm_group_recorder.h
 * \brief
 */

#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <sstream>
#include "tilefwk/error.h"
#include "tilefwk/core_func_data.h"

namespace npu::tile_fwk {
namespace Distributed {
class CommGroupRecorder {
public:
    CommGroupRecorder() = default;
    ~CommGroupRecorder() {};

    // 注册组，返回对应的 groupIndex（自动去重）
    inline uint32_t Input(const std::string &groupName)
    {
        auto it = name2Index_.find(groupName);
        if (it != name2Index_.end()) {
            return it->second; // 已存在，返回现有 index
        }

        // 新组：记录映射关系
        uint32_t newIndex = index2Name_.size();
        ASSERT(newIndex < DIST_COMM_GROUP_NUM);

        index2Name_.push_back(groupName);
        name2Index_[groupName] = newIndex;
        return newIndex;
    }

    // 获取所有 groupName 的列表（按 index 顺序）
    inline const std::vector<std::string> &Output() const { return index2Name_; }

    static inline std::string PrintString(std::vector<std::string> &commGroups)
    {
        std::ostringstream oss;
        oss << "distributed comm groups: [";
        for (size_t i = 0; i < commGroups.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << commGroups[i];
        }
        oss << "]";
        return oss.str();
    }

private:
    std::unordered_map<std::string, uint32_t> name2Index_;
    std::vector<std::string> index2Name_;
};
} // namespace Distributed
} // namespace npu::tile_fwk
