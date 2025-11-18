/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pass_dependency.h
 * \brief
 */

#pragma once

#include "interface/utils/common.h"
#include "passes/tile_graph_pass/graph_optimization/split_reshape.h"
#include "passes/pass_interface/pass_type.h"

namespace npu::tile_fwk {
class PassDependency {
public:
    static PassDependency &Instance();

    Status CheckStrategyDependency(const std::string &strategyName, const std::vector<std::string> &passes);

private:
    PassDependency();
    ~PassDependency() = default;

    PassDependency(const PassDependency&) = delete;
    PassDependency& operator=(const PassDependency&) = delete;

private:
    std::unordered_map<std::string, std::vector<std::string>> passDependencies_;
};
} // namespace npu::tile_fwk