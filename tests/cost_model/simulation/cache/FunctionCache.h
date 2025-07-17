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
 * \file FunctionCache.h
 * \brief
 */

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <climits>

#include "simulation/common/ISA.h"
#include "simulation/statistics/TraceLogger.h"

namespace CostModel {
class SimSys;
class FunctionCache {
public:
    std::unordered_map<uint64_t, FunctionPtr> cache;  // key is function hash
    std::unordered_map<std::string, uint64_t> funcNameToKey;
    void Insert(FunctionPtr func);
    void CountFunctionCache(uint64_t key, CostModel::Pid pid, CostModel::Tid tid, bool hit);
    bool Lookup(uint64_t key, CostModel::Pid pid = LLONG_MAX, CostModel::Tid tid = LLONG_MAX);
    bool LookupCache(uint64_t key);
    // Get Function Cache line.
    FunctionPtr GetFunction(uint64_t key);
    std::shared_ptr<SimSys> GetSim();
    void SetSim(std::shared_ptr<CostModel::SimSys> simPtr);
    void SetMaxCacheSize(uint64_t cacheSize);
    uint64_t GetMaxCacheSize() const;

private:
    std::shared_ptr<CostModel::SimSys> sim = nullptr;
    std::unordered_map<uint64_t, uint64_t> funcLastUseTime;
    std::set<std::pair<uint64_t, uint64_t>> inCacheFunction;
    uint64_t cacheTblTime = 0;
    uint64_t totalCnt = 0;
    uint64_t hitCnt = 0;
    uint64_t missCnt = 0;
    uint64_t maxCacheSize = -1;
};
}  // namespace CostModel