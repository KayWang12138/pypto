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
 * \file op_info_manager.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {
constexpr uint64_t MAIN_KEY_MASK = 0xFFFFFFFFFFFFF;
constexpr uint64_t SUB_KEY_OFFSET = 52;
constexpr uint64_t SUB_KEY_MASK = 0xFFF0000000000000;

class OpInfoManager {
public:
    OpInfoManager() = default;
    ~OpInfoManager() = default;
    static OpInfoManager &GetInstance();
    void SetOpTilingKey(uint64_t opTilingKey);
    uint64_t GetOpTilingKey() const;
    uint64_t GetNewSubTilingKey();
    uint64_t GetCurSubTilingKey() const;
    void SetOpType(const std::string &opType);
    const std::string &GetOpType() const;
private:
  std::string opType_ = "tilefwk";
  uint64_t opTilingKey_{0};
  uint64_t subTilingKey_{0};
};
}
