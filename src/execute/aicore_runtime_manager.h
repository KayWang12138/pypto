/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#pragma once

#include <vector>
#include <unordered_map>
#include "graph/op_desc.h"
#include "machine/utils/common_def.h"

namespace npu::tile_fwk {
struct DevAscendProgram {
    DeviceArgs devArgs;
    uint64_t workspaceSize;
    uint64_t l2CacheOffset;
    uint64_t configKey;
};

class AicoreRtManager {
public:
    AicoreRtManager();
    ~AicoreRtManager();
    AicoreRtManager(AicoreRtManager &other) = delete;
    void operator=(const AicoreRtManager &other) = delete;

    static AicoreRtManager &Instance() {
        static AicoreRtManager Inst;
        return Inst;
    }
    bool AllocDevAddr(uint8_t **dev_addr, size_t size);
    void InsertHiddenInput(const int64_t &op_id, void *hidden_input);
    void* GetHiddenInput(const int64_t &op_id);
    bool GetAicoreRegInfo(const ge::OpDescPtr &op_desc, std::vector<int64_t> &aic, std::vector<int64_t> &aiv,
                          int32_t deviceId);
    bool InitDyBinData(const ge::OpDescPtr &op_desc, std::vector<int64_t> &aic, std::vector<int64_t> &aiv,
                       DevAscendProgram *host_args, int32_t deviceId);
    ge::graphStatus TileFwkHiddenInput(const ge::OpDescPtr &op_desc, std::vector<void *> &contexts);

private:
    std::vector<uint8_t *> allocated_addrs_;
    std::unordered_map<int64_t, void *> op_to_hiddeninput_;
};
} // namespace fe
