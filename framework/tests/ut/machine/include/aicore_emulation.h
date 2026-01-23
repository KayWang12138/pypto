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
 * \file aicore_emulation.h
 * \brief
 */

#include <chrono>

namespace npu::tile_fwk::machine {

class AicoreEmulationBase {
public:
    uint64_t GetSysCnt() {
        auto now = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        return ns;
    }

    virtual int64_t GetCoreId() {
        return 0;
    }

    virtual int GetBlockIdx() {
        return 0;
    }

    virtual void SetCond(uint64_t cond) {
        UNUSED(cond);
    }

    virtual uint64_t GetDataMainBase() {
        return 0;
    }
};

class AicoreEmulationManager {
public:
    static AicoreEmulationManager &GetInstance();

    void RegisterAicoreEmulation(std::shared_ptr<AicoreEmulationBase> base) { base_ = base; }

    std::shared_ptr<AicoreEmulationBase> GetEmulation() { return base_; }
private:
    std::shared_ptr<AicoreEmulationBase> base_;
};

}
