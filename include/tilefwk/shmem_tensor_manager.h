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
 * \file shmem_tensor_manager.h
 * \brief
 */

#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include "data_type.h"
#include "tensor.h"

namespace npu::tile_fwk {
namespace Distributed {

class ShmemTensorMgr {
public:
    static ShmemTensorMgr &GetInstance()
    {
        static ShmemTensorMgr instance;
        return instance;
    }

    Tensor CreateTensor(int32_t rankSize, int32_t groupIndex, DataType t, const Shape &shape,
        const std::string& tag = "");
    Tensor GetView(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<int64_t> &offsets);
    Tensor GetView(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<SymbolicScalar> &offsets);
    void Reset() { offset_ = 0; }
private:
    ShmemTensorMgr() = default;
    ~ShmemTensorMgr() = default;
    size_t offset_{0};
};

} // namespace Distributed
} // namespace npu::tile_fwk
