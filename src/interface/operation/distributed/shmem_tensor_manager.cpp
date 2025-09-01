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
 * \file shmem_tensor_manager.cpp
 * \brief
 */

#include "tilefwk/shmem_tensor_manager.h"
#include "interface/tensor/logical_tensor.h"
#include "operation/tilefwk_op.h"

namespace npu::tile_fwk {
namespace Distributed {

Tensor ShmemTensorMgr::CreateTensor(int32_t rankSize, int32_t groupIndex, DataType t, const Shape &shape,
    const std::string& tag)
{
    Shape shmemShape{rankSize, rankSize};
    shmemShape.insert(shmemShape.end(), shape.begin(), shape.end());
    Tensor shmemTensor(t, shmemShape, tag);
    shmemTensor.GetStorage()->tensor->SetShmemInfo(RawTensor::ShmemInfo{true, groupIndex, offset_});
    offset_ += shape[0] * shape[1] * BytesOf(t) * rankSize;
    return shmemTensor;
}

Tensor ShmemTensorMgr::GetView(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<int64_t> &offsets)
{
    auto v = View(operand, shapes, offsets);
    v.GetStorage()->tensor->SetShmemInfo(operand.GetStorage()->tensor->GetShmemInfo());
    return v;
}

Tensor ShmemTensorMgr::GetView(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<SymbolicScalar> &offsets)
{
    auto v = View(operand, shapes, offsets);
    v.GetStorage()->tensor->SetShmemInfo(operand.GetStorage()->tensor->GetShmemInfo());
    return v;
}

} // namespace Distributed
} // namespace npu::tile_fwk
