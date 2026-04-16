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
 * \file alignment_utils.h
 * \brief
 */

#pragma once

#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

class AlignmentUtils {
public:
    /**
     * @brief Get last-dimension alignment base in elements.
     *
     * @param tensor input logical tensor.
     * @return alignment base in elements; return 0 when tensor/dtype is invalid.
     */
    static int64_t GetLastDimAlignBase(const LogicalTensorPtr& tensor);

    /**
     * @brief Check whether the last dimension is 32-byte aligned.
     *
     * @param tensor input logical tensor.
     * @return true if last dim bytes is aligned to BLOCK_SIZE; otherwise false.
     */
    static bool IsLastDim32BAligned(const LogicalTensorPtr& tensor);

    static int64_t Pad(int64_t dim, int64_t padValue);

    static void ProcessLastDim32BAlignedOnUB(LogicalTensorPtr tensor);

    /**
     * @brief Check whether UB tensor last dimension needs padding.
     *
     * @param tensor input logical tensor.
     * @return true if UB tensor last dim is not aligned; otherwise false.
     */
    static bool NeedPadLastDim(const LogicalTensorPtr& tensor);

    /**
     * @brief Check whether tensor rawshape and oriRawshape differ on last dimension.
     *
     * @param tensor input logical tensor.
     * @return true if raw last dim is unaligned; otherwise false.
     */
    static bool IsRawLastDimUnaligned(const LogicalTensorPtr& tensor);

    /**
     * @brief Check whether any input/output of an op is unaligned.
     *
     * @param op input operation.
     * @return true if any non-combined-axis input/output is unaligned; otherwise false.
     */
    static bool HasUnalignedInputOrOutput(const Operation& op);

private:
    /**
     * @brief Check tensor basic validity for last-dimension alignment calculation.
     *
     * @param tensor input logical tensor.
     * @return true if tensor and shape are valid for last-dim check.
     */
    static bool IsValidForLastDimCheck(const LogicalTensorPtr& tensor);

    /**
     * @brief Check whether current operand index is marked as combined axis.
     *
     * @param combineAxis combine-axis attribute vector.
     * @param index operand index.
     * @return true if current index is marked combined and should be skipped.
     */
    static bool IsCombinedAxis(const std::vector<bool>& combineAxis, size_t index);
};

} // namespace npu::tile_fwk
