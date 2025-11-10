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
 * \file set_boundary.h
 * \brief
 */

#ifndef PASS_SET_BOUNDARY_H
#define PASS_SET_BOUNDARY_H
#include "pre_graph_common.h"

namespace npu::tile_fwk {
class SetBoundary {
public:
    SetBoundary() {}
    ~SetBoundary() = default;

    void InsertTemporaryCopyIn(Function &function, Operation &op) const;
    void SetTensorBoundary(Function &function) const;
};
} // namespace npu::tile_fwk
#endif // PASS_SET_BOUNDARY_H
    
