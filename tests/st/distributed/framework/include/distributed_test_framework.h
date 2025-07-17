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
 * \file distributed_test_framework.h
 * \brief
 */

#pragma once

#include "distributed_op_test_suite.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

namespace npu::tile_fwk {
namespace Distributed {

struct HcomTestParam {
    HcclComm hcclComm;
    int32_t rootRank;
    HcclRootInfo rootInfo;
};

void TestFrameworkInit(OpTestParam &testParam, HcomTestParam &hcomTestParam);
void TestFrameworkDestroy(int32_t timeout);

} // namespace Distributed
} // namespace npu::tile_fwk
