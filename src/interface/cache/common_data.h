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
 * \file common_data.h
 * \brief
 */

#pragma once

#ifndef COMMON_DATA_H
#define COMMON_DATA_H

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

namespace npu::tile_fwk {

constexpr int32_t DEV_SHAPE_DIM_MAX = 5;

struct DevAscendShape {
    int dimSize{0};
    int dim[DEV_SHAPE_DIM_MAX];
};

struct DevAscendTensorData {
    uint64_t address{0};
    DevAscendShape shape;
};

struct DevStartArgsBase {
    __gm__ DevAscendTensorData *inputTensorList;
    uint64_t inputTensorSize;
    __gm__ DevAscendTensorData *outputTensorList;
    uint64_t outputTensorSize;
};

}

#endif