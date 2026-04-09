/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_QUANT__H
#define TILEOP_TILE_OPERATOR_QUANT__H

#include <pto/npu/a5/TQuant.hpp>

#include "pto_tile.h"

#define OP_TILE_OP_QUANT_MX TQuantMX
template <typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuantMX(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    auto dstTile = PtoTile<T0>(dst);
    auto expTile = PtoTile<T1>(exp);
    auto maxTile = PtoTile<T2>(maxScratch);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    auto srcTile = PtoTile<T4>(src);
    pto::TQUANT_IMPL<pto::QuantType::MXFP8>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
}

#endif
