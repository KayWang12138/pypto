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
 * \file hal_define.cpp
 * \brief
 */

#include "adapter/api/hal_define.h"
#ifdef BUILD_WITH_CANN
#include "driver/ascend_hal_error.h"
#include "driver/ascend_hal_define.h"
#endif

namespace npu::tile_fwk {
#ifdef BUILD_WITH_CANN
static_assert(static_cast<int32_t>(HAL_ERROR_NONE) == static_cast<int32_t>(DRV_ERROR_NONE));
static_assert(sizeof(ResMapInfo) == sizeof(res_map_info));
#endif
}
