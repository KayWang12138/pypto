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
 * \file msprof_api.cpp
 * \brief
 */

#include "adapter/api/msprof_api.h"

#ifdef BUILD_WITH_CANN
#include "adapter/manager/adapter_manager.h"
#endif
#include "adapter/stubs/msprof_stubs.h"

namespace npu::tile_fwk {
uint64_t ProfSysCycleTime(void) {

}
uint64_t ProfGetHashId(const char *hashInfo, size_t length) {

}
int32_t ProfReportApi(uint32_t nonPersistantFlag, const struct MsprofApi *api);
int32_t ProfReportCompactInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length);
int32_t ProfReportAdditionalInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length);
int32_t ProfRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);
}