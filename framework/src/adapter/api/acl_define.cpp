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
 * \file acl_define.cpp
 * \brief
 */

#include "adapter/api/acl_define.h"
#ifdef BUILD_WITH_CANN
#include <type_traits>
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#endif

namespace npu::tile_fwk {
#ifdef BUILD_WITH_CANN
static_assert(std::is_same<AclError, aclError>::value);
static_assert(std::is_same<AclRtStream, aclrtStream>::value);
static_assert(std::is_same<AclRtEvent, aclrtEvent>::value);
static_assert(std::is_same<AclMdlRI, aclmdlRI>::value);
static_assert(std::is_same<AclRtExceptionInfoCallback, aclrtExceptionInfoCallback>::value);
static_assert(ACLRT_SUCCESS == ACL_SUCCESS);
static_assert(ACLRT_ERROR_REPEAT_INITIALIZE == ACL_ERROR_REPEAT_INITIALIZE);
static_assert(sizeof(AclRtStreamAttrValue) == sizeof(aclrtStreamAttrValue));
static_assert(sizeof(AclRtExceptionInfo) == sizeof(aclrtExceptionInfo));
#endif
}