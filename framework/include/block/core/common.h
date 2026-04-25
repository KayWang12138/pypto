/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file common.h
 * @brief Backward-compatible forwarding header for block/core/common.h
 */

#ifndef PYPTO_CORE_COMMON_H_
#define PYPTO_CORE_COMMON_H_

#include "core/common.h"

// Keep the historical pybind11-facing macro name available for block bindings.
#ifndef PYPTO_MODULE_DOC
#define PYPTO_MODULE_DOC PYPTO_NANOBIND_MODULE_DOC
#endif

#endif  // PYPTO_CORE_COMMON_H_
