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
 * \file bindings.h
 * \brief
 */

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;
namespace pypto {
void BindEnum(nb::module_ &m);
void BindElement(nb::module_ &m);
void BindTensor(nb::module_ &m);
void BindSymbolicScalar(nb::module_ &m);
void BindController(nb::module_ &m);
void BindOperation(nb::module_ &m);
void BindRuntime(nb::module_ &m);
void BindCostModelRuntime(nb::module_ &m);
void BindPass(nb::module_ &m);
void BindFunction(nb::module_ &m);
void BindIr(nb::module_ &m);
} // namespace pypto
