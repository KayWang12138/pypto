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
 * \file pybind11.cpp
 * \brief
 */

#include "pybind_common.h"
#include "bindings/bindings.h"

using namespace npu::tile_fwk;

namespace pypto {
PYBIND11_MODULE(pto_impl, m) {
    m.doc() = "PyPTO";
    bind_enum(m);
    BindElement(m);
    BindTensor(m);
    BindSymbolicScalar(m);
    bind_controller(m);
    bind_operation(m);
    BindRuntime(m);
    bind_pass(m);
};
} // namespace pypto
