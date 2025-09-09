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
 * \file bindings.h
 * \brief
 */

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace pypto {
    void bind_enum(py::module &m);
    void bind_tensor(py::module &m);
    void bind_symbolic_scalar(py::module &m);
    void bind_controller(py::module &m);
    void bind_operation(py::module &m);
}
