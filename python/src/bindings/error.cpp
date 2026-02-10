/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings.h"
#include "core/error.h"

namespace py = pybind11;

namespace pypto {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindErrors(py::module_ &m) {
  static py::exception<ir::Error> exc_error(m, "Error", PyExc_Exception);
  static py::exception<ir::ValueError> exc_value_error(m, "ValueError", PyExc_ValueError);
  static py::exception<ir::TypeError> exc_type_error(m, "TypeError", PyExc_TypeError);
  static py::exception<ir::RuntimeError> exc_runtime_error(m, "RuntimeError", PyExc_RuntimeError);
  static py::exception<ir::NotImplementedError> exc_not_implemented_error(m, "NotImplementedError",
                                                                          PyExc_NotImplementedError);
  static py::exception<ir::IndexError> exc_index_error(m, "IndexError", PyExc_IndexError);
  static py::exception<ir::AssertionError> exc_assertion_error(m, "AssertionError", PyExc_AssertionError);
  static py::exception<ir::InternalError> exc_internal_error(m, "InternalError", PyExc_RuntimeError);

  PyObject *internal_error_type = exc_internal_error.ptr();
  PyObject_SetAttrString(internal_error_type, "__module__", PyUnicode_FromString("pypto"));

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const ir::ValueError &e) {
      PyErr_SetString(PyExc_ValueError, e.GetFullMessage().c_str());
    } catch (const ir::TypeError &e) {
      PyErr_SetString(PyExc_TypeError, e.GetFullMessage().c_str());
    } catch (const ir::RuntimeError &e) {
      PyErr_SetString(PyExc_RuntimeError, e.GetFullMessage().c_str());
    } catch (const ir::NotImplementedError &e) {
      PyErr_SetString(PyExc_NotImplementedError, e.GetFullMessage().c_str());
    } catch (const ir::IndexError &e) {
      PyErr_SetString(PyExc_IndexError, e.GetFullMessage().c_str());
    } catch (const ir::AssertionError &e) {
      PyErr_SetString(PyExc_AssertionError, e.GetFullMessage().c_str());
    } catch (const ir::InternalError &e) {
      PyErr_SetString(exc_internal_error.ptr(), e.GetFullMessage().c_str());
    } catch (const ir::Error &e) {
      PyErr_SetString(PyExc_Exception, e.GetFullMessage().c_str());
    }
  });
}

} // namespace pypto
