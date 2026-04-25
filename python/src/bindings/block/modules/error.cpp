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
 * @file error.cpp
 * @brief Implementation of Python bindings for PyPTO error classes
 */

#include "core/error.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../module.h"

namespace py = pybind11;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindErrors(py::module_& m) {
  // Register custom exception types and map them to Python exceptions
  // These static objects ensure exceptions persist for the lifetime of the module
  static py::exception<Error> exc_error(m, "Error", PyExc_Exception);
  static py::exception<ValueError> exc_value_error(m, "ValueError", PyExc_ValueError);
  static py::exception<TypeError> exc_type_error(m, "TypeError", PyExc_TypeError);
  static py::exception<RuntimeError> exc_runtime_error(m, "RuntimeError", PyExc_RuntimeError);
  static py::exception<NotImplementedError> exc_not_implemented_error(m, "NotImplementedError",
                                                                              PyExc_NotImplementedError);
  static py::exception<IndexError> exc_index_error(m, "IndexError", PyExc_IndexError);
  static py::exception<AssertionError> exc_assertion_error(m, "AssertionError", PyExc_AssertionError);
  static py::exception<InternalError> exc_internal_error(m, "InternalError", PyExc_RuntimeError);

  // Set __module__ to "pypto" so the exception displays as "pypto.InternalError" instead of
  // "pypto.pypto_core.InternalError"
  PyObject* internal_error_type = exc_internal_error.ptr();
  PyObject_SetAttrString(internal_error_type, "__module__", PyUnicode_FromString("pypto"));

  // Register exception translator to convert C++ exceptions to Python exceptions
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const ValueError& e) {
      // Catch most specific exceptions first
      PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const TypeError& e) {
      PyErr_SetString(PyExc_TypeError, e.what());
    } catch (const RuntimeError& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (const NotImplementedError& e) {
      PyErr_SetString(PyExc_NotImplementedError, e.what());
    } catch (const IndexError& e) {
      PyErr_SetString(PyExc_IndexError, e.what());
    } catch (const AssertionError& e) {
      PyErr_SetString(PyExc_AssertionError, e.what());
    } catch (const InternalError& e) {
      PyErr_SetString(exc_internal_error.ptr(), e.what());
    } catch (const Error& e) {
      // Catch base Error last as a fallback
      PyErr_SetString(PyExc_Exception, e.what());
    }
  });
}

}  // namespace python
}  // namespace pypto
