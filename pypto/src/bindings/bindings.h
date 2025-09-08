#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace pypto {
    void bind_enum(py::module_ &m);
    void bind_tensor(py::module_ &m);
    void bind_symbolic_scalar(py::module_ &m);
    void bind_controller(py::module_ &m);
    void bind_operation(py::module_ &m);
}