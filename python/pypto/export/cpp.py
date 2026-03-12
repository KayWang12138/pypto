import inspect

from .helpers import _unwrap_decorated_func_source, _unwrap_decorated_func_name

# This module is WIP and subject to changes (pending requirement clarifications)

def _generate_pybind_wrapper(func, cpp_func_name: str) -> str:
    def _to_cpp_type(py_ann):
        if py_ann == int:
            return "int" # use int32_t instead? (need explicit conversion)
        elif py_ann == float:
            return "float"
        elif py_ann == bool:
            return "bool"
        elif py_ann.__origin__ == tuple:
            return f"std::tuple<{', '.join([_to_cpp_type(arg) for arg in py_ann.__args__])}>"
        elif py_ann.__origin__ == list:
            return f"std::vector<{_to_cpp_type(py_ann.__args__[0])}>"

    def _to_cpp_arg(py_arg):
        return f"{_to_cpp_type(py_arg.annotation)} {py_arg.name}"

    py_source = _unwrap_decorated_func_source(inspect.getsource(func))
    py_func_name = _unwrap_decorated_func_name(func.__name__)
    py_sig = inspect.signature(func)

    cpp_args_list = ", ".join([_to_cpp_arg(py_arg) for py_arg in py_sig.parameters.values()])
    cpp_return_type = _to_cpp_type(py_sig.return_annotation)

    pybind_args_list = ", ".join([f"py::cast({py_arg.name})" for py_arg in py_sig.parameters.values()])

    return f"""// Auto-generated

#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

{cpp_return_type} {cpp_func_name}({cpp_args_list}) {{
    py::scoped_interpreter guard{{}};

    const std::string py_source = R"(
{py_source}
)";

    py::module m = py::module_::import("__main__");
    py::dict globals = m.attr("__dict__");
    py::exec(py_source, globals, globals);
    py::object {py_func_name}_py = globals["{py_func_name}"];

    return {py_func_name}_py({pybind_args_list}).cast<{cpp_return_type}>();
}}
"""

def _generate_op_kernel_info():
    return f"""//Auto-generated

#include <iostream>
#include <tuple>

std::tuple<int, int> op_kernel_info(int inputs_num, int outputs_num) {{
    std::cout << "Entering op_kernel_info..." << std::endl;
    return std::make_tuple(inputs_num, outputs_num);
}}
"""

def _generate_op_compile():
    return f"""// Auto-generated

#include <iostream>

int op_compile(int flags) {{
    std::cout << "Entering op_compile..." << std::endl;
    int status = 0;
    return status;
}}
"""

def _generate_op_execute():
    return f"""// Auto-generated

#include <iostream>

int op_execute(int flags) {{
    std::cout << "Entering op_execute..." << std::endl;
    int status = 0;
    return status;
}}
"""
