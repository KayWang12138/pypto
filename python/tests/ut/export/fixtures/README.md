# Fixtures for export C++ codegen tests

`gert_ge_minimal.hpp` provides `ge::DataType` / `ge::DT_*`, `ge::graphStatus` / `GRAPH_*`, `gert::Shape`, `gert::StorageShape`, `gert::Tensor`, and `gert::InferShapeContext` matching usage in generated export test code.

`sinkable_executor_minimal.hpp` includes `gert_ge_minimal.hpp` and provides minimal `SinkableExecuteOp`, `MockSinkableOpExecutionContext`, `SinkableOpIo`, and `REG_AUTO_MAPPING_OP` for compiling executor fragments from `_generate_custom_executor_cpp(..., for_compile_test=True)` together with standard / pybind headers only. `gert::Tensor`, `gert::StorageShape`, and `gert::Shape` live in `gert_ge_minimal.hpp` (`Tensor` matches sinkable codegen: `GetAddr()`, `GetDataType()`, `GetShape()`).

## Optional compile tests (`@pytest.mark.cpp_codegen`)

`test_cpp_codegen_compile.py` uses one `_compile_and_run(..., options=CppCompileOptions(...))`: `embed_python=True` for infer-shape, default for custom-executor.

**Infer-shape TU** additionally requires:

- **pybind11** headers (`pip install pybind11`, or the build env that provides `pybind11.get_include()`).
- **Python development headers and lib** so the test TU can embed the interpreter (`python3-config --cflags --ldflags --embed` on Python 3.8+; on some distros install `python3-dev`).

**Custom-executor TU** uses embedded Python for `calc_workspace` pybind glue: same pybind11 + Python dev requirements as infer-shape, plus `embed_python=True` in the compile helper.

For all compile tests:

- A C++17 compiler (`g++` or `clang++`, or `CXX` in the environment).

If prerequisites for a given test are missing, that test is skipped automatically.
