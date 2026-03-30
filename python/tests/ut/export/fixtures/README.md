# Fixtures for export C++ codegen tests

`gert_ge_minimal.hpp` provides minimal `gert::Shape`, `gert::InferShapeContext`, and `ge::graphStatus` / `GRAPH_*` symbols matching usage in generated `InferShapeGeImpl` code.

`sinkable_executor_minimal.hpp` provides minimal `SinkableExecuteOp`, `MockSinkableOpExecutionContext`, `SinkableOpIo`, and `REG_AUTO_MAPPING_OP` for compiling generated sinkable executors (class name comes from ``op_type`` / ``_custom_executor_class_cpp_for_test(op_type=...)`` in `pypto.export.cpp.codegen`).

## Optional compile tests (`@pytest.mark.cpp_codegen`)

`test_cpp_codegen_compile.py` uses one `_compile_and_run(..., options=CppCompileOptions(...))`: `embed_python=True` for infer-shape, default for custom-executor.

**Infer-shape TU** additionally requires:

- **pybind11** headers (`pip install pybind11`, or the build env that provides `pybind11.get_include()`).
- **Python development headers and lib** so the test TU can embed the interpreter (`python3-config --cflags --ldflags --embed` on Python 3.8+; on some distros install `python3-dev`).

**Custom-executor TU** only needs a C++17 compiler (no pybind/Python link).

For all compile tests:

- A C++17 compiler (`g++` or `clang++`, or `CXX` in the environment).

If prerequisites for a given test are missing, that test is skipped automatically.
