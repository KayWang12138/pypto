# Build pypto_impl Python extension module.
find_package(Python3 REQUIRED
    COMPONENTS Interpreter Development
)
find_package(pybind11 REQUIRED)

# Collect all binding source files
set(BINDING_SOURCES
    pypto_impl.cpp
    type_bindings.cpp
    tensor_bindings.cpp
    function_bindings.cpp
    statement_bindings.cpp
    operation_bindings.cpp
    pass_bindings.cpp
    builder_bindings.cpp
)

pybind11_add_module(pypto_impl MODULE
    ${BINDING_SOURCES}
)

target_link_libraries(pypto_impl PRIVATE ir_proto pass_proto)

target_include_directories(pypto_impl 
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

# Place the built module next to the python package so that `import pypto_impl` works.
set_target_properties(pypto_impl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/pypto"
)

