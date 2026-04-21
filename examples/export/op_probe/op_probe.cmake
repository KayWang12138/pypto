# Shared CMake logic for per-op smoke probes. Each example's
# {onnx,torchair}/<example>/cpp[_<op>]/CMakeLists.txt reduces to:
#
#   cmake_minimum_required(VERSION 3.15)
#   project(op_probe LANGUAGES CXX)
#   include("<relative>/examples/export/op_probe.cmake")
#   add_op_probe_target()
#
# The companion header (op_probe_core.hpp, sibling of this file) is added
# to the target's include path automatically by add_op_probe_target().

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# COMPONENTS Interpreter makes CMake honor a -DPython_EXECUTABLE hint
# (Development.Embed alone ignores it), which is how build_and_run.sh
# pins the probe to the same interpreter the .so was built against.
find_package(Python COMPONENTS Interpreter Development.Embed REQUIRED)
message(STATUS "Probe Python: ${Python_EXECUTABLE} (${Python_VERSION})")

# Resolve Ascend arch subtree (aarch64-linux / x86_64-linux) under
# $ASCEND_HOME_PATH — same convention as load_and_compile.py.
if(NOT DEFINED ASCEND_HOME)
    set(_ascend_root "$ENV{ASCEND_HOME_PATH}")
    if(NOT _ascend_root)
        set(_ascend_root "/usr/local/Ascend/cann")
    endif()
    if(EXISTS "${_ascend_root}/aarch64-linux/include")
        set(ASCEND_HOME "${_ascend_root}/aarch64-linux")
    elseif(EXISTS "${_ascend_root}/x86_64-linux/include")
        set(ASCEND_HOME "${_ascend_root}/x86_64-linux")
    else()
        set(ASCEND_HOME "${_ascend_root}")
    endif()
endif()
message(STATUS "ASCEND_HOME=${ASCEND_HOME}")

# Directory containing this .cmake (and op_probe_core.hpp).
set(_OP_PROBE_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(add_op_probe_target)
    add_executable(op_probe custom_op_def_demo.cpp)

    # -rdynamic so libpython symbols linked into this binary are visible to
    # the dlopen'd .so (pybind11 looks up PyInstanceMethod_Type etc. via
    # the host process's dynamic symbol table).
    set_target_properties(op_probe PROPERTIES ENABLE_EXPORTS ON)

    target_include_directories(op_probe PRIVATE
        "${_OP_PROBE_INCLUDE_DIR}"
        "${ASCEND_HOME}/include"
    )
    target_link_directories(op_probe PRIVATE "${ASCEND_HOME}/lib64")

    # --no-as-needed around libpython: the demo .cpp reaches Py* symbols
    # mostly via the loaded module; without this the linker would drop
    # the libpython dependency and the .so's pybind call paths would fail.
    # libregister provides ops::OpDefFactory; libgraph[_base] provide
    # ge::AscendString / ge::DataType; libexe_graph provides gert::Shape.
    target_link_libraries(op_probe PRIVATE
        ${CMAKE_DL_LIBS}
        "-Wl,--no-as-needed"
        Python::Python
        "-Wl,--as-needed"
        register
        graph
        graph_base
        exe_graph
    )

    # Ask the pinned interpreter where its libpython lives and bake that
    # dir into the rpath so the probe runs without LD_LIBRARY_PATH set.
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
        "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
        OUTPUT_VARIABLE PYTHON_LIBDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Python libdir: ${PYTHON_LIBDIR}")

    set_target_properties(op_probe PROPERTIES
        INSTALL_RPATH "${ASCEND_HOME}/lib64;${PYTHON_LIBDIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endfunction()
