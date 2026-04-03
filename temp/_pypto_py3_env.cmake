
# Python3 Version
set(PYTHON3_VERSION_ID "3.11")
message(STATUS "PYTHON3_VERSION_ID=${PYTHON3_VERSION_ID}")

# Python3 module pybind11
get_filename_component(PY3_MOD_PYBIND11_CMAKE_DIR "/home/fangrui/.local/lib/python3.11/site-packages/pybind11/share/cmake/pybind11" REALPATH)
message(STATUS "PY3_MOD_PYBIND11_CMAKE_DIR=${PY3_MOD_PYBIND11_CMAKE_DIR}")

# Python3 module pybind11
set(PY3_MOD_TORCH_VERSION "2.8.0+cpu")
get_filename_component(PY3_MOD_TORCH_ROOT_PATH "/home/fangrui/.local/lib/python3.11/site-packages/torch" REALPATH)
get_filename_component(PY3_MOD_TORCH_CMAKE_DIR "/home/fangrui/.local/lib/python3.11/site-packages/torch/share/cmake" REALPATH)
set(PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI 1)
message(STATUS "PY3_MOD_TORCH_VERSION=${PY3_MOD_TORCH_VERSION}")
message(STATUS "PY3_MOD_TORCH_ROOT_PATH=${PY3_MOD_TORCH_ROOT_PATH}")
message(STATUS "PY3_MOD_TORCH_CMAKE_DIR=${PY3_MOD_TORCH_CMAKE_DIR}")
message(STATUS "PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI=${PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI}")
