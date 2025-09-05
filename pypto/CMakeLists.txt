cmake_minimum_required(VERSION 3.16.0)
project(pypto)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")  # "Debug"

# main ascendcpp path, now just need to point to parent dir, but can also be built as out-of-source extension
if(NOT DEFINED ENV{ASCENDCPP_DIR})
  message(FATAL_ERROR "ASCENDCPP_DIR is not set")
else()
  set(ASCENDCPP_DIR $ENV{ASCENDCPP_DIR})
endif()

message("***********************************************************")
message("* ASCENDCPP_DIR             : ${ASCENDCPP_DIR}")
message("***********************************************************")


find_package(PythonLibs REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# AscendC++ libraries required from PyPTO
set(ASCENDCPP_LIBS
    tile_fwk_interface
    tile_fwk_codegen
    tile_fwk_operator
    tile_fwk_compiler
    tile_fwk_passes
    tile_fwk_simulation
)

foreach(LIB ${ASCENDCPP_LIBS})
    find_library(${LIB} NAMES ${LIB}
                            PATHS ${ASCENDCPP_DIR}/build/src/interface
                                  ${ASCENDCPP_DIR}/build/src/operator
                                  ${ASCENDCPP_DIR}/build/src/passes
                                  ${ASCENDCPP_DIR}/build/src/machine
                                  ${ASCENDCPP_DIR}/build/src/codegen
                                  ${ASCENDCPP_DIR}/build/src/cost_model/simulation
                            NO_DEFAULT_PATH)
    if(${LIB})
        message(STATUS "Found ${LIB}: ${${LIB}}")
        list(APPEND FOUND_LIBS ${${LIB}})
    else()
        message(FATAL_ERROR "${LIB} not found")
    endif()
endforeach()


pybind11_add_module(pto src/pybind11.cpp)

target_sources(pto PRIVATE
  ${ASCENDCPP_DIR}/src
)

target_link_libraries(pto PRIVATE
  c_sec
  tile_fwk_passes
  tile_fwk_interface
  tile_fwk_codegen
  tile_fwk_operator
  tile_fwk_compiler
  tile_fwk_simulation
)

target_link_directories(pto PRIVATE
  ${ASCENDCPP_DIR}/build/src/passes
  ${ASCENDCPP_DIR}/build/src/interface
  ${ASCENDCPP_DIR}/build/src/operator
  ${ASCENDCPP_DIR}/build/src/machine
  ${ASCENDCPP_DIR}/build/src/codegen
  ${ASCENDCPP_DIR}/build/src/cost_model/simulation
  ${ASCENDCPP_DIR}/build/3rd/libboundscheck-v1.1.16/output/lib/
)

target_include_directories(pto PRIVATE
  ${ASCENDCPP_DIR}/include
  ${ASCENDCPP_DIR}/src
  ${ASCENDCPP_DIR}/build/3rd/libboundscheck-v1.1.16/output/include/
)

target_compile_options(pto PRIVATE
  -D_GLIBCXX_USE_CXX11_ABI=0
)

set_target_properties(pto PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
  CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
)
