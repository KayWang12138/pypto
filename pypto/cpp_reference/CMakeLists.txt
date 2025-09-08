cmake_minimum_required(VERSION 3.16.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")  # "Debug"

project(standalone_ascpp_reference)

# main ascendcpp path, now just need to point to parent dir, but can also be built as out-of-source extension
if(NOT DEFINED ENV{ASCENDCPP_DIR})
  message(FATAL_ERROR "ASCENDCPP_DIR is not set")
else()
  set(ASCENDCPP_DIR $ENV{ASCENDCPP_DIR})
endif()

message("***********************************************************")
message("* ASCENDCPP_DIR             : ${ASCENDCPP_DIR}")
message("***********************************************************")

# can add new standalone cpp examples here
add_executable(dynamic_attention src/dynamic_attention.cpp)
add_executable(dynamic_loop_nomacro src/dynamic_loop_nomacro.cpp)
add_executable(dynamic_loop src/dynamic_loop.cpp)
add_executable(print_dtype src/print_dtype.cpp)
add_executable(dump_graph src/dump_graph.cpp)
add_executable(vector_add src/vector_add.cpp)
add_executable(llama_graph src/llama_graph.cpp)
add_executable(manual_record src/manual_record.cpp)

set(
  EXECUTABLES
  dynamic_attention
  dynamic_loop_nomacro
  dynamic_loop
  print_dtype
  dump_graph
  vector_add
  llama_graph
  manual_record
)

# below are common ascendcpp dependencies, usually not need to modify
set (TILE_FWK_LIBS
  tile_fwk_interface
  tile_fwk_passes
  tile_fwk_operator
  c_sec
)

set (TILE_FWK_LIB_DIRS
  ${ASCENDCPP_DIR}/build/src/interface
  ${ASCENDCPP_DIR}/build/src/operator
  ${ASCENDCPP_DIR}/build/src/passes
  ${ASCENDCPP_DIR}/build/3rd/libboundscheck-v1.1.16/output/lib
)

set (TILE_FWK_INC_DIRS
  ${ASCENDCPP_DIR}/include
  ${ASCENDCPP_DIR}/src
  ${ASCENDCPP_DIR}/build/3rd/libboundscheck-v1.1.16/output/include/
)
# NOTE: some extra headers are under `src`, not only in `include` 

foreach(EXE IN LISTS EXECUTABLES)
  target_link_libraries(${EXE} PUBLIC ${TILE_FWK_LIBS})
  target_link_directories(${EXE} PUBLIC ${TILE_FWK_LIB_DIRS})
  target_include_directories(${EXE} PUBLIC ${TILE_FWK_INC_DIRS})
  target_compile_options(${EXE} PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=0
  )
  # NOTE: to avoid linking error `undefined reference` related to `std::__cxx11`
endforeach()
