# tools/tracr/tracr.cmake

message(WARNING "TraCR tools path: ${CMAKE_SOURCE_DIR}")
message(WARNING "TraCR tools path: ${CMAKE_BINARY_DIR}")

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/paraver/state.cfg
    ${CMAKE_CURRENT_BINARY_DIR}/output/bin/state.cfg
    COPYONLY
)

add_executable(tracr_process ${CMAKE_CURRENT_LIST_DIR}/tracr_process.cpp)

tracr_enable(tracr_process)