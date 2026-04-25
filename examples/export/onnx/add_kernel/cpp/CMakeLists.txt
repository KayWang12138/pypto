cmake_minimum_required(VERSION 3.15)
project(op_probe LANGUAGES CXX)

# All wiring (Python, Ascend, rpath) lives in the shared op_probe.cmake.
include("${CMAKE_CURRENT_LIST_DIR}/../../../op_probe/op_probe.cmake")
add_op_probe_target()
