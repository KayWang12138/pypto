cmake_minimum_required(VERSION 3.16)
# cmake_policy(SET CMP0148 NEW)

project(ir_spec_prototype LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PTO_ENABLE_TESTS "Build with GoogleTest tests" OFF)

add_subdirectory(src)

# =========================
# 只有开启时才引入测试
# =========================
if (PTO_ENABLE_TESTS)
    message(STATUS "PTO_ENABLE_TESTS = ON")
    enable_testing()
endif()

add_subdirectory(tests)

