cmake_minimum_required(VERSION 3.16)
project(SumLstmTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===================== 路径配置 =====================
# ASCEND_HOME 可通过环境变量或命令行参数指定
if(NOT DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/8.0.0")
else()
    set(ASCEND_HOME $ENV{ASCEND_HOME})
endif()

# INC_PATH: 头文件根目录
set(INC_PATH "${ASCEND_HOME}")

# LIB_PATH: 库文件目录（根据架构选择）
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(LIB_PATH "${ASCEND_HOME}/runtime/lib64")
else()
    set(LIB_PATH "${ASCEND_HOME}/runtime/lib64")
endif()

# OP_API_PATH: 自定义算子API路径
set(OP_API_PATH "${ASCEND_HOME}/opp/vendors/customize/op_api")

message(STATUS "===========================================")
message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")
message(STATUS "INC_PATH: ${INC_PATH}")
message(STATUS "LIB_PATH: ${LIB_PATH}")
message(STATUS "OP_API_PATH: ${OP_API_PATH}")
message(STATUS "===========================================")

# ===================== 头文件搜索路径 =====================
include_directories(
    ${INC_PATH}/runtime/include
    ${INC_PATH}/atc/include
    ${OP_API_PATH}/include
)

# ===================== 库文件搜索路径 =====================
link_directories(
    ${LIB_PATH}
    ${OP_API_PATH}/lib
)

# ===================== 编译选项 =====================
add_compile_options(-O2 -Wall)

# ===================== 可执行文件 =====================
add_executable(execute_sum_lstm main.cpp)

# ===================== 链接库（按官方文档顺序）=====================
target_link_libraries(execute_sum_lstm
    ascendcl
    cust_opapi
    nnopbase
    stdc++
)

# ===================== RPATH设置（运行时库查找）=====================
set_target_properties(execute_sum_lstm PROPERTIES
    BUILD_RPATH "${LIB_PATH};${OP_API_PATH}/lib"
    INSTALL_RPATH "${LIB_PATH};${OP_API_PATH}/lib"
)

# ===================== 安装 =====================
install(TARGETS execute_sum_lstm DESTINATION bin)
