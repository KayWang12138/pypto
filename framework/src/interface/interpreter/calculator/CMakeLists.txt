# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.3)

project(LibTileFwkCalculator
        LANGUAGES C CXX
        VERSION 1.0.0
)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# # 查找 torch 库
# find_package(Torch REQUIRED)

# # 检查是否找到 torch
# if (NOT Torch_FOUND)
#     message(FATAL_ERROR "Torch library not found. Please install PyTorch C++ library (libtorch).")
# endif()

set(TorchPath /usr/local/python3.11/lib/python3.11/site-packages/torch)

# 生成共享库
add_library(tile_fwk_calculator SHARED)
target_sources(tile_fwk_calculator
        PRIVATE
            calc_torch.cpp
)

# 设置编译选项
target_include_directories(tile_fwk_calculator
        PRIVATE
            ${TorchPath}/include
            ${TorchPath}/include/torch/csrc/api/include
            /home/wangfei/pypto/build_out/pypto/lib/include
)
target_link_directories(tile_fwk_calculator
        PUBLIC
            ${TorchPath}/lib
            ${TorchPath}/../torch.libs>
)
# 链接 torch 库
target_link_libraries(tile_fwk_calculator PRIVATE${TORCH_LIBRARIES})

# 设置输出路径（可选）
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY${PROJECT_BINARY_DIR}/lib)
