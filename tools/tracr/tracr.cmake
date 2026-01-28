# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "Compiling the TraCR post processing script 'tracr_process' in: ${CMAKE_CURRENT_BINARY_DIR}/output/bin")

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/paraver/state.cfg
    ${CMAKE_CURRENT_BINARY_DIR}/output/bin/state.cfg
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/custom_info.json
    ${CMAKE_CURRENT_BINARY_DIR}/output/bin/custom_info.json
    COPYONLY
)

add_executable(tracr_process ${CMAKE_CURRENT_LIST_DIR}/tracr_process.cpp)

tracr_enable(tracr_process)