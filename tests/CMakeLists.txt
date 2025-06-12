# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

include(cmake/config.cmake)
include(cmake/func.cmake)

if (ENABLE_TESTS_UTEST)
    add_executable(ascendcpp_utest)

    target_sources(ascendcpp_utest
            PRIVATE
                main.cpp
    )

    target_link_libraries(ascendcpp_utest
            PRIVATE
                GTest::gtest
                ascendcpp_stubs
    )
    if (ENABLE_TESTS_EXECUTE)
        add_custom_command(
            TARGET ascendcpp_utest POST_BUILD
            COMMAND ./ascendcpp_utest
            COMMENT "Run GTest(ascendcpp_utest)"
        )
        AscendCpp_GTest_GenerateCoverage(TARGET ascendcpp_utest)
    endif ()
endif ()

if (ENABLE_TESTS_STEST)
    add_executable(ascendcpp_stest)

    target_sources(ascendcpp_stest
            PRIVATE
                main.cpp
    )

    target_link_libraries(ascendcpp_stest
            PRIVATE
                GTest::gtest
                ascendcpp_stubs
    )
    if (ENABLE_TESTS_EXECUTE)
        add_custom_command(
            TARGET ascendcpp_stest POST_BUILD
            COMMAND ./ascendcpp_stest
            COMMENT "Run GTest(ascendcpp_stest)"
        )
    endif ()
endif ()
