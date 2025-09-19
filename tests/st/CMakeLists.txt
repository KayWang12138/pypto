# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (ENABLE_TESTS_STEST OR ENABLE_TESTS_STEST_DISTRIBUTED)
    add_subdirectory(utils)
endif ()

if (ENABLE_TESTS_STEST)
    add_subdirectory(operator)
    add_subdirectory(operation)
    add_subdirectory(machine)
    add_subdirectory(simulation)

    # 可执行程序
    PTO_Fwk_STest_AddExe_RunExe(TARGET ${PTO_Fwk_STestNamePrefix})
endif ()

if (ENABLE_TESTS_STEST_DISTRIBUTED)
    add_subdirectory(distributed)
endif ()

if (ENABLE_BUILD_PTO_FWK_PYTHON AND ENABLE_TESTS_STEST)
    PTO_Fwk_STest_RunPytest()
endif ()
