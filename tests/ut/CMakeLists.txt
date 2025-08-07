# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (ENABLE_TESTS_UTEST)
    add_subdirectory(stubs)

    add_subdirectory(operator)
    add_subdirectory(interface)
    add_subdirectory(passes)
    add_subdirectory(simulation)

    if (NOT AC_ENABLE_FRAMEWORK_WITHOUT_CANN)
        add_subdirectory(machine)
        add_subdirectory(codegen)
    endif ()

    # 可执行程序
    TileFwk_UTest_AddExe_RunExe(TARGET ${TileFwkUTestNamePrefix})
endif ()
