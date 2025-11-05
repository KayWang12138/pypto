# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (ENABLE_UTEST)
    add_subdirectory(stubs)

    add_subdirectory(operator)
    add_subdirectory(interface)
    add_subdirectory(passes)
    add_subdirectory(simulation)

    if (BUILD_WITH_CANN)
        add_subdirectory(machine)
        add_subdirectory(execute)
        add_subdirectory(codegen)
    endif ()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} "validate.py" "tile_fwk_config.json" "tile_fwk_config_schema.json"
        WORKING_DIRECTORY ${PTO_FWK_SRC_ROOT}/framework/src/interface/configs
        RESULT_VARIABLE VALIDATE_RESULT
    )
    if (NOT ${VALIDATE_RESULT} EQUAL 0)
        message(FATAL_ERROR "Config validation failed, please check config file")
    endif ()

    # 可执行程序
    PTO_Fwk_UTest_AddExe_RunExe(TARGET ${PTO_Fwk_UTestNamePrefix})
endif ()
