# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


set(AI_CORE_FLAGS
    "-c -O3 -g -x cce -Wall -Werror -std=c++17 \
    --cce-aicore-only \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -D__MIX__ \
    -D__STATIC_KERNEL__ \
    -D__TILINGKEY__=0   \
    -D__OPTYPE__=tilefwk   \
    -I${PTO_FWK_SRC_ROOT}/framework/src \
    -I${PTO_FWK_SRC_ROOT}/framework/src/interface \
    -I${PTO_FWK_SRC_ROOT}/framework/include"
)

separate_arguments(AI_CORE_FLAGS)

add_custom_target(tile_fwk_core ALL DEPENDS aicore.cpp
    COMMAND ${BISHENG_PROGRAM} ${AI_CORE_FLAGS} -D__AIC__ --cce-aicore-arch=dav-c220-cube -o kernel_aic.o ${CMAKE_CURRENT_SOURCE_DIR}/aicore.cpp
    COMMAND ${BISHENG_PROGRAM} ${AI_CORE_FLAGS} -D__AIV__ --cce-aicore-arch=dav-c220-vec -o  kernel_aiv.o ${CMAKE_CURRENT_SOURCE_DIR}/aicore.cpp
    COMMAND ${BISHENG_LD} -m aicorelinux -Ttext=0 -static -n -o  kernel.o kernel_aic.o kernel_aiv.o
    COMMAND ld -r -b binary -o kernel_host.o kernel.o
    BYPRODUCTS kernel_host.o kernel.o
)
