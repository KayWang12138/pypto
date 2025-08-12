#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if [ -z "$ASCEND_HOME_PATH" ]; then
    if [ -z "$ASCEND_AICPU_PATH" ]; then
        echo "please set env"
        exit 1
    else 
        export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
    fi
else
    export ASCEND_HOME_PATH=$ASCEND_HOME_PATH
fi

echo "using ASCEND_HOME_PATH=$ASCEND_HOME_PATH"

BASEPATH=$(cd "$(dirname $0)"; pwd)
BUILD_PATH="${BASEPATH}/build"
OUTPUT_PATH="${BASEPATH}/output"
COMPUTE_UNIT=$1
if [ ! -n "${COMPUTE_UNIT}" ]
then
    COMPUTE_UNIT="ascend910b,ascend910_93"
fi
BINARY_OUTPUT_PATH=$2
if [ ! -n "${BINARY_OUTPUT_PATH}" ]
then
    BINARY_OUTPUT_PATH="${OUTPUT_PATH}/op_kernel/binary"
fi

build_ops()
{
    mkdir -p "${BUILD_PATH}"
    cd "${BUILD_PATH}"
    cmake -D ENABLE_BUILD_BINARY=True \
          -D ENABLE_BUILD_HOST=False \
          -D ASCEND_HOME_PATH=${ASCEND_HOME_PATH} \
          -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
          -D ASCEND_COMPUTE_UNIT=${COMPUTE_UNIT} \
          -D BINARY_OUTPUT_PATH=${BINARY_OUTPUT_PATH} \
          ..
    make -j32 VERBOSE=1 && make install
}

main() {
    echo "---------------Tile fwk ops build begin----------------"
    g++ -v
    cmake --version

    cd ${BASEPATH}/../
    python3 build.py --disable_auto_execute
    mkdir -p ${BASEPATH}/../output/lib64
    find ${BASEPATH}/../build -name "libtile_fwk_*.so" | xargs -I so_file cp so_file ${BASEPATH}/../output/lib64
    find ${BASEPATH}/../build -name "kernel.o" | xargs -I kl_file cp kl_file ${BASEPATH}/../output/lib64
    mkdir -p ${BASEPATH}/../output/conf
    ln -sf ${BASEPATH}/../src/interface/configs/tile_fwk_config.json ${BASEPATH}/../output/conf/tile_fwk_config.json
    ln -sf ${BASEPATH}/../src/passes/pass_config/tile_fwk_platform_info.json ${BASEPATH}/../output/conf/tile_fwk_platform_info.json
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${BASEPATH}/../output/lib64
    env

    cd ${BASEPATH}
    mkdir -p ${OUTPUT_PATH}
    build_ops || { echo "Tile fwk ops build failed."; exit 1; }
    echo "---------------Tile fwk ops build successfully----------------"
}

main "$@"