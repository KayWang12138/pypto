# /bin/env

# 1. 检查 TILE_FWK_DEVICE_ID 是否设置
if [ -z "${TILE_FWK_DEVICE_ID}" ]; then
    echo "❌ TILE_FWK_DEVICE_ID 变量未设置（值为空）, 设置device 0......"
    export TILE_FWK_DEVICE_ID=0
fi

# 2. 检查PTO_TILE_LIB_CODE_PATH是否设置
if [ -z "${PTO_TILE_LIB_CODE_PATH}" ]; then
    echo "❌ PTO_TILE_LIB_CODE_PATH 变量未设置（值为空）, 尝试设置......"
    export PTO_TILE_LIB_CODE_PATH=./pto_isa/pto-isa/
fi
if [ -d "${PTO_TILE_LIB_CODE_PATH}" ]; then
    echo "✅ 目录存在：${PTO_TILE_LIB_CODE_PATH}"
else
    echo "❌ 目录不存在：${PTO_TILE_LIB_CODE_PATH}, 尝试拉取源码......"
    mkdir -p ./pto_isa && cd ./pto_isa && git clone https://gitcode.com/cann/pto-isa.git
fi
