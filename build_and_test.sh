#!/bin/bash
set -e

export ASCEND_PATH=/opt/cann_mobile/
export PTO_TILE_LIB_CODE_PATH=/opt/pto-isa/
export PYPTO_THIRD_PARTY_PATH=/workspace/third_party_path/
source /opt/cann_mobile/ascend-toolkit/latest/set_env.sh
export LD_LIBRARY_PATH=/opt/cann_mobile/ascend-toolkit/latest/x86_64-linux/simulator/Kirin9030/lib/:$LD_LIBRARY_PATH

cd /workspace

echo "=== 编译 PyPTO (CANN Mobile) ==="
PYPTO_BUILD_EXT_ARGS='--cmake-build-type Release --cmake-options "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_WITH_CANN_MOBILE=ON"' python3 -m pip install -e . --verbose 2>&1 | tail -5
echo "=== 编译完成 ==="

echo "=== 拷贝 Kirin9030.ini ==="
cp framework/src/platform/parser/simulation_platform/platform_config/Kirin9030.ini python/pypto/lib/configs/

echo "=== 执行 hello_world 用例 ==="
python3 examples/00_hello_world/hello_world.py --run_mode='sim'
