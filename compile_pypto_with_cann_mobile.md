1、安装需要 linux x86 的环境，当前已经创建好docker，直接使用即可
```bash
docker start -ai pypto-dev
```
2、设置环境变量，当前docker中已经安装了cann-mobile，只需要设置环境变量即可
```bash
export ASCEND_PATH=/opt/cann_mobile/
export PTO_TILE_LIB_CODE_PATH=/opt/pto-isa/
export PYPTO_THIRD_PARTY_PATH=/workspace/third_party_path/
source /opt/cann_mobile/ascend-toolkit/latest/set_env.sh
export LD_LIBRARY_PATH=/opt/cann_mobile/ascend-toolkit/latest/x86_64-linux/simulator/Kirin9030/lib/:$LD_LIBRARY_PATH
```
3、编译命令
```bash
PYPTO_BUILD_EXT_ARGS='--cmake-build-type Release --cmake-options "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_WITH_CANN_MOBILE=ON"' python3 -m pip install -e . --verbose
```