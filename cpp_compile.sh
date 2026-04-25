export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/GBowen666/pto-isa
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_MODULE_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH=/mnt/workspace/gitCode/GBowen666/pypto/logs

cd /mnt/workspace/gitCode/GBowen666/pypto/logs
rm -r *

cd /mnt/workspace/gitCode/GBowen666/pypto/

rm -r build
mkdir build
cd build
cmake ../ -DENABLE_UTEST=ON -DENABLE_FEATURE_PYTHON_FRONT_END=OFF
make -j16

cd output/bin

# ./tile_fwk_utest --gtest_filter=LiteNPUCodeGenMatmul.*
./tile_fwk_utest --gtest_filter=LiteNPUCodeGenMatmul.test_matmul_001
# ./tile_fwk_utest --gtest_filter=LiteNPUCodeGenMatmul.test_matmul_011
# ./tile_fwk_utest --gtest_filter=LiteNPUCodeGenMatmul.test_matmul_s8s8_004

# ./tile_fwk_utest --gtest_filter=LiteNPUCodeGenSigmoid.test_sigmoid_fp16_001
# ./tile_fwk_utest --gtest_filter=LiteNPUCodeGenIndexPut.*

# ./tile_fwk_utest --gtest_filter=LiteNPUCodeGenIndexPut.test_index_put_012

# ./tile_fwk_utest --gtest_filter=LiteNPUCodeGenSigmoid.*
