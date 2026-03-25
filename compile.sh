cd /mnt/workspace/gitCode/GBowen666/pypto/

rm -r build
mkdir build
cd build
cmake ../ -DENABLE_UTEST=ON -DENABLE_FEATURE_PYTHON_FRONT_END=OFF
make -j384

cd output/bin
# ./tile_fwk_utest --gtest_filter=LiteNPUCodeGenMatmul.test_matmul_001
./tile_fwk_utest --gtest_filter=LiteNPUCodeGenMatmul.tmp_test_pow