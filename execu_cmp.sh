#!/bin/bash

# # 清理并创建构建目录
# echo "=== 清理构建目录 ==="
# rm -rf build
# mkdir -p build
# cd build

# # 配置并编译项目
# echo "=== 配置项目 ==="
# cmake ../ -DENABLE_UTEST=ON -DENABLE_FEATURE_PYTHON_FRONT_END=OFF

# echo "=== 编译项目 ==="
# make -j16

# 测试程序路径
TEST_BIN="./output/bin/tile_fwk_utest"
LOG_FILE="test_cmp_output.log"
> $LOG_FILE

echo "===== 开始执行测试，日志将保存到 $LOG_FILE ====="

# ======================
# 批量拼接生成测试用例
# ======================
TEST_CASES=()

# 前缀固定
PREFIX="LiteNPUCodeGenCompare.test_compare_eq_"

# 1. fp16: 001~034
for i in {001..034}; do
    TEST_CASES+=("${PREFIX}fp16_$i")
done

# 2. fp32: 001~038
for i in {001..038}; do
    TEST_CASES+=("${PREFIX}fp32_$i")
done

# 遍历执行
for case in "${TEST_CASES[@]}"; do
    echo -e "\n========================================" | tee -a $LOG_FILE
    echo "执行用例：$case"                          | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    $TEST_BIN --gtest_filter="$case" 2>&1 | tee -a $LOG_FILE
done

echo -e "\n===== 所有用例执行完成 =====" | tee -a $LOG_FILE
echo "完整日志：$LOG_FILE"