#!/bin/bash

# 测试程序路径
TEST_BIN="./pypto/build/output/bin/tile_fwk_utest"
LOG_FILE="test_max_output.log"
> $LOG_FILE

echo "===== 开始执行测试，日志将保存到 $LOG_FILE ====="

# ======================
# 批量拼接生成测试用例
# ======================
TEST_CASES=()

# 前缀固定
PREFIX="LiteNPUCodeGenMaximum.test_maximum_"

# 1. fp16: 001~018
for i in {001..018}; do
    TEST_CASES+=("${PREFIX}fp16_$i")
done

# 2. int16: 001~016
for i in {001..016}; do
    TEST_CASES+=("${PREFIX}int16_$i")
done

# 3. fp32: 001~018
for i in {001..018}; do
    TEST_CASES+=("${PREFIX}fp32_$i")
done

# 4. int32: 001~020
for i in {001..020}; do
    TEST_CASES+=("${PREFIX}int32_$i")
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