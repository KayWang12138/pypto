#!/bin/bash

# 设置环境变量
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/GBowen666/pto-isa

# 切换到项目根目录
cd /mnt/workspace/gitCode/GBowen666/pypto/

# 清理并创建构建目录
echo "=== 清理构建目录 ==="
rm -rf build
mkdir -p build
cd build

# 配置并编译项目
echo "=== 配置项目 ==="
cmake ../ -DENABLE_UTEST=ON -DENABLE_FEATURE_PYTHON_FRONT_END=OFF

echo "=== 编译项目 ==="
make -j16

# 切换到测试执行目录
cd output/bin

# 运行测试用例
echo "=== 运行测试用例 ==="
echo ""

# 定义测试用例数组
TEST_CASES=(
    "LiteNPUCodeGenAmax.test_amax_fp32_2d_axis_neg1_keepdims_true"
    "LiteNPUCodeGenAmax.test_amax_fp32_2d_axis_0_keepdims_false"
    "LiteNPUCodeGenAmax.test_amax_fp32_2d_axis_1_keepdims_true"
    "LiteNPUCodeGenAmax.test_amax_fp16_2d_axis_neg1_keepdims_false"
    "LiteNPUCodeGenAmax.test_amax_fp32_1d_axis_neg1_keepdims_true"
    "LiteNPUCodeGenAmax.test_amax_fp32_3d_axis_1_keepdims_true"
    "LiteNPUCodeGenAmax.test_amax_fp32_4d_axis_2_keepdims_false"
    "LiteNPUCodeGenAmin.test_amin_fp32_2d_axis_neg1_keepdims_true"
    "LiteNPUCodeGenAmin.test_amin_fp32_2d_axis_0_keepdims_false"
    "LiteNPUCodeGenAmin.test_amin_fp32_2d_axis_1_keepdims_true"
    "LiteNPUCodeGenAmin.test_amin_fp16_2d_axis_neg1_keepdims_false"
    "LiteNPUCodeGenAmin.test_amin_fp32_1d_axis_neg1_keepdims_true"
    "LiteNPUCodeGenAmin.test_amin_fp32_3d_axis_1_keepdims_true"
    "LiteNPUCodeGenAmin.test_amin_fp32_4d_axis_2_keepdims_false"
    "LiteNPUCodeGenAssemble.test_assemble_fp32_2d_axis_0"
    "LiteNPUCodeGenAssemble.test_assemble_fp32_2d_axis_1"
    "LiteNPUCodeGenAssemble.test_assemble_fp16_2d_axis_0"
    "LiteNPUCodeGenAssemble.test_assemble_fp32_3d_axis_1"
    "LiteNPUCodeGenAssemble.test_assemble_fp32_4d_axis_2"
    "LiteNPUCodeGenAssemble.test_assemble_fp32_multiple_tensors_axis_0"
    "LiteNPUCodeGenAssemble.test_assemble_call_fp32_2d_parallel_true"
    "LiteNPUCodeGenAssemble.test_assemble_call_fp32_2d_parallel_false"
    "LiteNPUCodeGenAssemble.test_assemble_call_fp32_1d"
    "LiteNPUCodeGenAssemble.test_assemble_call_fp32_3d"
    "LiteNPUCodeGenAssemble.test_assemble_call_fp16_2d_parallel_true"
    "LiteNPUCodeGenTranspose.test_transpose_fp32_2d_axis_neg1_0"
    "LiteNPUCodeGenTranspose.test_transpose_fp32_2d_axis_1_0"
    "LiteNPUCodeGenTranspose.test_transpose_fp16_2d_axis_1_0"
    "LiteNPUCodeGenTranspose.test_transpose_fp32_3d_axis_0_2_1"
    "LiteNPUCodeGenTranspose.test_transpose_fp32_3d_axis_2_1_0"
    "LiteNPUCodeGenTranspose.test_transpose_fp32_4d_axis_0_1_3_2"
    "LiteNPUCodeGenTranspose.test_transpose_fp32_4d_axis_3_2_1_0"
    "LiteNPUCodeGenUnsqueeze.test_unsqueeze_fp32_1d_axis_0"
    "LiteNPUCodeGenUnsqueeze.test_unsqueeze_fp32_1d_axis_neg1"
    "LiteNPUCodeGenUnsqueeze.test_unsqueeze_fp32_2d_axis_0"
    "LiteNPUCodeGenUnsqueeze.test_unsqueeze_fp32_2d_axis_1"
    "LiteNPUCodeGenUnsqueeze.test_unsqueeze_fp32_2d_axis_neg1"
    "LiteNPUCodeGenUnsqueeze.test_unsqueeze_fp16_2d_axis_0"
    "LiteNPUCodeGenUnsqueeze.test_unsqueeze_fp32_3d_axis_1"
    "LiteNPUCodeGenUnsqueeze.test_unsqueeze_fp32_4d_axis_2"
    "LiteNPUCodeGenView.test_view_fp32_2d_to_2d"
    "LiteNPUCodeGenView.test_view_fp32_2d_to_1d"
    "LiteNPUCodeGenView.test_view_fp32_1d_to_2d"
    "LiteNPUCodeGenView.test_view_fp32_2d_to_3d"
    "LiteNPUCodeGenView.test_view_fp32_3d_to_2d"
    "LiteNPUCodeGenView.test_view_fp32_3d_to_4d"
    "LiteNPUCodeGenView.test_view_fp16_2d_to_2d"
    "LiteNPUCodeGenView.test_view_fp32_4d_to_2d"
    "LiteNPUCodeGenView.test_view_fp32_2d_to_4d"
    "LiteNPUCodeGenReshape.test_reshape_fp32_2d_to_2d"
    "LiteNPUCodeGenReshape.test_reshape_fp32_2d_to_1d"
    "LiteNPUCodeGenReshape.test_reshape_fp32_1d_to_2d"
    "LiteNPUCodeGenReshape.test_reshape_fp32_2d_to_3d"
    "LiteNPUCodeGenReshape.test_reshape_fp32_3d_to_2d"
    "LiteNPUCodeGenReshape.test_reshape_fp32_3d_to_4d"
    "LiteNPUCodeGenReshape.test_reshape_fp16_2d_to_2d"
    "LiteNPUCodeGenReshape.test_reshape_fp32_4d_to_2d"
    "LiteNPUCodeGenReshape.test_reshape_fp32_2d_to_4d"
    "LiteNPUCodeGenReshape.test_reshape_fp32_3d_to_1d"
)

# 初始化统计变量
total_tests=${#TEST_CASES[@]}
passed_tests=0
failed_tests=0
failed_cases=()

# 运行所有测试用例
for test_case in "${TEST_CASES[@]}"; do
    echo "运行测试: $test_case"
    ./tile_fwk_utest --gtest_filter="$test_case"
    
    if [ $? -eq 0 ]; then
        echo "✅ 通过: $test_case"
        ((passed_tests++))
    else
        echo "❌ 失败: $test_case"
        ((failed_tests++))
        failed_cases+=("$test_case")
    fi
    echo ""
done

# 输出测试结果统计
echo "=== 测试结果统计 ==="
echo "总测试用例数: $total_tests"
echo "通过测试用例数: $passed_tests"
echo "失败测试用例数: $failed_tests"

if [ $failed_tests -gt 0 ]; then
    echo ""
    echo "失败的测试用例:"
    for failed_case in "${failed_cases[@]}"; do
        echo "  - $failed_case"
    done
    exit 1
else
    echo ""
    echo "🎉 所有测试用例通过！"
    exit 0
fi