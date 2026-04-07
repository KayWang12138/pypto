#!/bin/bash
# PyPTO 性能退化二分查找 - 测试脚本
# 用于 git bisect run

set -e

# 获取当前 commit 信息
CURRENT_COMMIT=$(git rev-parse --short HEAD)
CURRENT_COMMIT_MSG=$(git log --format="%s" -1 HEAD)

# 编译
echo "编译 PyPTO..."
rm -rf build_out
python3 build_ci.py -f=python3 > /dev/null 2>&1
pip install build_out/pypto*.whl --force-reinstall --no-deps > /dev/null 2>&1

# 运行测试
echo "运行测试..."
rm -rf output/*
python3 $TEST_CASE > /dev/null 2>&1

# 保存输出结果到带版本信息的目录
OUTPUT_DIR="output_${CURRENT_COMMIT}"
echo "保存输出结果到: ${OUTPUT_DIR}"
cp -r output "${OUTPUT_DIR}"

# 保存 commit 信息到输出目录
cat > "${OUTPUT_DIR}/commit_info.txt" << EOF
Commit: ${CURRENT_COMMIT}
Message: ${CURRENT_COMMIT_MSG}
Date: $(git log --format="%ai" -1 HEAD)
Author: $(git log --format="%an <%ae>" -1 HEAD)
EOF

# 分析性能
echo "分析性能..."
python3 scripts/analyze_performance.py \
    --output-dir "${OUTPUT_DIR}/" \
    --analysis-method "$ANALYSIS_METHOD" \
    --good-threshold $GOOD_THRESHOLD \
    --bad-threshold $BAD_THRESHOLD

# 返回分析脚本的退出码
