#!/bin/bash
# multi_npu_test.sh

# 设置要使用的 NPU 卡号列表（根据你的机器情况修改）
# NPUS=(0 1 2 3 4 5 6 7)   # 例如使用 8 张卡
NPUS=(0 1)   # 例如使用 2 张卡

# 获取脚本所在目录（可选，确保路径正确）
SCRIPT_PATH="./test_fused_cross_entropy.py"

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Test script not found at $SCRIPT_PATH"
    exit 1
fi

# 要测试的 pypto 版本
VERSION=3
TIME=$(date +%Y%m%d%H%M%S)
# 启动每个 NPU 上的独立进程
for npu in "${NPUS[@]}"; do
    echo "Launching test on NPU $npu with version $VERSION ..."

    python "$SCRIPT_PATH" --version "$VERSION" --npu "$npu" > "./logs/log_npu${npu}_${TIME}_fused_cross_entropy.log" 2>&1 &
done

# 等待所有后台任务完成
wait

echo "All NPU tests completed. Check logs/log_npu*.log for details."