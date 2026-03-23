#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import re
import sys
import matplotlib.pyplot as plt
import numpy as np

# 检查参数
if len(sys.argv) < 2:
    print("错误: 必须传入 log 文件路径作为参数")
    print("用法: python3 plot_accuracy.py <verify_result.log>")
    sys.exit(1)

# 读取文件
log_file = sys.argv[1]
with open(log_file, 'r') as f:
    lines = f.readlines()

# 先找到所有检查点名称和对应的行号
checkpoint_pattern = r'^(\d+_[^:\s]+):'
checkpoints = []
checkpoint_line_indices = []

for i, line in enumerate(lines):
    match = re.match(checkpoint_pattern, line)
    if match:
        checkpoint_name = match.group(1)
        checkpoints.append(checkpoint_name)
        checkpoint_line_indices.append(i)

# 然后在每个检查点后面查找 Actual 行
results = []
for i, checkpoint in enumerate(checkpoints):
    start_line = checkpoint_line_indices[i]
    # 查找后面的 Actual 行（通常在检查点名称行后面5-10行）
    for j in range(start_line, min(start_line + 15, len(lines))):
        if 'Actual: rtol=' in lines[j]:
            # 使用更宽松的正则表达式匹配浮点数
            actual_match = re.search(r'Actual: rtol=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), atol=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', lines[j])
            if actual_match:
                rtol = actual_match.group(1)
                atol = actual_match.group(2)
                results.append((checkpoint, float(rtol), float(atol)))
                break

print(f'提取到 {len(results)} 个检查点的精度数据')
for i, (ckpt, rtol, atol) in enumerate(results):
    print(f'{i+1}. {ckpt}: rtol={rtol:.6f}, atol={atol:.6f}')

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 提取数据
checkpoints_list = [r[0] for r in results]
rtol_values = [r[1] for r in results]
atol_values = [r[2] for r in results]

# 绘制 rtol 折线图
ax1.plot(range(len(checkpoints_list)), rtol_values, marker='o', linewidth=2, markersize=8, color='blue')
ax1.set_xlabel('Checkpoint', fontsize=14)
ax1.set_ylabel('Relative Tolerance (rtol)', fontsize=14)
ax1.set_title('Relative Tolerance (rtol) Change Across Checkpoints', fontsize=16, fontweight='bold', y=1.02)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(len(checkpoints_list)))
ax1.set_xticklabels(checkpoints_list, rotation=45, ha='right', fontsize=11)

# 标注所有非0的点
for i, (ckpt, rtol) in enumerate(zip(checkpoints_list, rtol_values)):
    if rtol > 0:
        ax1.annotate(f'{rtol:.4f}', (i, rtol), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=8, color='red', fontweight='bold')

# 绘制 atol 折线图
ax2.plot(range(len(checkpoints_list)), atol_values, marker='s', linewidth=2, markersize=8, color='red')
ax2.set_xlabel('Checkpoint', fontsize=14)
ax2.set_ylabel('Absolute Tolerance (atol)', fontsize=14)
ax2.set_title('Absolute Tolerance (atol) Change Across Checkpoints', fontsize=16, fontweight='bold', y=1.02)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(len(checkpoints_list)))
ax2.set_xticklabels(checkpoints_list, rotation=45, ha='right', fontsize=11)

# 标注所有非0的点
for i, (ckpt, atol) in enumerate(zip(checkpoints_list, atol_values)):
    if atol > 0:
        ax2.annotate(f'{atol:.2f}', (i, atol), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=8, color='blue', fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_change.png', dpi=300, bbox_inches='tight')
print("\n精度变化折线图已保存至: accuracy_change.png")

# 打印数据表格
print("\n精度数据汇总:")
print("-" * 80)
print(f"{'Checkpoint':<30} {'rtol':<15} {'atol':<15} {'Status':<10}")
print("-" * 80)
for ckpt, rtol, atol in results:
    status = "FAIL" if (rtol > 0.0001 or atol > 1e-5) else "PASS"
    print(f"{ckpt:<30} {rtol:<15.6f} {atol:<15.6f} {status:<10}")
print("-" * 80)
