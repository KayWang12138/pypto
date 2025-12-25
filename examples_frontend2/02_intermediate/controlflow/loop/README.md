# 循环 (Loop) 特性样例

本样例展示了 PyPTO 中循环控制的高级特性及其在算子内核中的使用规则。

## 总览介绍

在 PyPTO 的 JIT 内核中，循环不仅用于遍历数据，还涉及到编译期的优化和代码生成。本样例涵盖了以下关键点：
- **基础循环用法**: 包括 `start`, `end`, `step` 参数的使用。
- **循环起始与结束判定**: 如何判断循环的开始和结束。
- **循环展开 (Unroll)**: 使用 `unroll` 接口优化性能。
- **算子位置规则**: 算子通常必须放置在最内层循环中以确保正确生成代码。

## 代码文件说明

- **`loop.py`**: 包含循环特性的详细示例和验证逻辑。

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 执行脚本

```bash
# 运行所有循环特性示例
python3 loop.py

# 列出所有可用的用例
python3 loop.py --list
```

## 核心概念与规则

### 1. 基础用法
```python
for i in pypto.loop(start=0, end=10, step=1):
    # 循环体逻辑
```

### 2. 算子放置规则 (OP Position Rule)
为了确保计算指令能够被正确地分块（Tiling）并生成高性能的 NPU 代码，计算类算子（如 `add`, `matmul` 等）应当放置在内核的最内层循环中。

### 3. 循环展开
对于迭代次数较少的循环，可以使用展开技术来减少分支开销。

## 注意事项
- 循环的步长 `step` 必须是常数。
- 内核中的循环结构直接影响到 Tiling 策略的执行，复杂的循环嵌套可能需要更精细的分块配置。

