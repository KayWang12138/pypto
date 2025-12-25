# PyPTO 编程入门介绍样例 (Introduction)

本样例展示了 PyPTO 的编程入门介绍，适合初学者快速上手。

## 总览介绍

本样例涵盖了 PyPTO 编程中的基本语法介绍，包括：
- Tensor 创建、使用及计算
- JIT 编译与执行
- 循环与数据切分
- 条件与分支（静态/动态）

## 代码结构

- **`add_direct.py`**: 入门示例 - 直接加法操作
- **`add_scalar.py`**: 入门示例 - 标量加法
- **`add_scalar_loop.py`**: 循环示例 - 展示如何使用循环处理数据
- **`add_scalar_loop_dyn_axis.py`**: 动态 Shape 示例 - 支持动态维度
- **`add_scalar_loop_multi_jit.py`**: 多 JIT 示例 - 多个 JIT 函数协同工作
- **`add_scalar_loop_view_assemble.py`**: View/Assemble 示例 - 张量视图和组装操作
- **`add_scalar_loop_dyn_axis_static_cond.py`**: 静态分支示例 - 编译时条件分支
- **`add_scalar_loop_dyn_axis_dyn_cond.py`**: 动态分支示例 - 运行时条件判断
- **`add_scalar_loop_dyn_axis_dyn_loop_cond.py`**: 动态循环条件示例 - 循环边界检测

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
# 运行单个示例
python3 add_direct.py
python3 add_scalar.py
python3 add_scalar_loop_dyn_axis.py

# 运行条件分支示例
python3 add_scalar_loop_dyn_axis_static_cond.py
python3 add_scalar_loop_dyn_axis_dyn_cond.py
python3 add_scalar_loop_dyn_axis_dyn_loop_cond.py
```

## 关键概念

### 动态 Shape vs 静态 Shape
- **静态 Shape**: 在编译时确定，性能更优
- **动态 Shape**: 支持运行时变化的维度（如 batch size），使用 `pypto.frontend.dynamic()` 标记

## 注意事项

- **环境**: 确保 `torch_npu` 已正确安装并能识别到昇腾设备
- **设备 ID**: 运行前需设置 `TILE_FWK_DEVICE_ID` 环境变量
- **性能**: 静态 Shape 和静态分支通常比动态版本性能更好，但灵活性较低

