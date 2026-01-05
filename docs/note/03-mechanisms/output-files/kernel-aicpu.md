# kernel_aicpu 目录详细说明

## 目录概述

`kernel_aicpu` 目录包含 AICPU（AI CPU）相关的控制流编译产物，包括 Host 端和 Device 端的控制流代码、表达式定义文件等。

**目录位置**：输出目录下的 `kernel_aicpu/`（例如：`output/output_<timestamp>_<pid>/kernel_aicpu/`）

## 目录结构

```
kernel_aicpu/
├── controlFlow_host_10193433060527433398.cpp          # Host 端控制流 C++ 源文件
├── controlFlow_host_10193433060527433398.cpp.s        # Host 端汇编文件
├── controlFlow_host_10193433060527433398.cpp.o        # Host 端目标文件
├── controlFlow_host_10193433060527433398.cpp.bin      # Host 端二进制段（最终产物）
├── controlFlow_dev_10193433060527433398.cpp           # Device 端控制流 C++ 源文件
├── controlFlow_dev_10193433060527433398.cpp.s         # Device 端汇编文件
├── controlFlow_dev_10193433060527433398.cpp.o         # Device 端目标文件
├── controlFlow_dev_10193433060527433398.cpp.bin       # Device 端二进制段（最终产物）
├── expression_0.h                                      # 表达式定义头文件
├── libTENSOR_softmax_kernel_npu_210193433060527433398_control.so  # AICPU 控制流共享库
├── libTENSOR_softmax_kernel_npu_210193433060527433398_control.json  # 共享库元数据
└── TENSOR_softmax_kernel_npu_210193433060527433398/   # 函数特定目录
    └── aicpu/
        ├── control_flow_kernel.cpp                     # AICPU 控制流内核源文件
        ├── control_flow_kernel.o                       # AICPU 控制流内核目标文件
        ├── controlFlow_devTENSOR_softmax_kernel_npu_210193433060527433398.h  # Device 端控制流头文件
        ├── controlFlow_devTENSOR_softmax_kernel_npu_210193433060527433398_raw.h  # 原始头文件
        ├── controlFlow_devTENSOR_softmax_kernel_npu_210193433060527433398_expect.h  # 期望输出头文件
        └── expression_0.h                              # 表达式定义头文件（函数特定）
```

## 文件详细说明

### 1. Host 端控制流文件

#### controlFlow_host_*.cpp
- **作用**：Host 端控制流 C++ 源文件
- **生成位置**：[`framework/src/machine/host/backend.cpp`](../../../../framework/src/machine/host/backend.cpp)
- **编译流程**：C++ → 汇编 → 目标文件 → 二进制段
- **用途**：在 Host 端（x86_64）执行控制流逻辑

#### controlFlow_host_*.cpp.s
- **作用**：Host 端汇编文件
- **生成命令**：`g++ -S controlFlow_host_*.cpp -o controlFlow_host_*.cpp.s`
- **用途**：中间产物，用于调试和分析

#### controlFlow_host_*.cpp.o
- **作用**：Host 端目标文件
- **生成命令**：`g++ -c controlFlow_host_*.cpp.s -o controlFlow_host_*.cpp.o`
- **用途**：中间产物，用于链接

#### controlFlow_host_*.cpp.bin
- **作用**：Host 端二进制段（最终产物）
- **生成命令**：`objcopy --dump-section ast2=controlFlow_host_*.cpp.bin controlFlow_host_*.cpp.o`
- **用途**：运行时加载的控制流二进制代码

### 2. Device 端控制流文件

#### controlFlow_dev_*.cpp
- **作用**：Device 端控制流 C++ 源文件
- **生成位置**：[`framework/src/machine/host/backend.cpp`](../../../../framework/src/machine/host/backend.cpp)
- **编译流程**：C++ → 汇编 → 目标文件 → 二进制段
- **用途**：在 Device 端（ARM64/NPU）执行控制流逻辑

#### controlFlow_dev_*.cpp.s
- **作用**：Device 端汇编文件
- **生成命令**：`g++ -S controlFlow_dev_*.cpp -o controlFlow_dev_*.cpp.s`
- **用途**：中间产物，用于调试和分析

#### controlFlow_dev_*.cpp.o
- **作用**：Device 端目标文件
- **生成命令**：`g++ -c controlFlow_dev_*.cpp.s -o controlFlow_dev_*.cpp.o`
- **用途**：中间产物，用于链接

#### controlFlow_dev_*.cpp.bin
- **作用**：Device 端二进制段（最终产物）
- **生成命令**：`objcopy --dump-section ast2=controlFlow_dev_*.cpp.bin controlFlow_dev_*.cpp.o`
- **用途**：运行时加载的控制流二进制代码

### 3. 表达式定义文件

#### expression_0.h
- **作用**：表达式定义头文件，包含符号表达式的宏定义
- **生成位置**：[`framework/src/machine/host/backend.cpp`](../../../../framework/src/machine/host/backend.cpp)
- **内容**：包含 `EXPR_DEV_ROOT_COA_0_*` 等宏定义
- **用途**：在控制流代码中引用符号表达式

### 4. AICPU 控制流共享库

#### libTENSOR_softmax_kernel_npu_*_control.so
- **作用**：AICPU 控制流共享库
- **生成位置**：通过链接 `control_flow_kernel.o` 生成
- **用途**：运行时加载到 NPU 设备，供 AICPU 调用

#### libTENSOR_softmax_kernel_npu_*_control.json
- **作用**：共享库元数据文件
- **内容**：包含共享库的元信息
- **用途**：运行时加载和验证

### 5. 函数特定目录

#### TENSOR_softmax_kernel_npu_*/aicpu/
包含特定函数的 AICPU 控制流文件：

- **control_flow_kernel.cpp**：AICPU 控制流内核源文件
  - 包含 `ControlFlowEntry` 函数的实现
  - 由 `controlFlow_dev*.h` 编译生成

- **controlFlow_dev*.h**：Device 端控制流头文件
  - 包含 `ControlFlowEntry` 函数定义
  - 包含 `SetExprSubFunc` 辅助函数
  - 详细说明见 [PyPTO 控制流编译与日志分析](../00-overview.md)

- **controlFlow_dev*_raw.h**：原始头文件
  - 未经过优化的原始版本

- **controlFlow_dev*_expect.h**：期望输出头文件
  - 用于验证生成的代码格式

- **expression_0.h**：函数特定的表达式定义文件

## 编译流程

### Host 端编译流程

```
controlFlow_host_*.cpp
  ↓ (g++ -S)
controlFlow_host_*.cpp.s
  ↓ (g++ -c)
controlFlow_host_*.cpp.o
  ↓ (objcopy --dump-section ast2=)
controlFlow_host_*.cpp.bin ✅
```

### Device 端编译流程

```
controlFlow_dev_*.cpp
  ↓ (g++ -S)
controlFlow_dev_*.cpp.s
  ↓ (g++ -c)
controlFlow_dev_*.cpp.o
  ↓ (objcopy --dump-section ast2=)
controlFlow_dev_*.cpp.bin ✅
```

### AICPU 控制流共享库编译流程

```
control_flow_kernel.cpp
  ↓ (aarch64-g++ -c)
control_flow_kernel.o
  ↓ (aarch64-g++ -shared)
libTENSOR_softmax_kernel_npu_*_control.so ✅
```

## 相关文档

- [PyPTO 控制流编译与日志分析](../00-overview.md)
- [kernel_aicore 目录说明](./kernel-aicore.md)
- [输出目录与产物总览](./README.md)

