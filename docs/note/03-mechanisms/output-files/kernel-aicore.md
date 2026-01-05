# kernel_aicore 目录详细说明

## 目录概述

`kernel_aicore` 目录包含 AICore（AI Core）相关的编译产物，包括 AICore 内核代码、链接脚本、目标文件等。

**目录位置**：输出目录下的 `kernel_aicore/`（例如：`output/output_<timestamp>_<pid>/kernel_aicore/`）

## 目录结构

```
kernel_aicore/
├── aicore.cpp                                          # AICore 主源文件
├── dy_kernel_10193433060527433398_0.o                 # 动态内核目标文件（最终产物）
├── dy_kernel_10193433060527433398_aic_0.o             # AIC 内核目标文件
├── dy_kernel_10193433060527433398_aiv_0.o             # AIV 内核目标文件
├── mid_kernel_10193433060527433398_aic_0.o            # AIC 中间内核目标文件
├── mid_kernel_10193433060527433398_aiv_0.o            # AIV 中间内核目标文件
├── link_aic_634453.sh                                 # AIC 链接脚本
├── link_aiv_634453.sh                                 # AIV 链接脚本
├── link_mix_634453.sh                                 # MIX 链接脚本
├── sub_func_aiv_call_0.h                              # AIV 子函数调用头文件
└── TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8_3528554642701557793_0_aiv.cpp  # AIV 内核源文件
└── TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8_3528554642701557793_0_aiv.o   # AIV 内核目标文件
```

## 文件详细说明

### 1. AICore 主源文件

#### aicore.cpp
- **作用**：AICore 主源文件，包含 AICore 运行时和内核入口
- **内容**：
  - AICore 运行时函数定义
  - 内核入口宏定义
  - 性能追踪结构定义
  - 任务统计结构定义
- **用途**：作为 AICore 内核的基础框架

### 2. 内核目标文件

#### dy_kernel_*_0.o
- **作用**：动态内核目标文件（最终产物）
- **生成方式**：通过链接 `dy_kernel_*_aic_0.o` 和 `dy_kernel_*_aiv_0.o` 生成
- **链接脚本**：`link_mix_634453.sh`
- **用途**：运行时加载到 NPU 设备执行

#### dy_kernel_*_aic_0.o
- **作用**：AIC（AI Cube）内核目标文件
- **生成方式**：通过链接 `mid_kernel_*_aic_0.o` 生成
- **链接脚本**：`link_aic_634453.sh`
- **用途**：包含 Cube 操作的内核代码

#### dy_kernel_*_aiv_0.o
- **作用**：AIV（AI Vector）内核目标文件
- **生成方式**：通过链接函数特定的 AIV 目标文件和 `mid_kernel_*_aiv_0.o` 生成
- **链接脚本**：`link_aiv_634453.sh`
- **用途**：包含 Vector 操作的内核代码

#### mid_kernel_*_aic_0.o / mid_kernel_*_aiv_0.o
- **作用**：中间内核目标文件
- **用途**：链接过程中的中间产物

### 3. 链接脚本

#### link_aic_634453.sh
- **作用**：AIC 内核链接脚本
- **链接器**：`ld.lld`（LLVM 链接器）
- **参数**：
  - `-m aicorelinux`：目标架构
  - `-Ttext=0`：文本段起始地址
  - `-static`：静态链接
  - `-r`：可重定位
- **输入**：`mid_kernel_*_aic_0.o`
- **输出**：`dy_kernel_*_aic_0.o`

#### link_aiv_634453.sh
- **作用**：AIV 内核链接脚本
- **链接器**：`ld.lld`
- **输入**：函数特定的 AIV 目标文件 + `mid_kernel_*_aiv_0.o`
- **输出**：`dy_kernel_*_aiv_0.o`

#### link_mix_634453.sh
- **作用**：MIX（混合）内核链接脚本
- **链接器**：`ld.lld`
- **输入**：`dy_kernel_*_aic_0.o` + `dy_kernel_*_aiv_0.o`
- **输出**：`dy_kernel_*_0.o`（最终产物）

### 4. 函数特定文件

#### TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8_*_aiv.cpp
- **作用**：特定函数的 AIV 内核源文件
- **命名规则**：`函数名_哈希值_索引_aiv.cpp`
- **内容**：包含该函数的 AIV 内核实现
- **用途**：编译生成对应的 `.o` 文件

#### sub_func_aiv_call_0.h
- **作用**：AIV 子函数调用头文件
- **内容**：包含子函数调用的声明和定义
- **用途**：在 AIV 内核中调用子函数

## 编译流程

### AIC 内核编译流程

```
mid_kernel_*_aic_0.o
  ↓ (ld.lld via link_aic_634453.sh)
dy_kernel_*_aic_0.o
```

### AIV 内核编译流程

```
TENSOR_*_aiv.cpp
  ↓ (编译)
TENSOR_*_aiv.o
  ↓ (ld.lld via link_aiv_634453.sh, 与 mid_kernel_*_aiv_0.o 链接)
dy_kernel_*_aiv_0.o
```

### MIX 内核编译流程

```
dy_kernel_*_aic_0.o + dy_kernel_*_aiv_0.o
  ↓ (ld.lld via link_mix_634453.sh)
dy_kernel_*_0.o ✅
```

## AICore vs AICPU

| 特性 | AICore | AICPU |
|------|--------|-------|
| **用途** | 计算密集型操作（Cube/Vector） | 控制流和复杂逻辑 |
| **编译产物** | `.o` 目标文件 | `.so` 共享库 |
| **链接方式** | 静态链接 | 动态链接 |
| **执行位置** | NPU 设备 | NPU 设备（通过 AICPU 运行时） |

## 相关文档

- [PyPTO 控制流编译与日志分析](../00-overview.md)
- [kernel_aicpu 目录说明](./kernel-aicpu.md)
- [输出目录与产物总览](./README.md)

