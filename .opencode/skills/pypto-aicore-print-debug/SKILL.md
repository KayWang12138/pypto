---
name: pypto-aicore-print-debug
description: Pypto AICORE_PRINT 自动化调试技能。用于在测试案例出现 aic-error 时，自动化配置 AICORE_PRINT 环境，指导用户在 CCE 文件中添加打印语句，并自动分析日志以定位问题。
tag: [Pypto, 调试, AICORE_PRINT]
---

# Pypto AICORE_PRINT 自动化调试技能

本技能提供完整的 AICORE_PRINT 调试工作流，帮助用户快速定位和解决 aic-error 问题。

## 核心原则

1. **自动化配置**：自动修改配置文件和宏定义，减少手动操作
2. **渐进式调试**：从简单打印开始，逐步增加调试信息
3. **日志自动分析**：自动搜索和提取关键日志信息
4. **安全恢复**：提供恢复原始配置的方法

## 使用方法

### 快速开始

```bash
# 进入 skill 脚本目录
cd .opencode/skills/pypto-aicore-print-debug/scripts

# 查看帮助信息
python3 aicore_print_debug.py
```

### 完整调试流程

#### 步骤 1：配置 AICORE_PRINT 环境

```bash
python3 aicore_print_debug.py setup
```

此命令会`：
- 备份原始配置文件
- 修改 `tile_fwk_config.json`
- 打开 `ENABLE_AICORE_PRINT` 宏定义
- 验证修改结果

#### 步骤 2：重新编译和安装 Pypto

```bash
python3 aicore_print_debug.py rebuild
```

此命令会：
- 清理编译产物
- 重新编译 wheel 包
- 安装 wheel 包

#### 步骤 3：运行测试用例生成 CCE 文件

```bash
python3 custom/your_operator/test_case.py
```

#### 步骤 4：查找 CCE 文件

```bash
python3 aicore_print_debug.py find_cce
```

此命令会：
- 搜索所有 CCE 文件
- 显示 CCE 文件列表
- 提供修改提示

#### 步骤 5：修改 CCE 文件添加打印语句

在 CCE 文件中添加头文件和打印语句：

```cpp
#include "tilefwk/aicore_print.h"

// 在关键位置添加打印
AiCoreLogF(param->ctx, "variable = %lu\n", variable);
AiCoreLogF(param->ctx, "address: 0x%lx\n", tensor.GetAddr());

// 打印数据
for (uint32_t i = 0; i < 5; i++) {
    AiCoreLogF(param->ctx, "data: %f\n", ((float*)addr)[i]);
}
```

#### 步骤 6：配置日志环境

```bash
python3 aicore_print_debug.py setup_log
```

此命令会：
- 设置 `ASCEND_WORK_PATH` 环境变量
- 设置 `ASCEND_GLOBAL_LOG_LEVEL=0`
- 创建日志目录

#### 步骤 7：重新执行测试用例

```bash
python3 custom/your_operator/test_case.py
```

#### 步骤 8：分析日志

```bash
python3 aicore_print_debug.py analyze
```

此命令会：
- 查找 device 日志文件
- 搜索 `DumpAicoreLog` 日志
- 提取关键信息

#### 步骤 9：恢复原始配置

调试完成后，恢复原始配置：

```bash
python3 aicore_print_debug.py restore
```

此命令会：
- 恢复原始配置文件
- 恢复原始宏定义
- 验证恢复结果

### 一键执行

```bash
# 配置环境并重新编译
python3 aicore_print_debug.py all
```

### 验证工具

```bash
# 验证工具完整性
python3 aicore_print_debug.py verify
```

## 常见打印场景

### 打印变量值

```cpp
AiCoreLogF(param->ctx, "dim_0 = %lu\n", dim_0);
AiCoreLogF(param->ctx, "loop_count = %d\n", loop_count);
```

### 打印地址

```cpp
AiCoreLogF(param->ctx, "ub addr: 0x%lx\n", ubTensor.GetAddr());
AiCoreLogF(param->ctx, "gm addr: 0x%lx\n", gmTensor.GetAddr());
```

### 打印数据

```cpp
// 打印 float 数据
for (uint32_t i = 0; i < 5; i++) {
    AiCoreLogF(param->ctx, "data: %f\n", ((float*)addr)[i]);
}

// 打印十六进制
for (uint32_t i = 0; i < 5; i++) {
    AiCoreLogF(param->ctx, "data: 0x%x\n", ((uint32_t*)addr)[i]);
}
```

## 使用限制

1. **AiCoreLogF 只支持打印一个参数**
2. **只支持 int/uint/float 数据类型**
3. **默认打印空间为 16K**

## 修改打印空间大小

当需要打印大量数据时，修改 `aicpu_common.h`：

```bash
# 修改打印空间大小为 64K
sed -i 's/const uint64_t PRINT_BUFFER_SIZE = .*/const uint64_t PRINT_BUFFER_SIZE = 65536;/' framework/src/interface/machine/device/tilefwk/aicpu_common.h
```

## 完整示例

```bash
# 1. 进入脚本目录
cd .opencode/skills/pypto-aicore-print-debug/scripts

# 2. 配置环境并重新编译
python3 aicore_print_debug.py all

# 3. 运行测试用例生成 CCE 文件
cd ../../..
python3 custom/your_operator/test_case.py

# 4. 查找并修改 CCE 文件
cd .opencode/skills/pypto-aicore-print-debug/scripts
python3 aicore_print_debug.py find_cce
# 手动编辑 CCE 文件添加打印语句

# 5. 配置日志环境
python3 aicore_print_debug.py setup_log

# 6. 重新执行测试用例
cd ../../..
python3 custom/your_operator/test_case.py

# 7. 分析日志
cd .opencode/skills/pypto-aicore-print-debug/scripts
python3 aicore_print_debug.py analyze

# 8. 恢复原始配置
python3 aicore_print_debug.py restore
```

## 检查清单

使用 A`ICORE_PRINT 调试时，确保：

- [ ] 已配置 AICORE_PRINT 环境
- [ ] 已重新编译和安装
- [ ] 已运行测试用例生成 CCE 文件
- [ ] 已在 CCE 文件中添加打印语句
- [ ] 已配置日志环境变量
- [ ] 已重新执行测试用例
- [ ] 已搜索和分析日志
- [ ] 调试完成后已恢复原始配置

## 参考资料

- AICORE_PRINT 使用文档：`AICORE_PRINT.md`
- 官方示例：`examples/01_beginner/basic/add_direct.py`
- 脚本文档：`scripts/README.md`
