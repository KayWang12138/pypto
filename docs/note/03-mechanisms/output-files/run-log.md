# run.log 文件详细说明

## 文件概述

`run.log` 是 PyPTO 编译和运行过程中生成的详细日志文件，记录了整个编译流程、运行时状态、内存管理、函数调用等详细信息。

**文件位置（示例）**：`<work_dir>/output/output_<timestamp>_<pid>/run.log`

**文件大小（示例）**：与算子/配置/日志级别相关，请以实际产物为准

## 日志格式

日志采用标准格式：
```
时间戳 日志级别 | 日志内容
```

**时间戳格式**：`YYYY-MM-DD HH:MM:SS.mmm`

**日志级别**：
- `D`：DEBUG - 调试信息
- `I`：INFO - 一般信息
- `W`：WARN - 警告信息
- `E`：ERROR - 错误信息

## 主要日志类型

### 1. SlotManager 日志

**格式**：
```
[slotManager] 操作序号 op:操作类型 Id:张量ID slot:槽地址(张量名) value:值地址(形状信息)
```

**操作类型（常见类型，非全量）**：
- `read`：读取张量
- `write`：写入张量
- `construct`：构造张量
- `destruct`：销毁张量
- `input`：输入张量
- `checkpoint`：检查点
- `restore`：恢复

**注意：** 操作类型列表可能因版本不同而变化，请以实际 `run.log` 输出为准。

**示例**：
```
2025-12-30 16:50:04.414 D | [slotManager] 1 op:read       Id:0 slot:0xaaab4118cd20(TENSOR_1)      value:0xaaab402fb480(<-1 x 32 x 1 x 256 x DT_FP32 / RUNTIME_GetInputShapeDim(ARG_TENSOR_1,0) x 32 x 1 x 256 x DT_FP32> %0@0#(-1)MEM_UNKNOWN::MEM_UNKNOWN)
```

**说明**：
- `Id:0`：张量在函数中的索引
- `slot:0xaaab4118cd20(TENSOR_1)`：内存槽地址和名称
- `value:0xaaab402fb480(...)`：张量值地址和形状信息
- 形状格式：`<动态维度 x 固定维度 x ... x 数据类型>`（可能因版本不同而变化）
- `RUNTIME_GetInputShapeDim(ARG_TENSOR_1,0)`：运行时动态形状表达式

**注意：** 示例中的地址、符号、形状格式等可能因版本/平台不同而变化，请关注字段含义而非具体数值。

### 2. 函数处理日志

**格式**：
```
func.end.finish: name=函数名
Hash for function 函数ID 函数名 is ... hash value is 哈希值
```

**示例**：
```
2025-12-30 16:50:04.420 D | func.end.finish: name=TENSOR_TENSOR_softmax_kernel_npu_loop_Unroll1_PATH0_hiddenfunc0_5
2025-12-30 16:50:04.421 D | Hash for function 5 TENSOR_TENSOR_softmax_kernel_npu_loop_Unroll1_PATH0_hiddenfunc0_5 is 4 0 TENSOR_TENSOR_softmax_kernel_npu_loop_Unroll1_PATH0_hiddenfunc0_5  hash value is 13208989832139651355
```

**说明**：
- 记录函数编译完成和哈希值计算
- 哈希值用于函数去重和缓存

### 3. Cast 操作日志

**格式**：
```
originOut cast name 输出索引 原始索引
raw out cast number 数量
same raw out cast number 数量
```

**示例**：
```
2025-12-30 16:50:04.588 I | originOut cast name 0 2
2025-12-30 16:50:04.588 I | originOut cast name 5 7
2025-12-30 16:50:04.588 I | originOut cast name 1 1
2025-12-30 16:50:04.588 I | raw out cast number 3
2025-12-30 16:50:04.588 I | same raw out cast number 1
```

**说明**：
- 记录输出 Cast 操作的映射关系
- 用于优化和去重

### 4. 编译命令日志

**格式**：
```
[RunCmd] 编译命令
PreCompileCmd is 预编译命令
```

**示例**：
```
2025-12-30 15:45:01.081 I | [RunCmd] LD_PRELOAD= g++ -fPIC -O2 ... -S output/.../controlFlow_host_10193433060527433398.cpp -o output/.../controlFlow_host_10193433060527433398.cpp.s
2025-12-30 15:45:01.307 D | PreCompileCmd is /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ -Wall -O2 -fPIC -c ...
```

**说明**：
- `[RunCmd]`：Host 端编译命令（INFO 级别）
- `PreCompileCmd`：Device 端预编译命令（DEBUG 级别）

## 关键信息提取

### 1. 动态形状信息

日志中大量出现动态形状表达式：
- `RUNTIME_GetInputShapeDim(ARG_TENSOR_1,0)`：获取输入张量的动态维度
- `RUNTIME_GetViewValidShapeDim(...)`：获取 View 操作的有效形状
- `loop_idx_0`：循环变量

### 2. 内存管理

通过 SlotManager 日志可以追踪：
- 张量的生命周期（construct → read/write → destruct）
- 内存地址分配
- 张量形状变化

### 3. 函数调用链

通过函数名可以追踪调用链：
- `TENSOR_softmax_kernel_npu`：主函数
- `TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0`：循环展开后的隐藏函数
- `TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8`：子函数

## 日志分析技巧

1. **按时间戳排序**：了解编译和运行的时序
2. **过滤关键操作**：使用 `grep` 过滤特定操作类型
3. **追踪张量生命周期**：通过 `Id` 和 `slot` 地址追踪
4. **分析函数调用**：通过函数名和哈希值分析调用关系

## 常见问题排查

### 1. 内存泄漏

查找未配对的 `construct` 和 `destruct`：
```bash
grep "op:construct\|op:destruct" run.log | sort
```

### 2. 形状不匹配

查找形状相关的错误：
```bash
grep -i "shape\|dim" run.log | grep -i "error\|warn"
```

### 3. 编译错误

查找编译相关的错误：
```bash
grep -i "error\|fail" run.log
```

## 相关文档

- [PyPTO 控制流编译与日志分析](../00-overview.md)
- [kernel_aicpu 目录说明](./kernel-aicpu.md)
- [topo.json 文件详细说明](./topo-json.md)
- [program.json 文件详细说明](./program-json.md)
- [输出目录与产物总览](./README.md)

