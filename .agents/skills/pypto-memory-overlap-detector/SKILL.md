---
name: pypto-memory-overlap-detector
description: 定位 PyPTO MACHINE 内存处理导致的精度问题，包括 workspace 大小、内存管理策略、内存重叠等。Triggers: machine内存问题、内存重叠。
license: 完整条款见 LICENSE.txt
---

# PyPTO MACHINE 内存处理精度问题排查技能

此技能用于定位 PyPTO 测试中的 MACHINE 内存精度问题，通过系统化的排查流程，找出导致精度异常的内存相关问题。

## 工作流程概述

1. 收集必要信息（必须执行）
2. workspace 问题排查
3. 内存重叠检测准备
4. 执行内存重叠检测
5. 清理和恢复
6. 总结：内存重叠检测

---

## 步骤 1：收集必要信息（必须执行）

**⚠️ 重要：第一步必须使用 `question` 工具向用户收集信息，严禁猜测或使用默认值。**

使用 `question` 工具收集以下信息：

- **pypto_path**: pypto 项目的根目录路径（绝对路径）
- **device_log**: 落盘日志根目录（如 ./my_log，绝对路径）
- **run_path**: 运行测试命令的目录路径（绝对路径）
- **test_cmd**: 触发精度问题的完整测试命令

将收集的路径全部转换成绝对路径，收集到所有信息后才能继续后续步骤。

---

**⚠️ 重要提示**: 将bash运行命令超时时间设置为1800000ms

## 步骤 2：workspace 问题排查

**⚠️ 重要：如果用户触发词中包含内存重叠，则跳过 步骤 2，直接开始 步骤 3 **


### 2.1 扩大 workspace 大小

**操作步骤**：

#### 2.1.1 修改代码

修改文件：`python/pypto/frontend/parser/entry.py`

```python
# 原始代码
workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)

# 修改为：扩大 10 倍
workspace_tensor = torch.empty(workspace_size * 10, dtype=torch.uint8, device=device)
```

#### 2.1.2 重新编译并安装

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

#### 2.1.3 运行测试验证

```bash
cd run_path && test_cmd
cd -
```

**判断标准**：
- 如果问题不复现 → workspace 大小计算问题，需进一步分析计算逻辑
- 如果问题仍然存在 → 继续下一步排查

#### 2.1.4 恢复原代码

测试完成后，恢复原代码：

```python
workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
```

---

### 2.2 切换内存管理策略

**问题现象**：精度异常，怀疑 torch 管理的 workspace 存在内存踩踏

**适用场景**：
- workspace 从 torch 管理，怀疑存在内存踩踏
- 需要验证是否为内存管理策略问题

**操作步骤**：

#### 2.2.1 修改代码

修改文件：`framework/src/machine/runtime/device_launcher.h`

定位到 `PrepareDevProgArgs` 函数中的 workspace 分配逻辑：

```cpp
// 原始代码
if (config.workspaceAddr) {
    kArgs.workspace = (int64_t *)config.workspaceAddr;
} else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
    kArgs.workspace = (int64_t *)devMem.AllocDev(...);
}
```

修改为：强制使用内部自管理

```cpp
// 修改后
if (0) {  // 禁用 torch 管理
    kArgs.workspace = (int64_t *)config.workspaceAddr;
} else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
    kArgs.workspace = (int64_t *)devMem.AllocDev(...);
}
```

#### 2.2.2 重新编译并安装

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

#### 2.2.3 运行测试验证

```bash
cd run_path && test_cmd
cd -
```

**判断标准**：
- 如果问题不复现 → workspace 使用问题，存在内存踩踏等
- 如果问题仍然存在 → 继续下一步排查

#### 2.2.4 恢复原代码

测试完成后，恢复原代码：

```cpp
// 恢复为
if (config.workspaceAddr) {
    kArgs.workspace = (int64_t *)config.workspaceAddr;
} else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
    kArgs.workspace = (int64_t *)devMem.AllocDev(...);
}
```

---

## 步骤 3：内存重叠检测准备

**操作步骤**：

### 3.1 打开 Operation 信息 Dump 开关

修改 `framework/src/machine/utils/device_switch.h`：

- 设置 `#define ENABLE_DUMP_OPERATION` 为 `1`

### 3.2 重新编译 PyPTO whl 包并安装

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

### 3.3 使用脚本添加 debug_options

使用脚本自动向测试文件添加 `debug_options`：

```bash
python3 .agents/skills/pypto-memory-overlap-detector/scripts/add_debug_options.py add <test_file_path>
```

**脚本参数说明**:
- `add`: 添加 debug_options
- `test_file_path`: 测试用例文件路径（如 test_operator.py）

### 3.4 清理日志并运行测试

```bash
rm -rf device_log/* && cd run_path && export ASCEND_GLOBAL_LOG_LEVEL=1 && export ASCEND_PROCESS_LOG_PATH=device_log && test_cmd
cd -
```

---

## 步骤 4：执行内存重叠检测

### 4.1 使用脚本获取检测参数

运行脚本自动获取 `-d` 和 `-t` 参数：

```bash
python3 .agents/skills/pypto-memory-overlap-detector/scripts/get_memory_check_paths.py device_log run_path/output
```

**脚本参数说明**:
- `device_log`: 落盘日志根目录（如 ./my_log）
- `run_path/output`: 运行目录中的 output **根目录**（**注意**：不要传递具体的 output_xxx 子目录，脚本会自动查找最新的子目录）

**示例**：
```bash
# 正确 ✅
python3 .agents/skills/.../get_memory_check_paths.py /path/to/wk /path/to/output

# 错误 ❌（不要传递具体的 output_xxx 子目录）
python3 .agents/skills/.../get_memory_check_paths.py /path/to/wk /path/to/output/output_20260401_165509
```

### 4.2 执行内存重叠检测

使用脚本输出的命令执行检测：

```bash
python3 tools/schema/schema_memory_check.py -d <device_log_dir> -t <dyn_topo_file>
```

### 4.3 结果解读

**检测结果显示**：

1. **无异常**：提示 "device task 无内存重叠"
2. **存在异常**：提示内存重叠的 device task 以及 leaf function

**错误信息说明**：

| 错误信息 | 含义 | 说明 |
|---------|------|------|
| `memory reuse must happen for full match` | 两个需要内存复用的 rawtensor 范围不一致 | 非内存重叠问题 |
| `memory reuse must happen for same dimension` | 两个内存复用的 rawtensor 的 shape 不一致 | 非内存重叠问题 |

**⚠️ 注意**：上述两种情况非内存重叠，脚本会断言并提示日志信息错误。

---

## 步骤 5：清理和恢复

**⚠️ 重要提示**：这是清理步骤，必须在所有排查和检测完成后执行，恢复测试环境。

### 5.1 恢复测试用例文件

使用脚本恢复被修改的测试用例文件并删除备份：

```bash
python3 .agents/skills/pypto-memory-overlap-detector/scripts/add_debug_options.py restore <test_file_path>
```

**脚本参数说明**:
- `restore`: 从备份恢复并删除备份
- `test_file_path`: 测试用例文件路径（如 test_operator.py）

**输出**:
```
已从备份恢复: /path/to/test_operator.py
已删除备份: /path/to/test_operator.py.backup
恢复成功！已删除备份文件
```

### 5.2 确认恢复完成并重新编译

**操作步骤**：

#### 5.2.1 恢复 PyPTO 源代码

修改 `framework/src/machine/utils/device_switch.h`：

- 设置 `#define ENABLE_DUMP_OPERATION` 为 `0`

#### 5.2.2 检查所有修改是否已恢复

**检查清单**:
- [ ] PyPTO 源代码 `device_switch.h` - `ENABLE_DUMP_OPERATION` 已恢复为 0
- [ ] 测试用例文件已恢复（`debug_options` 已移除）
- [ ] 备份文件已删除

#### 5.2.3 重新编译安装 PyPTO

由于源代码已恢复，需要重新编译安装：

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

#### 5.2.4 最终确认

```bash
# 检查备份文件是否已删除
ls -la run_path/*.backup 2>/dev/null || echo "所有备份文件已清理"
```

---

## 步骤 6：总结 - 内存重叠检测结果

**⚠️ 重要**：这是最后一步，在完成所有清理和恢复工作后，对检测结果进行总结。

### 检测结果
- 是否存在内存重叠：[是/否]
- 重叠的 device task：[task ID]
- 重叠类型：[RACE_READ_WRITE / RACE_WRITE_WRITE / RACE_READ_READ]
- 冲突的 leaf task 对：[src_task_id → dst_task_id]

### 结论
- [✅ 定位到内存重叠问题 / ❌ 无内存重叠，需要继续排查其他方向]
- [建议的修复方案或下一步排查方向]
---

## 关键注意事项

1. **信息收集**: 第一步必须通过 `question` 工具收集信息，严禁猜测
2. **路径规范**: 
   - 所有路径必须使用绝对路径
   - 步骤 4.1 传递 output **根目录**，不要传递具体的 output_xxx 子目录
3. **中断恢复**: 如果排查过程中断，务必手动执行步骤 5 清理环境
