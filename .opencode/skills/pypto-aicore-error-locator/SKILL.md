---
name: pypto-aicore-error-locator
description: 定位测试案例中出现 aicore error 时的问题 CCE 文件。当需要分析 aicore 错误并找到导致错误的 CCE 文件时使用此技能。
license: 完整条款见 LICENSE.txt
---

# AICore Error 定位器

此技能帮助定位测试案例中出现 aicore error 时的问题 CCE 文件。

## 工作流程

### 1. 收集必要信息

首先使用 `question` 工具向用户收集以下必要信息：

- **pypto目录路径**: 用户必须提供 pypto 项目的根目录路径
- **device log落盘路径**: 用户必须提供 device log 的落盘路径（若不存在则需创建）
- **运行命令**: 用户必须提供触发 aicore error 的测试命令
- **运行目录**: 用户必须提供运行测试命令的目录路径

示例问题配置：
```
question: 
  - header: "PyPTO配置"
    question: "请提供 pypto 目录的完整路径"
    options: []
  - header: "日志路径"
    question: "请提供 device log 落盘路径（不存在将自动创建）"
    options: []
  - header: "运行命令"
    question: "请提供触发 aicore error 的测试命令"
    options: []
  - header: "运行目录"
    question: "请提供运行测试命令的目录路径"
    options: []
```

### 2. 启用追踪日志

使用用户提供的 pypto 目录路径，修改以下配置以启用详细的追踪日志：

- **配置文件**: 修改 `tile_fwk_config.json`
  - 设置 `"fixed_output_path"` 为 `true`
  - 设置 `"force_overwrite"` 为 `false`

- **头文件**: 修改 `aicore_entry.h`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

- **工具头文件**: 修改 `device_switch.h`
  - 设置 `#define ENABLE_COMPILE_VERBOSE_LOG` 为 `1`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

### 2. 重新编译和安装

进入用户提供的 pypto 目录路径，重新编译 pypto 包并pip安装。

### 3. 配置日志路径

使用用户提供的 device log 落盘路径，设置环境变量：

- 设置 device log 落盘路径：`export ASCEND_WORK_PATH=<用户提供的路径>`
- 若路径不存在则创建该目录
- 设置日志级别：`export ASCEND_GLOBAL_LOG_LEVEL=0`

### 4. 清理日志

清理日志目录和运行目录中的 kernel_aic* 文件夹：

- 清理 device log 落盘路径下的所有日志信息
- 在用户提供的运行目录下清理 `kernel_aic*` 文件夹

**重要**: 每次运行前必须清理，避免日志混淆。

### 5. 运行测试

在用户提供的运行目录中执行用户提供的测试命令。

**重要**: 一定要进入运行目录，再执行测试命令。

### 6. 分析追踪日志

在 device log 落盘路径中搜索 "trace"和"LActStart"、"trace"和"LActFinish" 关键字：

**搜索 LActStart 事件**:
```bash
grep -rn "trace" <log-file> | grep "LActStart"
```

**搜索 LActFinish 事件**:
```bash
grep -rn "trace" <log-file> | grep "LActFinish"
```

### 7. 定位问题 CCE 文件

**对比日志**:
1. 从 LActStart 日志中提取所有 Uid
2. 从 LActFinish 日志中提取所有 Uid
3. 对齐两者的Uid，找出 LActFinish 中缺失的 Uid，提取出`<value2>`（括号中的最后一个值）
**重要**: 如果有多个缺失的Uid，提取出公共的`<value2>`

**解析 LEvent 格式**:
LEvent 格式为 `#LEvent{LUid{0,0,0,<value1>,<value2>},LActStart{<uid>}}`
- 提取 `<value2>`（括号中的最后一个值）

**查找 CCE 文件**:
在 `kernel_aicore` 目录中查找包含 `_<value2>_` 的 C++ 文件。

### 8. 输出结果

输出找到的 CCE 文件路径。

## 关键点

- 确保 `fixed` 模式启用以保持输出路径不变
- 每次运行前清理日志以避免混淆
- 通过对比 LActStart 和 LActFinish 事件定位失败的 Uid
- CCE 文件名中包含对应 `_<value2>_` 的标识符
