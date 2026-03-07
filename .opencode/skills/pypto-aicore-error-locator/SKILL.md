---
name: aicore-error-locator
description: 定位测试案例中出现 aicore error 时的问题 CCE 文件。当需要分析 aicore 错误并找到导致错误的 CCE 文件时使用此技能。
license: 完整条款见 LICENSE.txt
---

# AICore Error 定位器

此技能帮助定位测试案例中出现 aicore error 时的问题 CCE 文件。

## 工作流程

### 1. 启用追踪日志

修改以下配置以启用详细的追踪日志：

- **配置文件**: 修改 `tile_fwk_config.json`
  - 设置 `"fixed_output_path"` 为 `true`
  - 设置 `"force_overwrite"` 为 `false`

- **头文件**: 修改 `aicore_entry.h`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

- **工具头文件**: 修改 `device_switch.h`
  - 设置 `#define ENABLE_COMPILE_VERBOSE_LOG` 为 `1`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

### 2. 重新编译和安装

重新编译 pypto 包并安装。

### 3. 配置日志路径

设置环境变量：

- 设置 device log 落盘路径（例如：`export ASCEND_WORK_PATH=<your-path>`）
- 设置日志级别（例如：`export ASCEND_GLOBAL_LOG_LEVEL=0`）

### 4. 清理日志

清理日志目录和运行目录中的 kernel_aic* 文件：

- 清理 device log 落盘路径下的所有日志信息
- 清理运行目录下的 `kernel_aic*` 文件

**重要**: 如果切换运行目录，需要重新清理 `kernel_aic*` 文件。

### 5. 运行测试

执行测试命令，例如：
```bash
python test.py 4
```

### 6. 分析追踪日志

在 device log 落盘路径中搜索 trace 日志：

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
3. 找出 LActFinish 中缺失的 Uid
4. 在 LActStart 日志中找到缺失 Uid 对应的完整 LEvent 信息

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
- CCE 文件名中包含对应 Uid 的标识符
