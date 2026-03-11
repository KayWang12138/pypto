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

根据用户提供的 pypto 目录路径，修改以下配置以启用详细的追踪日志：

- **配置文件**: 搜索修改 `tile_fwk_config.json`
  - 设置 `"fixed_output_path"` 为 `true`
  - 设置 `"force_overwrite"` 为 `false`

- **头文件**: 搜索修改 `aicore_entry.h`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

- **工具头文件**: 搜索修改 `device_switch.h`
  - 设置 `#define ENABLE_COMPILE_VERBOSE_LOG` 为 `1`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

### 3. 重新编译和安装

进入用户提供的 pypto 目录路径，重新编译 pypto 包并pip安装。
**编译命令**: `cd <用户提供的 pypto 目录路径> && python3 build_ci.py -f python3 --disable_auto_execute`
**pip安装命令**: `pip install <用户提供的 pypto 目录路径>/build_out/pypto*.whl --force --no-deps`

### 4. 清理日志

清理日志目录和运行目录中的 kernel_aic* 文件夹：

- 清理 device log 落盘路径下的所有日志信息
- 在用户提供的运行目录下清理 `kernel_aic*` 文件夹

**重要**: 每次运行前必须清理，避免日志混淆。

### 5. 运行测试

在用户提供的运行目录中执行用户提供的测试命令。

**配置日志路径和设置日志级别**：
- 设置 device log 落盘路径：`export ASCEND_PROCESS_LOG_PATH=<用户提供的路径>`
- 若路径不存在则创建该目录
- 设置日志级别：`export ASCEND_GLOBAL_LOG_LEVEL=0`

**重要**: 一定要进入**运行目录**，同时**配置日志路径和设置日志级别**，再执行测试命令。
**重要**: 运行测试的打屏日志中必须出现aicore error，如果未出现，则不适用于该SKILL，请停止运行

### 6. 分析追踪日志

在 device log 落盘路径中搜索 "trace"和"LActStart"、"trace"和"LActFinish" 关键字：
**重要**: <log-file>在 device log 落盘路径中的debug文件夹下，且命名包含device关键字，同时后缀为log
**搜索 LActStart 事件**:
```bash
grep -rn "trace" <log-file> | grep "LActStart"
```

**搜索 LActFinish 事件**:
```bash
grep -rn "trace" <log-file> | grep "LActFinish"
```

### 7. 定位问题 CCE 文件
**解析 LEvent 事件 格式**:
LActStart 事件 格式为 `LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex),LActStart(coreIdx))`
LActFinish 事件 格式为 `LEvent(LUid(deviceTaskId, funcId, rootIndex, opIdx, leafIndex),LActFinish(coreIdx))`

**对比日志**:
1. 从 LActStart 事件日志中提取所有的 coreIdx，将所有的coreIdx从小到大排序，若不存在LActStart 事件，则为空
2. 从 LActFinish 事件日志中提取所有的 coreIdx，，将所有的coreIdx从小到大排序，若不存在LActFinish 事件，则为空
3. 对比两者的coreIdx，找出缺失的全部coreIdx（缺失条件为coreIdx在LActStart事件中存在且在LActFinish事件中不存在），从coreIdx对应的LActFinish 事件提取出`<leafIndex>`（LUid中的最后一个值）
**重要**: 理论上，所有缺失的coreIdx对应的是同一个`<leafIndex>`


**查找 CCE 文件**:
**CCE 文件名称格式**: <CCE_pre_name>_<CCE_hash>_<CCE_ID>_<core_type>.cpp
**Record 文件名称格式** sub_func_<core_type>_call_<ID>.h
1. 在 `kernel_aicore` 目录下查找 Record 文件，在多个Record 文件中搜索`case <leafIndex>`，从而得到Record 文件名称，通过Record 文件名称得到<core_type>
2. 阅读存在`case <leafIndex>`的Record 文件，通过`case <leafIndex>`找到对应的函数名，函数名格式为<CCE_pre_name>_<CCE_ID>_<func_hash>，通过函数名得到<CCE_pre_name>_<CCE_ID>
3. 通过<CCE_pre_name>、<CCE_ID>、<core_type>得到CCE 文件名的关键信息，在`kernel_aicore` 目录下查找文件名称，若存在，则查找完成

### 8. 找到问题代码
**方案一**: 阅读报错日志和CCE文件代码，找到问题代码

**方案二**: 
1. 修改CCE文件，将CCE文件中的set_flag和wait_flag全部注释，再次运行，观察是否仍能复现问题，若不能复现问题，此方案不生效
2. 在set_flag和wait_flag全部注释的基础上，从后往前注释CCE文件，循环运行，直到不报错为止，从而找到问题代码
3. 将上述步骤中对CCE文件的所有修改全部恢复

### 9. 输出结果

输出找到的 CPP 文件路径和问题代码。

## 关键点

- 确保 `fixed` 模式启用以保持输出路径不变
- 每次运行前清理日志以避免混淆
- 执行每条命令时，务必确保在用户提供的运行目录下执行
