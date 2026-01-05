# PyPTO 环境变量参考手册

> **适用对象：** 开发者、运维人员、测试人员  
> **更新时间：** 2026年1月  
> **学习目标：** 了解 PyPTO 支持的所有环境变量及其用法

---

## 目录

**第一部分：当前已存在的环境变量**
1. [日志相关环境变量](#1-日志相关环境变量)
2. [编译相关环境变量](#2-编译相关环境变量)
3. [运行时环境变量](#3-运行时环境变量)
4. [测试相关环境变量](#4-测试相关环境变量)
5. [系统依赖环境变量](#5-系统依赖环境变量)

**第二部分：未来规划的环境变量**
6. [统一日志系统环境变量](#6-统一日志系统环境变量)
7. [调试与性能分析环境变量](#7-调试与性能分析环境变量)
8. [缓存管理环境变量](#8-缓存管理环境变量)

**第三部分：使用指南**
9. [环境变量使用示例](#9-环境变量使用示例)
10. [最佳实践](#10-最佳实践)

---

## 说明

本文档分为两个主要部分：
- **第一部分（第1-5章）**：当前代码中已实现并使用的环境变量
- **第二部分（第6-8章）**：未来规划的环境变量（已在设计文档中提出，但代码中尚未实现）

---

## 第一部分：当前已存在的环境变量

---

## 1. 日志相关环境变量

### 1.1 GLOBAL_LOG_LEVEL

**说明**：设置 C++ 层日志级别（旧版，建议迁移到 PYPTO_LOG_LEVEL）

| 属性 | 值 |
|-----|---|
| 类型 | 整数 |
| 默认值 | `3` (ERROR) |
| 可选值 | `0`=DEBUG, `1`=INFO, `2`=WARN, `3`=ERROR, `4`=FATAL, `5`=EVENT, `6`=NONE |
| 影响范围 | C++ `LoggerManager` |

**使用示例**：

```bash
# 启用调试日志
export GLOBAL_LOG_LEVEL=0

# 只显示错误及以上
export GLOBAL_LOG_LEVEL=3
```

**日志级别对照表**：

| 级别名称 | PYPTO_LOG_LEVEL | GLOBAL_LOG_LEVEL |
|---------|-----------------|-----------------|
| TRACE | TRACE | - |
| DEBUG | DEBUG | 0 |
| INFO | INFO | 1 |
| WARN | WARN / WARNING | 2 |
| ERROR | ERROR | 3 |
| FATAL | FATAL / CRITICAL | 4 |
| EVENT | - | 5 |
| OFF | OFF | 6 |

### 1.2 PTO_BACKTRACE

**说明**：控制 Python 异常时是否显示完整回溯

| 属性 | 值 |
|-----|---|
| 类型 | 整数 |
| 默认值 | `0` |
| 可选值 | `0`=隐藏回溯, `1`=显示回溯 |
| 影响范围 | `pypto.frontend.parser` 模块异常 |

**使用示例**：

```bash
# 显示完整回溯（便于调试）
export PTO_BACKTRACE=1

# 隐藏回溯（生产环境）
export PTO_BACKTRACE=0
```

### 1.3 ASCEND_GLOBAL_LOG_LEVEL

**说明**：设置华为 CANN 工具链的日志级别

| 属性 | 值 |
|-----|---|
| 类型 | 整数 |
| 默认值 | 由 CANN 决定 |
| 可选值 | `0`=DEBUG, `1`=INFO, `2`=WARN, `3`=ERROR |
| 影响范围 | CANN 工具链日志输出 |

**使用示例**：

```bash
# 启用 CANN 调试日志
export ASCEND_GLOBAL_LOG_LEVEL=0
```

---

## 2. 编译相关环境变量

### 2.1 TILEFWK_CONFIG_PATH

**说明**：指定框架配置文件路径

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（文件路径） |
| 默认值 | `<库路径>/configs/tile_fwk_config.json` |
| 影响范围 | 编译配置、Pass 配置、全局设置 |

**使用示例**：

```bash
# 使用自定义配置
export TILEFWK_CONFIG_PATH=/path/to/custom_config.json
```

### 2.2 ASCEND_HOME_PATH

**说明**：华为 Ascend 工具链安装路径

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（目录路径） |
| 默认值 | 无 |
| 影响范围 | AICPU 编译、设备端执行、后端库加载 |

**使用示例**：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

**注意事项**：
- 设置后会影响 `python/pypto/__init__.py` 中的后端库加载逻辑
- 如果未设置，框架会回退到 `cost_model` 后端
- 必须设置才能使用 NPU 设备执行

### 2.3 PYPTO_BUILD_EXT_ARGS

**说明**：传递 CMake 构建参数给 `setup.py` 的 `build_ext` 命令

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（命令行参数） |
| 默认值 | 无 |
| 影响范围 | 编译配置（build_type、asan、verbose 等） |

**使用示例**：

```bash
# 启用 Debug 构建
export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Debug'

# 启用 ASAN 和详细输出
export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Debug --cmake-verbose'

# 多个参数
export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Release --cmake-options="-DENABLE_ASAN=ON"'
```

**支持的参数**：
- `--cmake-build-type`: Debug/Release/RelWithDebInfo
- `--cmake-generator`: CMake 生成器类型
- `--cmake-options`: 额外的 CMake 选项
- `--cmake-verbose`: 启用详细输出

### 2.4 PYPTO_THIRD_PARTY_PATH

**说明**：指定第三方源码包路径（当无法访问 `cann-src-third-party` 时使用）

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（目录路径） |
| 默认值 | 无 |
| 影响范围 | 编译时第三方依赖查找 |

**使用示例**：

```bash
export PYPTO_THIRD_PARTY_PATH=/path/to/third-party
```

### 2.5 PTO_TILE_LIB_CODE_PATH

**说明**：指定 PTO Tile 库代码路径

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（目录路径） |
| 默认值 | 无 |
| 影响范围 | 代码生成阶段的 Tile 库路径 |

**使用示例**：

```bash
export PTO_TILE_LIB_CODE_PATH=/path/to/tile_lib
```

---

## 3. 运行时环境变量

### 3.1 TILE_FWK_DEVICE_ID

**说明**：框架设备 ID（C++ 层，运行时必需）

| 属性 | 值 |
|-----|---|
| 类型 | 整数 |
| 默认值 | 无（必须设置） |
| 影响范围 | 设备端执行、示例程序运行 |

**使用示例**：

```bash
# 使用设备 0
export TILE_FWK_DEVICE_ID=0

# 使用设备 1
export TILE_FWK_DEVICE_ID=1
```

**注意事项**：
- 所有示例程序都需要设置此变量
- 必须与 `torch.npu.set_device()` 设置的设备 ID 一致
- 未设置会导致运行时错误

### 3.2 TILE_FWK_OUTPUT_DIR

**说明**：指定框架输出目录（用于调试和对比）

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（目录路径） |
| 默认值 | `output/output_<timestamp>_<pid>` |
| 影响范围 | 所有框架输出文件（IR、日志、二进制等） |

**使用示例**：

```bash
# 固定输出目录（便于对比和回归）
export TILE_FWK_OUTPUT_DIR=/tmp/pypto_output

# 用于精度调试
export TILE_FWK_OUTPUT_DIR=/path/to/debug_output
```

**输出内容**：
- IR JSON 文件（各 Pass 前后）
- 编译日志
- 生成的二进制文件
- 运行日志

### 3.3 PYPTO_HOME

**说明**：指定 PyPTO 运行时数据目录

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（目录路径） |
| 默认值 | `$HOME/.pypto` |
| 影响范围 | 运行时数据目录（rundata） |

**使用示例**：

```bash
# 使用自定义目录
export PYPTO_HOME=/tmp/pypto_data
```

**说明**：
- 如果未设置，使用 `$HOME/.pypto` 作为默认目录
- 运行时数据存储在 `$PYPTO_HOME/run/rundata_<timestamp>/` 目录
- 最多保留 127 个历史目录，旧的会自动清理

### 3.4 AST_DATADUMP_PATH

**说明**：启用 AST 数据转储功能

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（布尔值） |
| 默认值 | 无（未启用） |
| 可选值 | `true`（启用）, 其他值（禁用） |
| 影响范围 | 设备端数据转储 |

**使用示例**：

```bash
# 启用数据转储
export AST_DATADUMP_PATH=true

# 禁用数据转储（或不设置）
unset AST_DATADUMP_PATH
```

**说明**：
- 用于设备端数据转储功能
- 设置为 `"true"`（不区分大小写）时启用
- 主要用于调试和性能分析

---

## 4. 调试环境变量

### 4.1 PYPTO_DEBUG

**说明**：启用调试模式

| 属性 | 值 |
|-----|---|
| 类型 | 整数 |
| 默认值 | `0` |
| 可选值 | `0`=关闭, `1`=启用 |
| 影响范围 | 额外的调试信息输出、断言检查 |

**使用示例**：

```bash
export PYPTO_DEBUG=1
```

### 4.2 PYPTO_DUMP_IR

**说明**：转储中间表示

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（目录路径） |
| 默认值 | 无（不转储） |
| 影响范围 | 各编译阶段的 IR 输出 |

**使用示例**：

```bash
export PYPTO_DUMP_IR=/tmp/ir_dump
```

### 4.3 PYPTO_PROFILE

**说明**：启用性能分析

| 属性 | 值 |
|-----|---|
| 类型 | 整数 |
| 默认值 | `0` |
| 可选值 | `0`=关闭, `1`=启用 |
| 影响范围 | 编译和执行时间统计 |

**使用示例**：

```bash
export PYPTO_PROFILE=1
```

---

## 5. 测试相关环境变量

### 5.1 TILE_FWK_STEST_DEVICE_ID

**说明**：系统测试（STest）使用的设备 ID

| 属性 | 值 |
|-----|---|
| 类型 | 整数 |
| 默认值 | `0` |
| 影响范围 | Python 系统测试用例 |

**使用示例**：

```bash
export TILE_FWK_STEST_DEVICE_ID=1
```

### 5.2 ENABLE_STEST_EXECUTE_DEVICE_ID

**说明**：CMake 选项，自动执行测试时指定设备 ID

| 属性 | 值 |
|-----|---|
| 类型 | CMake 选项（ON/OFF） |
| 默认值 | `OFF` |
| 影响范围 | 测试自动执行 |

**使用示例**：

```bash
# 在 CMake 配置时设置
cmake -DENABLE_STEST_EXECUTE_DEVICE_ID=ON ..
```

### 5.3 ENABLE_STEST_DUMP_JSON

**说明**：测试时是否转储 JSON 文件

| 属性 | 值 |
|-----|---|
| 类型 | CMake 选项（ON/OFF） |
| 默认值 | `OFF` |
| 影响范围 | 测试输出 |

### 5.4 ENABLE_STEST_INTERPRETER_CONFIG

**说明**：启用测试解释器配置

| 属性 | 值 |
|-----|---|
| 类型 | CMake 选项（ON/OFF） |
| 默认值 | `OFF` |
| 影响范围 | 测试执行模式 |

### 5.5 ENABLE_STEST_BINARY_CACHE

**说明**：启用测试二进制缓存（自动在增量测试时启用）

| 属性 | 值 |
|-----|---|
| 类型 | CMake 选项（ON/OFF） |
| 默认值 | `OFF` |
| 影响范围 | 测试编译缓存、设备端二进制文件缓存 |

**缓存内容**：

`ENABLE_STEST_BINARY_CACHE` 缓存以下内容：

1. **设备程序二进制文件**（`devProgBinary`）：
   - 文件路径：`$HOME/ast_data/<SOC_VERSION>/ast_op_<cacheKey>.o`
   - 内容：编译后的设备端程序二进制数据
   - 用途：避免重复编译相同的程序

2. **内核二进制文件**（`kernelBinary`）：
   - 文件路径：`$HOME/ast_data/<SOC_VERSION>/ast_op_<cacheKey>_kernel.o`
   - 内容：内核代码的二进制数据
   - 用途：复用已编译的内核代码

3. **自定义控制 SO 文件**（`controlBin`）：
   - 文件路径：`$HOME/ast_data/<SOC_VERSION>/lib<OpFuncName>_control.so`
   - 内容：自定义操作的动态库二进制数据
   - 用途：缓存自定义操作的编译结果

4. **自定义控制 JSON 文件**：
   - 文件路径：`$HOME/ast_data/<SOC_VERSION>/lib<OpFuncName>_control.json`
   - 内容：自定义操作的配置信息
   - 用途：存储自定义操作的元数据

**缓存目录**：
- 基础路径：`$HOME/ast_data/<SOC_VERSION>/`
- SOC 版本：由 `PlatformManager::Instance().GetShortSocVersion()` 获取
- 如果 `HOME` 环境变量未设置，缓存初始化会失败

**缓存键（cacheKey）**：
- 每个函数/操作都有唯一的 `cacheKey`
- 基于函数特征生成，相同函数会生成相同的 `cacheKey`
- 用于区分不同的缓存文件

**使用场景**：
- ✅ 增量测试时自动启用（`--changed_files` 参数存在时）
- ✅ CI/CD 流水线中加速测试执行
- ✅ 重复运行相同测试用例时复用编译结果

**注意事项**：
- 缓存文件存储在用户主目录，需要确保有足够的磁盘空间
- 并行测试时，相同 `OpFuncName` 的测试用例可能共享自定义 SO/JSON 文件
- 缓存文件不会自动清理，需要手动管理

**相关文档**：
- 详细说明见 [pypto_smoke.sh 脚本详解](06-pyto-smoke-script.md#1111-并行测试时的缓存安全性分析)

### 5.6 ENABLE_STEST_GOLDEN_PATH

**说明**：指定 Golden 文件路径（用于测试对比）

| 属性 | 值 |
|-----|---|
| 类型 | CMake 选项（路径） |
| 默认值 | 无 |
| 影响范围 | 测试对比基准 |

### 5.7 ENABLE_STEST_GOLDEN_PATH_CLEAN

**说明**：清理 Golden 文件

| 属性 | 值 |
|-----|---|
| 类型 | CMake 选项（ON/OFF） |
| 默认值 | `OFF` |
| 影响范围 | Golden 文件管理 |

### 5.8 PYPTO_UTEST_PARALLEL_NUM

**说明**：单元测试并行数量

| 属性 | 值 |
|-----|---|
| 类型 | 整数 |
| 默认值 | 由 `build_ci.py` 的 `job_num` 决定 |
| 影响范围 | 测试执行并行度 |

**使用示例**：

```bash
export PYPTO_UTEST_PARALLEL_NUM=4
```

---

## 6. 系统依赖环境变量

### 6.1 LD_LIBRARY_PATH

**说明**：动态库搜索路径（系统环境变量）

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（路径列表，用 `:` 分隔） |
| 默认值 | 系统默认 |
| 影响范围 | 运行时库加载 |

**使用示例**：

```bash
# 添加 CANN 库路径
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib:$LD_LIBRARY_PATH"
```

### 6.2 PATH

**说明**：可执行文件搜索路径（系统环境变量）

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（路径列表，用 `:` 分隔） |
| 默认值 | 系统默认 |
| 影响范围 | 工具查找（cmake、编译器等） |

**使用示例**：

```bash
# 添加 CANN 工具路径
export PATH="$ASCEND_HOME_PATH/bin:$PATH"
```

### 6.3 PYTHONPATH

**说明**：Python 模块搜索路径（系统环境变量）

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（路径列表，用 `:` 分隔） |
| 默认值 | 系统默认 |
| 影响范围 | Python 模块导入 |

**使用示例**：

```bash
# 添加当前目录到 Python 路径
export PYTHONPATH=".:$PYTHONPATH"
```

### 6.4 ASCEND_VISIBLE_DEVICES

**说明**：控制 Ascend 设备可见性（CANN 环境变量）

| 属性 | 值 |
|-----|---|
| 类型 | 字符串（设备 ID 列表，用 `,` 分隔） |
| 默认值 | 所有设备可见 |
| 影响范围 | 设备访问控制 |

**使用示例**：

```bash
# 只使用设备 0
export ASCEND_VISIBLE_DEVICES=0

# 使用设备 0 和 1
export ASCEND_VISIBLE_DEVICES=0,1
```

---

## 7. 环境变量使用示例

### 7.1 开发调试场景

```bash
#!/bin/bash
# 开发调试配置

# 启用详细日志
export PYPTO_LOG_LEVEL=DEBUG

# 显示完整异常回溯
export PTO_BACKTRACE=1

# 启用调试模式
export PYPTO_DEBUG=1

# 转储 IR 用于分析
export PYPTO_DUMP_IR=./ir_dump

# 运行 Python 脚本
python my_script.py
```

### 7.2 生产环境场景

```bash
#!/bin/bash
# 生产环境配置

# 只记录错误
export PYPTO_LOG_LEVEL=ERROR

# 日志输出到文件
export PYPTO_LOG_FILE=/var/log/pypto/app.log

# 使用 JSON 格式便于日志分析
export PYPTO_LOG_FORMAT=json

# 隐藏回溯
export PTO_BACKTRACE=0

# 运行应用
python app.py
```

### 7.3 性能分析场景

```bash
#!/bin/bash
# 性能分析配置

# 启用性能分析
export PYPTO_PROFILE=1

# 只记录关键日志
export PYPTO_LOG_LEVEL=INFO

# 运行并收集性能数据
python benchmark.py
```

### 7.4 CI/CD 场景

```bash
#!/bin/bash
# CI/CD 配置

# 使用 JSON 格式便于解析
export PYPTO_LOG_FORMAT=json
export PYPTO_LOG_LEVEL=INFO

# 启用编译缓存加速
export PYPTO_CACHE_DIR=/cache/pypto

# 转储 IR 用于验证
export PYPTO_DUMP_IR=/artifacts/ir

# 运行测试
pytest tests/
```

### 7.5 运行示例程序场景

```bash
#!/bin/bash
# 运行示例程序配置

# 设置设备 ID（必需）
export TILE_FWK_DEVICE_ID=0

# 设置 CANN 环境
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib:$LD_LIBRARY_PATH"
export PATH="$ASCEND_HOME_PATH/bin:$PATH"

# 可选：固定输出目录便于查看
export TILE_FWK_OUTPUT_DIR=/tmp/pypto_output

# 运行示例
python examples/hello_world/hello_world.py
```

### 7.6 精度调试场景

```bash
#!/bin/bash
# 精度调试配置

# 固定输出目录（便于对比）
export TILE_FWK_OUTPUT_DIR=/path/to/debug_output

# 启用详细日志
export PYPTO_LOG_LEVEL=DEBUG
export GLOBAL_LOG_LEVEL=0

# 转储 IR 用于分析
export PYPTO_DUMP_IR=$TILE_FWK_OUTPUT_DIR/ir

# 设置设备
export TILE_FWK_DEVICE_ID=0

# 运行调试
python my_model.py
```

---

## 8. 最佳实践

### 8.1 日志级别选择指南

| 场景 | 推荐级别 | 说明 |
|-----|---------|------|
| 开发调试 | DEBUG | 显示所有调试信息 |
| 集成测试 | INFO | 显示关键流程信息 |
| 生产环境 | WARN 或 ERROR | 只显示异常情况 |
| 问题排查 | DEBUG + LOG_FILE | 记录到文件便于分析 |

### 8.2 环境变量管理

**推荐使用 `.env` 文件**：

```bash
# .env.development
PYPTO_LOG_LEVEL=DEBUG
PTO_BACKTRACE=1
PYPTO_DEBUG=1
```

```bash
# .env.production
PYPTO_LOG_LEVEL=ERROR
PYPTO_LOG_FILE=/var/log/pypto/app.log
PYPTO_LOG_FORMAT=json
```

**加载方式**：

```bash
# 使用 dotenv
source .env.development

# 或使用 Python python-dotenv
# from dotenv import load_dotenv
# load_dotenv('.env.development')
```

### 8.3 环境变量优先级

**优先级顺序**（从高到低）：
1. 命令行参数
2. 环境变量
3. 代码默认值

**示例**：
```python
# 代码中：level = os.environ.get("PYPTO_LOG_LEVEL", "INFO")
# 如果设置了环境变量，使用环境变量的值
# 如果未设置，使用默认值 "INFO"
```

### 8.4 注意事项

1. **环境变量优先级**：环境变量会覆盖代码中的默认配置
2. **日志文件权限**：确保日志目录有写入权限
3. **敏感信息**：不要在日志中记录敏感信息
4. **性能影响**：DEBUG 级别会产生大量日志，生产环境慎用
5. **磁盘空间**：启用日志文件时注意磁盘空间

---

## 附录：环境变量速查表

### A.1 日志相关

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|-------|------|
| `PYPTO_LOG_LEVEL` | 字符串 | INFO | 全局日志级别 |
| `PYPTO_LOG_FILE` | 路径 | - | 日志文件路径 |
| `PYPTO_LOG_FORMAT` | 字符串 | detailed | 日志格式 |
| `PYPTO_LOG_JSON` | 整数 | 0 | JSON 格式日志 |
| `GLOBAL_LOG_LEVEL` | 整数 | 3 | C++日志级别（旧版） |
| `PTO_BACKTRACE` | 整数 | 0 | 显示异常回溯 |
| `ASCEND_GLOBAL_LOG_LEVEL` | 整数 | - | CANN 日志级别 |

### A.2 编译相关

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|-------|------|
| `PYPTO_BUILD_EXT_ARGS` | 字符串 | - | CMake 构建参数 |
| `PYPTO_THIRD_PARTY_PATH` | 路径 | - | 第三方源码包路径 |
| `ASCEND_HOME_PATH` | 路径 | - | Ascend工具链路径 |
| `TILEFWK_CONFIG_PATH` | 路径 | - | 配置文件路径 |
| `PTO_TILE_LIB_CODE_PATH` | 路径 | - | Tile 库代码路径 |

### A.3 运行时相关

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|-------|------|
| `PYPTO_CACHE_DIR` | 路径 | ~/.cache/pypto | 缓存目录 |
| `PYPTO_DEVICE_ID` | 整数 | 0 | 默认设备ID（Python） |
| `TILE_FWK_DEVICE_ID` | 整数 | - | 设备ID（C++，必需） |
| `TILE_FWK_OUTPUT_DIR` | 路径 | - | 输出目录 |
| `TILE_FWK_STEST_DEVICE_ID` | 整数 | 0 | 测试设备ID |
| `PYPTO_HOME` | 路径 | $HOME/.pypto | 运行时数据目录 |
| `AST_DATADUMP_PATH` | 字符串 | - | 启用数据转储（true） |
| `HOME` | 路径 | - | 用户主目录（系统变量，用于缓存和运行时数据） |

### A.4 调试相关

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|-------|------|
| `PYPTO_DEBUG` | 整数 | 0 | 调试模式 |
| `PYPTO_DUMP_IR` | 路径 | - | IR转储目录 |
| `PYPTO_PROFILE` | 整数 | 0 | 性能分析 |

### A.5 测试相关（CMake 选项）

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|-------|------|
| `ENABLE_STEST_EXECUTE_DEVICE_ID` | ON/OFF | OFF | 自动执行测试设备ID |
| `ENABLE_STEST_DUMP_JSON` | ON/OFF | OFF | 转储 JSON |
| `ENABLE_STEST_INTERPRETER_CONFIG` | ON/OFF | OFF | 解释器配置 |
| `ENABLE_STEST_BINARY_CACHE` | ON/OFF | OFF | 二进制缓存（增量测试时自动启用） |
| `ENABLE_STEST_GOLDEN_PATH` | 路径 | - | Golden 路径 |
| `ENABLE_STEST_GOLDEN_PATH_CLEAN` | ON/OFF | OFF | 清理 Golden |
| `PYPTO_UTEST_PARALLEL_NUM` | 整数 | - | 测试并行数 |

### A.6 系统依赖

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|-------|------|
| `LD_LIBRARY_PATH` | 路径列表 | - | 动态库搜索路径 |
| `PATH` | 路径列表 | - | 可执行文件搜索路径 |
| `PYTHONPATH` | 路径列表 | - | Python 模块搜索路径 |
| `ASCEND_VISIBLE_DEVICES` | 字符串 | - | 设备可见性 |

---

## 相关文档

- [日志机制分析报告](04-logging-analysis.md) - 日志系统详细分析
- [完整调试指南](00-complete-guide.md) - 调试方法总览
- [编译阶段功能详解](../03-mechanisms/01-compile-stage.md) - 编译配置

