# pypto_smoke.sh 脚本详解

> **用途：** PyPTO Smoke 测试自动化脚本  
> **适用场景：** CI/CD 流水线、本地批量测试、多设备并行测试  
> **执行环境：** Docker 容器或 CI 服务器

---

## 目录

1. [脚本概述](#1-脚本概述)
2. [核心功能](#2-核心功能)
3. [执行流程](#3-执行流程)
4. [关键组件](#4-关键组件)
5. [使用示例](#5-使用示例)
6. [参数说明](#6-参数说明)
7. [环境要求](#7-环境要求)

---

## 1. 脚本概述

`pypto_smoke.sh` 是一个自动化测试脚本，用于在多个 NPU 设备上并行执行 PyPTO 的 Smoke 测试（冒烟测试）。该脚本主要用于：

- **CI/CD 流水线**：自动检测代码变更并运行相关测试
- **多设备测试**：在 16 个设备（0-15）上并行执行测试
- **Python 和 C++ 测试**：同时运行 Python 层和 C++ 层的测试用例

### 1.1 脚本特点

- ✅ **错误处理**：使用 `set -e` 确保任何命令失败时立即退出
- ✅ **日志记录**：带时间戳的彩色日志输出
- ✅ **耗时统计**：记录每个任务和总执行时间
- ✅ **环境隔离**：检查环境升级锁，避免并发冲突
- ✅ **CI 模式**：支持增量测试（只测试变更的文件）

---

## 2. 核心功能

### 2.1 日志系统

脚本定义了 4 种日志函数：

```bash
LOG_HEAD()    # 绿色标题日志，用于重要步骤
LOG_DO()      # 紫色命令日志，显示执行的命令
LOG_INFO()    # 普通信息日志
LOG_ERROR()   # 红色错误日志
```

**示例输出**：
```
[INFO] 20250105-120000 Run SMoke in env 0001
[Command] 20250105-120001 source /root/.bashrc
[ERROR] 20250105-120002 Python interpreter not found!
```

### 2.2 任务执行函数

`run_build_ci()` 函数封装了任务执行逻辑：

```bash
run_build_ci <python_path> <description> <build_ci_args...>
```

**功能**：
- 验证 Python 解释器是否存在
- 记录任务开始时间
- 执行 `build_ci.py` 并传递参数
- 记录任务结束时间和耗时
- 检查返回码，失败时退出

---

## 3. 执行流程

### 流程图

```
开始
  ↓
步骤1: 检查环境升级锁
  ↓ (如果锁存在，退出)
步骤2: 初始化环境
  ├─ 加载 .bashrc
  ├─ 激活 conda 环境 (py39)
  └─ 设置路径变量
  ↓
步骤3: 配置 CANN 环境
  ├─ 读取配置文件 ($HOME/pypto_cann.cfg)
  ├─ 提取 CANN 路径
  └─ 加载 setenv.bash
  ↓
步骤4: 解析命令行参数
  ├─ --src: 源码目录（必需）
  └─ --ci: CI 模式标志（可选）
  ↓
步骤5: 执行测试任务
  ├─ CI 模式: 生成变更文件列表
  ├─ 任务1: Python 测试（STest + Examples）
  └─ 任务2: C++ 测试（STest）
  ↓
结束（输出总耗时）
```

### 3.1 详细步骤说明

#### 步骤 1: 检查升级锁文件

```bash
ENV_UPGRADE_LOCK_FILE="$HOME/env_upgrade_flag_v1"
if [ -f "$ENV_UPGRADE_LOCK_FILE" ]; then
    LOG_ERROR "Current Environment upgrading, please wait a moment."
    exit 1
fi
```

**目的**：防止在环境升级过程中运行测试，避免冲突。

#### 步骤 2: 预设环境

```bash
source $HOME/.bashrc
source /root/miniconda3/bin/activate py39

DATA_DIR="/data/ci"
GOLDEN_PATH="$DATA_DIR/pypto_golden"           # Golden 文件路径
CANN_3RD_LIB_PATH="$DATA_DIR/pypto_3rd_lib_path"  # 第三方库路径
PYTHON3_EXE="/root/miniconda3/envs/py39/bin/python"
```

**关键变量**：
- `GOLDEN_PATH`: 用于测试对比的基准文件目录
- `CANN_3RD_LIB_PATH`: CANN 第三方库路径
- `PYTHON3_EXE`: Python 3 解释器路径

#### 步骤 3: 设置 CANN 环境变量

```bash
CONFIG_FILE="$HOME/pypto_cann.cfg"
CANN_PATH=$(awk -F '=' '/^CANN=/ {print $2}' "$CONFIG_FILE")
SETENV_SH="$CANN_PATH/bin/setenv.bash"
source $SETENV_SH
```

**配置文件格式** (`$HOME/pypto_cann.cfg`):
```
CANN=/usr/local/Ascend/ascend-toolkit/latest
```

**作用**：加载 CANN 工具链的环境变量（`LD_LIBRARY_PATH`、`PATH` 等）。

#### 步骤 4: 解析脚本参数

**支持的参数**：
- `--src <dir>`: 源码目录（必需）
- `--ci`: 启用 CI 模式

**CI 模式功能**：
- 自动检测 Git 变更文件（`git diff HEAD~1 HEAD`）
- 生成变更文件列表（`$DATA_DIR/pypto_changed_files.txt`）
- 传递给 `build_ci.py` 进行增量测试

#### 步骤 5: 执行测试任务

**设备参数**：
```bash
device_params=(
    "-d=0"  "-d=1"  "-d=2"  "-d=3"
    "-d=4"  "-d=5"  "-d=6"  "-d=7"
    "-d=8"  "-d=9"  "-d=10" "-d=11"
    "-d=12" "-d=13" "-d=14" "-d=15"
)
```

**通用参数**：
```bash
common_params=(
    "--clean"                                    # 清理构建产物
    "--verbose"                                  # 详细输出
    "--cann_3rd_lib_path=$CANN_3RD_LIB_PATH"    # 第三方库路径
    "--golden_path=$GOLDEN_PATH"                 # Golden 路径
    "$CHANGED_FILES_PARAM"                       # 变更文件列表（CI 模式）
)
```

**执行的任务**：

1. **Python 测试**：
```bash
run_build_ci "$PYTHON3_EXE" "Python(STest & Examples)" \
    "${common_params[@]}" \
    --timeout=420 \          # 超时 420 秒
    --no_isolation \         # 不使用隔离模式
    --stest \                # 运行系统测试
    --example \              # 运行示例程序
    "${device_params[@]}"    # 在 16 个设备上执行
```

2. **C++ 测试**：
```bash
run_build_ci "$PYTHON3_EXE" "C++(STest)" \
    --frontend=cpp \         # 使用 C++ 前端
    "${common_params[@]}" \
    --timeout=300 \          # 超时 300 秒
    --stest \                # 运行系统测试
    "${device_params[@]}"    # 在 16 个设备上执行
```

---

## 4. 关键组件

### 4.1 日志函数详解

#### LOG_HEAD()
```bash
function LOG_HEAD() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "\n${BGreen}[INFO] ${date_time} ${assert_msg}${Color_Off}"
}
```
- **颜色**：绿色（`BGreen`）
- **用途**：标记重要步骤的开始/结束

#### LOG_DO()
```bash
function LOG_DO() {
   local cmd="$*"
   date_time=$(date +%Y%m%d-%H%M%S)
   echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}${cmd}${Color_Off}"
   ${cmd}
}
```
- **颜色**：紫色（`BPurple`）
- **用途**：显示并执行命令
- **特点**：会实际执行命令，失败时触发 `set -e` 退出

#### LOG_ERROR()
```bash
function LOG_ERROR() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BRed}[ERROR] ${date_time} ${assert_msg}${Color_Off}"
}
```
- **颜色**：红色（`BRed`）
- **用途**：输出错误信息

### 4.2 任务执行函数

```bash
run_build_ci() {
    local _python3="$1"       # Python 解释器路径
    local desc="$2"            # 任务描述
    shift 2                   # 移除前两个参数
    
    # 验证 Python 解释器
    if ! command -v "$_python3" &>/dev/null; then
        LOG_ERROR "Python interpreter '$_python3' not found!"
        exit 1
    fi
    
    # 记录开始时间
    start_time=$(date +%s)
    LOG_HEAD "[BGN] $desc "
    
    # 执行 build_ci.py
    LOG_DO "$_python3" "build_ci.py" "$@"
    local ret=$?
    
    # 计算耗时
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    LOG_HEAD "[END] $desc, Ret $ret, duration $elapsed secs."
    
    # 检查返回码
    if [ $ret -ne 0 ]; then
        LOG_ERROR "$desc failed"
        exit $ret
    fi
}
```

**关键点**：
- 使用 `shift 2` 移除前两个参数，剩余参数传递给 `build_ci.py`
- 记录开始和结束时间，计算耗时
- 检查返回码，失败时退出

---

## 5. 使用示例

### 5.1 基本用法

```bash
# 在本地运行（非 CI 模式）
./pypto_smoke.sh --src /path/to/pypto

# 在 CI 环境中运行（增量测试）
./pypto_smoke.sh --src /path/to/pypto --ci
```

### 5.2 完整执行流程示例

```bash
# 1. 脚本开始
[INFO] 20250105-120000 Run SMoke in env 0001

# 2. 加载环境
[Command] 20250105-120001 source /root/.bashrc
[Command] 20250105-120002 source /root/miniconda3/bin/activate py39
[INFO] 20250105-120003 GOLDEN_PATH=/data/ci/pypto_golden

# 3. 加载 CANN 环境
[Command] 20250105-120004 source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 4. 执行 Python 测试
[INFO] 20250105-120005 [BGN] Python(STest & Examples)
[Command] 20250105-120006 /root/miniconda3/envs/py39/bin/python build_ci.py ...
[INFO] 20250105-120700 [END] Python(STest & Examples, Ret 0, duration 695 secs.

# 5. 执行 C++ 测试
[INFO] 20250105-120701 [BGN] C++(STest)
[Command] 20250105-120702 /root/miniconda3/envs/py39/bin/python build_ci.py ...
[INFO] 20250105-121000 [END] C++(STest), Ret 0, duration 299 secs.

# 6. 总耗时
[INFO] 20250105-121001 All builds completed successfully, Total execution time: 994 seconds
execute sample success
```

---

## 6. 参数说明

### 6.1 脚本参数

| 参数 | 类型 | 必需 | 说明 |
|-----|------|------|------|
| `--src <dir>` | 路径 | ✅ | PyPTO 源码目录 |
| `--ci` | 标志 | ❌ | 启用 CI 模式（增量测试） |

### 6.2 build_ci.py 参数（通过脚本传递）

| 参数 | 说明 |
|-----|------|
| `--clean` | 清理构建产物 |
| `--verbose` | 详细输出 |
| `--cann_3rd_lib_path=<path>` | CANN 第三方库路径 |
| `--golden_path=<path>` | Golden 文件路径 |
| `--changed_files=<file>` | 变更文件列表（CI 模式） |
| `--timeout=<seconds>` | 超时时间 |
| `--no_isolation` | 不使用隔离模式 |
| `--stest` | 运行系统测试 |
| `--example` | 运行示例程序 |
| `--frontend=cpp` | 使用 C++ 前端 |
| `-d=<id>` | 设备 ID（0-15） |

---

## 7. 环境要求

### 7.1 必需文件

1. **配置文件**：`$HOME/pypto_cann.cfg`
   ```
   CANN=/usr/local/Ascend/ascend-toolkit/latest
   ```

2. **CANN 环境脚本**：`$CANN_PATH/bin/setenv.bash`
   - 必须存在且可执行
   - 用于设置 CANN 相关环境变量

### 7.2 必需目录

- `$DATA_DIR/pypto_golden`：Golden 文件目录
- `$DATA_DIR/pypto_3rd_lib_path`：第三方库目录
- `$SRC_DIR`：PyPTO 源码目录（通过 `--src` 指定）

### 7.3 必需环境

- **Conda 环境**：`py39`（Python 3.9）
- **Python 解释器**：`/root/miniconda3/envs/py39/bin/python`
- **Git**：用于 CI 模式的变更检测
- **CANN 工具链**：已安装并配置

### 7.4 环境变量

脚本会设置以下环境变量（通过 `setenv.bash`）：
- `ASCEND_HOME_PATH`
- `LD_LIBRARY_PATH`
- `PATH`
- 其他 CANN 相关变量

---

## 8. 故障排查

### 8.1 常见错误

#### 错误 1: 环境升级锁存在
```
[ERROR] Current Environment upgrading, please wait a moment.
```
**解决**：等待环境升级完成，或删除 `$HOME/env_upgrade_flag_v1` 文件。

#### 错误 2: CANN 配置文件不存在
```
[ERROR] $HOME/pypto_cann.cfg not found, Can't find spec cann path
```
**解决**：创建配置文件并设置 CANN 路径。

#### 错误 3: Python 解释器不存在
```
[ERROR] Python interpreter '/root/miniconda3/envs/py39/bin/python' not found!
```
**解决**：检查 conda 环境是否正确安装，或修改 `PYTHON3_EXE` 变量。

#### 错误 4: 源码目录无效
```
[ERROR] --src argument is required
[ERROR] /path/to/src is not a valid directory
```
**解决**：确保使用 `--src` 参数指定有效的源码目录。

### 8.2 调试技巧

1. **查看详细日志**：脚本已启用 `--verbose` 模式
2. **检查环境变量**：在脚本中添加 `env | grep ASCEND` 查看 CANN 环境
3. **单独测试**：手动执行 `build_ci.py` 命令进行调试
4. **检查设备状态**：确保所有设备（0-15）可用

---

## 9. 与 build_ci.py 的关系

`pypto_smoke.sh` 是 `build_ci.py` 的**包装脚本**，主要作用：

1. **环境准备**：自动设置 CANN 环境、conda 环境
2. **参数组装**：将设备列表、路径等参数传递给 `build_ci.py`
3. **任务编排**：按顺序执行 Python 和 C++ 测试
4. **日志记录**：提供统一的日志格式和耗时统计

**实际执行**：
```bash
python build_ci.py \
    --clean \
    --verbose \
    --cann_3rd_lib_path=/data/ci/pypto_3rd_lib_path \
    --golden_path=/data/ci/pypto_golden \
    --timeout=420 \
    --stest \
    --example \
    -d=0 -d=1 -d=2 ... -d=15
```

---

## 10. 最佳实践

### 10.1 CI/CD 集成

```yaml
# .gitlab-ci.yml 示例
smoke_test:
  script:
    - bash pypto_smoke.sh --src $CI_PROJECT_DIR --ci
  timeout: 2h
```

### 10.2 本地测试

```bash
# 1. 准备环境
export HOME=/root
mkdir -p $HOME
echo "CANN=/usr/local/Ascend/ascend-toolkit/latest" > $HOME/pypto_cann.cfg

# 2. 运行测试
bash pypto_smoke.sh --src /path/to/pypto
```

### 10.3 增量测试（CI 模式）

```bash
# 只测试变更的文件
bash pypto_smoke.sh --src /path/to/pypto --ci
```

**变更文件检测逻辑**：
```bash
git diff --name-only HEAD~1 HEAD > /data/ci/pypto_changed_files.txt
```

---

---

## 11. 增量测试详细逻辑

### 11.1 增量测试概述

增量测试（Incremental Testing）是 CI/CD 流水线中的核心优化机制，通过分析代码变更文件，只执行相关的测试用例，大幅减少测试时间。

**核心思想**：
- 不是所有代码变更都需要运行所有测试
- 根据变更文件路径，匹配对应的测试模块
- 只执行受影响的测试用例

### 11.2 增量测试流程

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: pypto_smoke.sh (CI 模式)                            │
│  └─> git diff HEAD~1 HEAD --name-only                       │
│  └─> 生成: /data/ci/pypto_changed_files.txt                │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: build_ci.py                                         │
│  └─> 接收: --changed_files=/data/ci/pypto_changed_files.txt│
│  └─> 解析: TestsExecuteParam.changed_file                   │
│  └─> 传递给 CMake: ENABLE_TESTS_EXECUTE_CHANGED_FILE        │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 3: CMake 配置阶段                                      │
│  └─> 设置: ENABLE_TESTS_EXECUTE_CHANGED_FILE=<file_path>   │
│  └─> 在测试阶段调用 analysis_changed_files.py              │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 4: analysis_changed_files.py                          │
│  ├─> 读取: classify_rule.yaml (规则文件)                    │
│  ├─> 读取: changed_files.txt (变更文件列表)                 │
│  ├─> 匹配: 变更文件 vs 模块白名单 (write_list)              │
│  └─> 输出: 需要执行的测试用例列表 (逗号分隔)                │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 5: CMake 测试执行                                      │
│  └─> 只执行匹配到的测试用例                                 │
│  └─> 跳过未匹配的测试用例                                   │
└─────────────────────────────────────────────────────────────┘
```

### 11.3 变更文件检测（pypto_smoke.sh）

**代码位置**：`pypto_smoke.sh` 第 171-180 行

```bash
if [[ "$CI_MODE" == true ]]; then
    changed_file_path="$DATA_DIR/pypto_changed_files.txt"
    if [ -f "$changed_file_path" ]; then
        rm "$changed_file_path"
    fi
    git diff --name-only HEAD~1 HEAD > "$changed_file_path"
    CHANGED_FILES_PARAM="--changed_files=$changed_file_path"
    LOG_INFO "Changed files content as follows:"
    LOG_DO "cat $changed_file_path"
fi
```

**功能说明**：
1. **检测条件**：仅在 `--ci` 参数启用时执行
2. **Git 命令**：`git diff --name-only HEAD~1 HEAD`
   - 比较当前提交（HEAD）与上一个提交（HEAD~1）
   - `--name-only`：只输出文件名，不输出差异内容
3. **输出格式**：每行一个文件路径（相对路径）
   ```
   framework/src/interface/utils/log.h
   python/pypto/_controller.py
   examples/hello_world/hello_world.py
   ```

**变更文件示例**：
```
framework/src/interface/utils/log.h
framework/src/machine/host/backend.cpp
python/pypto/frontend/parser/error.py
python/tests/st/test_sin.py
```

### 11.4 规则文件结构（classify_rule.yaml）

**文件位置**：`classify_rule.yaml` 或 `classify_rule_<type>.yaml`

**结构说明**：

```yaml
# 类型级别配置（stest 或 utest）
stest:
  # 类型级别的白名单（所有模块共享）
  write_list:
    - "docs/"           # 文档变更不触发测试
    - "*.md"            # Markdown 文件变更不触发测试
    - "CMakeLists.txt"  # CMake 文件变更不触发测试
  
  # 模块配置
  module_name_1:
    write_list:         # 模块级别的白名单（覆盖类型级别）
      - "framework/src/module1/"
    cases:              # 该模块对应的测试用例列表
      - "test_case_1"
      - "test_case_2"
  
  module_name_2:
    write_list:
      - "framework/src/module2/"
      - "python/pypto/module2/"
    cases:
      - "test_case_3"
      - "test_case_4"
```

**关键概念**：

1. **write_list（白名单）**：
   - 定义哪些文件路径的变更**不触发**该模块的测试
   - 支持相对路径和通配符
   - 如果变更文件命中白名单，该模块的测试会被跳过

2. **cases（测试用例列表）**：
   - 定义该模块对应的测试用例名称
   - 当模块被触发时，执行这些测试用例

3. **匹配规则**：
   - 如果变更文件**不在**任何模块的 `write_list` 中，触发该模块的所有测试
   - 如果变更文件**在**某个模块的 `write_list` 中，跳过该模块的测试

### 11.5 匹配算法（analysis_changed_files.py）

**核心函数**：`Module.is_trigger()`

```python
def is_trigger(self, changed: List[Path]) -> Tuple[bool, List[str]]:
    # 若无 changed, 默认触发所有用例
    if not changed:
        return True, self.cases
    
    # 当所有 changed 均命中白名单, 无需触发
    for c in changed:
        c_skip: bool = False
        for w in self.write:
            if self._relative_to(c, w):
                c_skip = True
                logging.debug("Changed(%s) hit writeList(%s), skip Module(%s)", c, w, self.name)
                break
        if not c_skip:
            logging.debug("Changed(%s) not hit writeList, trigger Module(%s)", c, self.name)
            return True, self.cases
    return False, []
```

**算法逻辑**：

1. **无变更文件**：返回 `True`，触发所有用例（全量测试）
2. **有变更文件**：
   - 遍历每个变更文件
   - 检查是否命中模块的 `write_list`
   - 如果**任何一个**变更文件**未命中**白名单，触发该模块
   - 如果**所有**变更文件都命中白名单，跳过该模块

**示例场景**：

**场景 1：触发测试**
```
变更文件: framework/src/interface/utils/log.h
模块 write_list: ["docs/", "*.md"]
结果: 触发（log.h 不在白名单中）
```

**场景 2：跳过测试**
```
变更文件: docs/note/05-debugging/06-pyto-smoke-script.md
模块 write_list: ["docs/", "*.md"]
结果: 跳过（文档变更命中白名单）
```

**场景 3：部分触发**
```
变更文件:
  - framework/src/interface/utils/log.h
  - docs/README.md
模块 write_list: ["docs/", "*.md"]
结果: 触发（log.h 不在白名单中，即使 README.md 命中白名单）
```

### 11.6 完整示例

#### 示例 1：单一文件变更

**变更文件** (`pypto_changed_files.txt`):
```
framework/src/interface/utils/log.h
```

**规则文件** (`classify_rule_stest.yaml`):
```yaml
stest:
  write_list:
    - "docs/"
    - "*.md"
  
  logging_module:
    write_list:
      - "framework/src/interface/utils/log.h"  # 白名单
    cases:
      - "test_logging_basic"
      - "test_logging_levels"
  
  backend_module:
    write_list:
      - "framework/src/machine/"
    cases:
      - "test_backend_init"
      - "test_backend_execute"
```

**匹配结果**：
- `logging_module`: 跳过（`log.h` 命中白名单）
- `backend_module`: 触发（`log.h` 不在白名单中）
- **执行用例**: `test_backend_init`, `test_backend_execute`

#### 示例 2：多个文件变更

**变更文件** (`pypto_changed_files.txt`):
```
framework/src/interface/utils/log.h
framework/src/machine/host/backend.cpp
python/pypto/_controller.py
```

**规则文件** (`classify_rule_stest.yaml`):
```yaml
stest:
  write_list:
    - "docs/"
  
  logging_module:
    write_list:
      - "framework/src/interface/utils/"
    cases:
      - "test_logging"
  
  backend_module:
    write_list:
      - "framework/src/machine/"
    cases:
      - "test_backend"
  
  controller_module:
    write_list:
      - "python/pypto/"
    cases:
      - "test_controller"
```

**匹配结果**：
- `logging_module`: 跳过（`log.h` 命中白名单）
- `backend_module`: 跳过（`backend.cpp` 命中白名单）
- `controller_module`: 跳过（`_controller.py` 命中白名单）
- **执行用例**: 无（所有模块都被跳过）

**注意**：如果所有模块都被跳过，可能意味着：
1. 变更文件都是文档或配置文件
2. 规则文件配置过于严格
3. 需要调整白名单规则

#### 示例 3：混合变更

**变更文件** (`pypto_changed_files.txt`):
```
framework/src/interface/utils/log.h
docs/note/05-debugging/06-pyto-smoke-script.md
python/tests/st/test_sin.py
```

**规则文件** (`classify_rule_stest.yaml`):
```yaml
stest:
  write_list:
    - "docs/"
    - "*.md"
  
  logging_module:
    write_list:
      - "framework/src/interface/utils/"
    cases:
      - "test_logging"
  
  test_module:
    write_list:
      - "python/tests/"  # 测试文件变更不触发测试
    cases:
      - "test_sin"
      - "test_cos"
```

**匹配结果**：
- `logging_module`: 跳过（`log.h` 命中白名单）
- `test_module`: 跳过（`test_sin.py` 命中白名单）
- **执行用例**: 无

**说明**：测试文件本身的变更通常不触发测试执行，因为：
- 测试文件变更可能是修复测试用例本身
- 避免测试文件修改导致测试循环触发

### 11.7 规则文件最佳实践

#### 1. 白名单设计原则

**推荐做法**：
```yaml
stest:
  # 类型级别：所有模块共享的白名单
  write_list:
    - "docs/"              # 文档目录
    - "*.md"               # Markdown 文件
    - "CMakeLists.txt"     # CMake 配置文件
    - "*.cmake"            # CMake 脚本
    - "*.yaml"             # YAML 配置文件
    - "*.yml"              # YAML 配置文件
    - "*.sh"               # Shell 脚本（非核心逻辑）
    - "*.py"               # Python 脚本（非核心逻辑，需谨慎）
  
  # 模块级别：特定模块的白名单
  module_name:
    write_list:
      - "framework/src/other_module/"  # 其他模块的路径
    cases:
      - "test_case_1"
```

**注意事项**：
- 白名单路径应该尽可能具体，避免误跳过
- 文档、配置文件通常加入白名单
- 测试文件本身通常加入白名单
- 核心代码路径**不要**加入白名单

#### 2. 模块划分原则

**按功能模块划分**：
```yaml
stest:
  frontend_module:
    write_list:
      - "framework/src/machine/"      # 后端代码
      - "python/pypto/runtime.py"     # 运行时代码
    cases:
      - "test_frontend_parser"
      - "test_frontend_ast"
  
  backend_module:
    write_list:
      - "python/pypto/frontend/"      # 前端代码
      - "framework/src/interface/"     # 接口代码
    cases:
      - "test_backend_execute"
      - "test_backend_memory"
```

**按测试类型划分**：
```yaml
stest:
  operator_tests:
    write_list:
      - "framework/src/passes/"        # Pass 代码
      - "python/pypto/frontend/"      # 前端代码
    cases:
      - "test_add"
      - "test_mul"
      - "test_softmax"
  
  controlflow_tests:
    write_list:
      - "framework/src/interface/"     # 接口代码
      - "python/pypto/runtime.py"     # 运行时代码
    cases:
      - "test_if"
      - "test_loop"
      - "test_while"
```

#### 3. 调试增量测试

**启用调试模式**：
```bash
# 直接调用 analysis_changed_files.py
python cmake/scripts/analysis_changed_files.py \
    -r . \
    -t stest \
    -c /data/ci/pypto_changed_files.txt \
    -d  # 启用调试模式
```

**输出示例**：
```
2025-01-05 12:00:00 - analysis_changed_files.py:54 - Changed(framework/src/interface/utils/log.h) not hit writeList, trigger Module(backend_module)
2025-01-05 12:00:00 - analysis_changed_files.py:51 - Changed(docs/README.md) hit writeList(docs/), skip Module(backend_module)
```

### 11.8 增量测试的优势与限制

#### 优势

1. **大幅减少测试时间**：
   - 全量测试：可能需要数小时
   - 增量测试：通常只需几分钟到几十分钟

2. **提高 CI/CD 效率**：
   - 快速反馈：开发者可以更快获得测试结果
   - 资源节约：减少 CI 服务器的计算资源消耗

3. **精准测试**：
   - 只测试受影响的模块
   - 避免无关测试的干扰

#### 限制

1. **规则文件维护成本**：
   - 需要及时更新规则文件
   - 新增模块需要配置对应规则

2. **误判风险**：
   - 白名单配置不当可能跳过必要的测试
   - 跨模块依赖可能被忽略

3. **首次提交**：
   - 首次提交或大范围重构时，可能触发大量测试
   - 此时增量测试的优势不明显

### 11.9 增量测试与二进制缓存的关系

#### 11.9.1 自动启用机制

**重要发现**：增量测试会自动启用二进制缓存（`ENABLE_STEST_BINARY_CACHE`）。

**代码逻辑**（`build_ci.py` 第 517 行）：
```python
self.stest_exec: STestExecuteParam = STestExecuteParam(
    args=args, 
    enable_binary_cache=self.exec.ci_model  # ci_model = True 当启用增量测试时
)
```

**判断条件**（`build_ci.py` 第 316-317 行）：
```python
@property
def ci_model(self) -> bool:
    return True if self.changed_file else False
```

**结论**：
- ✅ **启用增量测试**（`--changed_files` 存在）→ `ci_model = True` → `enable_binary_cache = True`
- ❌ **未启用增量测试**（`--changed_files` 不存在）→ `ci_model = False` → `enable_binary_cache = False`

#### 11.9.2 设计意图

**为什么增量测试要启用二进制缓存？**

1. **加速 CI/CD 执行**：
   - 增量测试通常用于 CI/CD 流水线
   - 二进制缓存可以复用之前编译的二进制文件
   - 避免重复编译，大幅减少测试时间

2. **提高效率**：
   - 增量测试只执行部分测试用例
   - 这些测试用例可能依赖相同的二进制文件
   - 缓存机制可以避免重复编译这些二进制文件

3. **资源节约**：
   - 减少编译时间，降低 CI 服务器负载
   - 提高测试反馈速度

#### 11.9.3 二进制缓存机制

**缓存位置**：
```cpp
// framework/src/machine/cache_manager/cache_manager.cpp
cacheDirPath_ = homeEnvPath + "/ast_data/" + PlatformManager::Instance().GetShortSocVersion();
```

**缓存目录结构**：
```
$HOME/ast_data/<SOC_VERSION>/
├── binary_cache_1.bin
├── binary_cache_2.bin
└── ...
```

**启用时机**：
- 在测试初始化时（`DeviceLauncherContext::DeviceInit()`）启用
- 在测试结束时（`DeviceLauncherContext::DeviceFini()`）恢复原配置

**代码实现**（`device_launcher.h`）：
```cpp
#ifdef ENABLE_STEST_BINARY_CACHE
    // BinaryCache
    oriEnableBinaryCache = config::GetHostConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
    config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, true);
#endif
```

#### 11.9.4 注意事项

1. **缓存一致性**：
   - 二进制缓存基于编译结果
   - 如果代码变更影响二进制文件，缓存可能失效
   - 系统会自动处理缓存失效和重新编译

2. **缓存目录管理**：
   - 缓存文件存储在 `$HOME/ast_data/` 目录
   - 需要确保有足够的磁盘空间
   - 定期清理旧缓存文件

3. **手动控制**：
   - 虽然增量测试自动启用缓存，但可以通过其他方式手动控制
   - 检查 `ENABLE_STEST_BINARY_CACHE` CMake 选项的值

#### 11.9.5 验证方法

**检查是否启用了二进制缓存**：

1. **查看 build_ci.py 输出**：
```bash
python build_ci.py --changed_files=/path/to/changed_files.txt --stest
# 输出中应该包含：
# Enable Binary Cache : True
```

2. **检查 CMake 配置**：
```bash
# 在构建目录中
grep ENABLE_STEST_BINARY_CACHE CMakeCache.txt
# 应该显示：ENABLE_STEST_BINARY_CACHE:BOOL=ON
```

3. **检查运行时日志**：
```bash
# 查看测试日志，应该看到缓存相关的信息
grep -i "binary.*cache" test.log
```

### 11.10 故障排查

#### 问题 1：应该执行的测试被跳过

**原因**：
- 变更文件路径命中白名单
- 规则文件配置过于严格

**解决**：
1. 检查变更文件列表：`cat /data/ci/pypto_changed_files.txt`
2. 检查规则文件：查看 `classify_rule_stest.yaml`
3. 调整白名单：移除或缩小相关路径的白名单范围

#### 问题 2：不应该执行的测试被触发

**原因**：
- 变更文件路径未命中任何白名单
- 规则文件配置不完整

**解决**：
1. 检查变更文件列表
2. 在规则文件中添加相应的白名单规则
3. 更新模块的 `write_list`

#### 问题 3：所有测试都被跳过

**原因**：
- 所有变更文件都命中白名单
- 变更都是文档或配置文件

**解决**：
1. 检查变更文件列表
2. 如果是文档变更，这是正常行为
3. 如果是代码变更，检查规则文件配置

#### 问题 4：二进制缓存未生效

**原因**：
- 增量测试未启用（`--changed_files` 未设置）
- 缓存目录权限问题
- 缓存目录空间不足

**解决**：
1. 确认是否使用了 `--ci` 参数或 `--changed_files` 参数
2. 检查 `$HOME/ast_data/` 目录权限：`ls -ld $HOME/ast_data/`
3. 检查磁盘空间：`df -h $HOME`
4. 查看测试日志中的缓存相关错误信息

### 11.11 并行测试时的缓存安全性分析

#### 11.11.1 问题：并行测试是否会相互影响？

**核心问题**：当多个 STest 用例并行执行时，`ENABLE_STEST_BINARY_CACHE` 是否会导致测试用例相互影响？

#### 11.11.2 保护机制分析

**1. 互斥锁保护（内存级别）**

**代码位置**：`cache_manager.h` 第 54 行
```cpp
mutable std::mutex cacheMutex_;
```

**保护范围**：
- `MatchBinCache()`：检查缓存是否存在（第 82 行）
- `SaveTaskFile()`：保存缓存文件（第 113 行）
- `RecoverTask()`：恢复缓存任务（第 161 行）

**作用**：
- 确保同一进程内的多个线程不会同时访问缓存管理器
- 防止内存状态的竞态条件

**2. 文件锁保护（文件系统级别）**

**代码位置**：`cache_manager.cpp` 第 120-122 行
```cpp
std::string lockFilePath = cacheDirPath_ + "/" + CACHE_FILE_PREFIX + 
                           deviceAgentTask->compileTask->GetCacheKey() + CACHE_LOCK_FILE_SUFFIX;
FILE *fp = LockAndOpenFile(lockFilePath);
```

**实现机制**（`file_utils.cpp`）：
```cpp
FILE* LockAndOpenFile(const std::string &lockFilePath) {
    FILE *fp = fopen(lockFilePath.c_str(), "a+");
    if (fp == nullptr) {
        return nullptr;
    }
    // 使用 fcntl 文件锁
    if (!FcntlLockFile(fileno(fp), F_WRLCK)) {
        ALOG_WARN("Fail to lock file:", lockFilePath.c_str());
        fclose(fp);
        return nullptr;
    }
    return fp;
}
```

**作用**：
- 使用 `fcntl` 文件锁（`F_WRLCK`）保护文件写入
- 防止不同进程同时写入同一个缓存文件
- 锁文件命名：`ast_op_<cacheKey>.lock`

#### 11.11.3 缓存文件命名规则

**文件类型**：

1. **主二进制文件**（基于 cacheKey）：
   ```
   ast_op_<cacheKey>.o
   ```
   - 每个测试用例的 cacheKey 应该是唯一的
   - 如果 cacheKey 相同，会共享同一个缓存文件

2. **内核文件**（基于 cacheKey）：
   ```
   ast_op_<cacheKey>_kernel.o
   ```

3. **自定义 SO 文件**（基于 OpFuncName）：
   ```
   lib<OpFuncName>_control.so
   ```
   - ⚠️ **潜在问题**：如果多个测试用例使用相同的 `OpFuncName`，会共享同一个文件

4. **自定义 JSON 文件**（基于 OpFuncName）：
   ```
   lib<OpFuncName>_control.json
   ```
   - ⚠️ **潜在问题**：同上

5. **锁文件**（基于 cacheKey）：
   ```
   ast_op_<cacheKey>.lock
   ```

#### 11.11.4 潜在冲突场景

**场景 1：相同 cacheKey 的测试用例并行执行**

**情况**：
- 测试用例 A 和 B 生成相同的 `cacheKey`
- 两个用例并行执行

**影响**：
- ✅ **安全**：文件锁机制会确保只有一个进程写入缓存文件
- ✅ **安全**：互斥锁确保内存状态一致
- ⚠️ **注意**：如果用例 A 正在写入，用例 B 会等待锁释放

**场景 2：相同 OpFuncName 的测试用例并行执行**

**情况**：
- 测试用例 A 和 B 使用相同的 `OpFuncName`（如 `"add"`）
- 两个用例并行执行

**影响**：
- ⚠️ **潜在问题**：`customSoPath` 和 `customJsonPath` 基于 `OpFuncName`
- ⚠️ **风险**：如果用例 A 正在写入 `libadd_control.so`，用例 B 可能读取到不完整的文件
- ✅ **缓解**：文件锁机制会保护写入操作
- ⚠️ **注意**：如果用例 A 写入后，用例 B 可能读取到用例 A 的缓存，而不是自己的

**场景 3：不同 cacheKey 但相同 OpFuncName**

**情况**：
- 测试用例 A：`cacheKey = "test1"`, `OpFuncName = "add"`
- 测试用例 B：`cacheKey = "test2"`, `OpFuncName = "add"`
- 两个用例并行执行

**影响**：
- ✅ **主二进制文件**：安全（不同的 cacheKey，不同的文件）
- ⚠️ **自定义 SO/JSON**：可能冲突（相同的 OpFuncName，相同的文件）

#### 11.11.5 代码分析

**关键代码**（`cache_manager.cpp` 第 107-110 行）：
```cpp
std::string customSoPath = cacheDirPath_ + "/lib" + 
                           OpInfoManager::GetInstance().GetOpFuncName() + 
                           CACHE_CUSTOM_BIN_FILE_SUFFIX;
std::string customJsonPath = cacheDirPath_ + "/lib" + 
                             OpInfoManager::GetInstance().GetOpFuncName() + 
                             CACHE_CUSTOM_JSON_FILE_SUFFIX;
```

**问题**：
- `GetOpFuncName()` 可能是全局状态
- 如果多个测试用例使用相同的操作名称，会共享同一个自定义 SO/JSON 文件

**保护机制**：
```cpp
std::lock_guard<std::mutex> lock_guard(cacheMutex_);  // 互斥锁
if (!RealPath(customSoPath).empty() && !RealPath(customJsonPath).empty()) {
    // 如果文件已存在，直接返回（不覆盖）
    return;
}
// 写入时使用文件锁
FILE *fp = LockAndOpenFile(lockFilePath);
```

#### 11.11.6 结论与建议

**结论**：

1. **主二进制文件（基于 cacheKey）**：
   - ✅ **安全**：文件锁和互斥锁双重保护
   - ✅ **隔离**：不同 cacheKey 使用不同文件

2. **自定义 SO/JSON 文件（基于 OpFuncName）**：
   - ⚠️ **潜在风险**：相同 OpFuncName 的测试用例会共享文件
   - ✅ **写入保护**：文件锁确保写入安全
   - ⚠️ **读取风险**：如果用例 A 写入后，用例 B 可能读取到用例 A 的缓存

**建议**：

1. **确保 cacheKey 唯一性**：
   - 确保每个测试用例生成唯一的 `cacheKey`
   - 如果 cacheKey 生成逻辑有问题，可能导致缓存冲突

2. **避免相同 OpFuncName 并行执行**：
   - 如果可能，确保使用相同 `OpFuncName` 的测试用例串行执行
   - 或者修改缓存命名规则，包含更多唯一标识符

3. **监控并行测试结果**：
   - 如果发现测试结果不稳定，检查是否有缓存冲突
   - 查看日志中的缓存匹配信息

4. **测试隔离**：
   - 如果怀疑缓存冲突，可以临时禁用二进制缓存进行对比测试
   - 或者为每个测试用例使用独立的缓存目录

#### 11.11.7 验证方法

**检查缓存文件冲突**：

```bash
# 1. 查看缓存目录中的文件
ls -la $HOME/ast_data/<SOC_VERSION>/

# 2. 检查是否有重复的 OpFuncName
grep -r "GetOpFuncName" test_logs/ | sort | uniq -d

# 3. 检查锁文件
ls -la $HOME/ast_data/<SOC_VERSION>/*.lock

# 4. 查看测试日志中的缓存信息
grep -i "cache" test_logs/ | grep -i "matched\|missed"
```

**测试并行安全性**：

```bash
# 运行相同的测试用例多次，检查结果是否一致
for i in {1..10}; do
    python -m pytest tests/st/test_add.py -v
done
```

**如果发现冲突**：

1. 检查 cacheKey 生成逻辑
2. 检查 OpFuncName 是否唯一
3. 考虑禁用二进制缓存或使用独立缓存目录

---

## 相关文档

- [build_ci.py 使用说明](../02-core/11-build.md) - 构建系统详解
- [环境变量参考](05-environment-variables.md) - 环境变量配置
- [调试指南](00-complete-guide.md) - 调试方法总览

