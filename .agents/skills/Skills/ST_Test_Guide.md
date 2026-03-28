# PyPTO Operation ST 测试指南

## 1. 概述

本文档用于指导 PyPTO 算子的 ST（System Test）测试执行，涵盖环境准备、检查、测试命令、自动化测试循环与常见问题排查。

文档分为两部分：
- **Part 1（第 2~7 章）**：面向人类操作者的环境与测试手册
- **Part 2（第 8~11 章）**：面向 AI Agent 的自动化测试与修复循环指令

---

## Part 1：环境与测试手册

---

## 2. 目录结构

```
pypto/
├── framework/tests/st/operation/
│   ├── test_case/                          # 测试用例 CSV 文件
│   │   ├── {OperationName}_st_test_cases.csv
│   │   └── ...
│   ├── src/                                # 测试执行源码（C++）
│   │   ├── test_{operation}_operation.cpp
│   │   └── ...
│   └── python/
│       └── vector_operator_golden.py       # 默认 golden 脚本
├── tools/scripts/
│   └── run_operation_test_with_config.py   # ST 测试入口脚本
└── examples/02_intermediate/operators/softmax/
    └── softmax.py                          # 环境验证脚本
```

### 核心源码路径

| 内容 | 路径 |
|------|------|
| 测试用例 CSV | `pypto/framework/tests/st/operation/test_case/{OperationName}_st_test_cases.csv` |
| ST 执行源码（C++） | `pypto/framework/tests/st/operation/src/test_{operation}_operation.cpp` |
| Golden 脚本 | `pypto/framework/tests/st/operation/python/vector_operator_golden.py` |
| 算子核心逻辑源码 | `pto-isa/interface/operation/vector/`（例如 `permute.cpp`） |
| 环境配置 Skill | `.agents/skills/pypto-environment-setup/SKILL.md` |

---

## 3. 环境准备

### 3.1 环境检查与安装

参考 Skill：`pypto-environment-setup`（路径：`.agents/skills/pypto-environment-setup/SKILL.md`）

#### 步骤 1：环境诊断

```bash
# 检查当前是否在 PyPTO 仓库根目录
[ -f "pyproject.toml" ] && [ -d "framework" ] && echo "✓ 当前目录是 PyPTO 仓库"

# 运行环境诊断脚本
python3 scripts/diagnose_env.py --checklist
```

> 所有项 ✅ OK 才能继续。有 ⚠️/❌ 项需按步骤 2 修复。

#### 步骤 2：安装缺失组件（按需执行）

```bash
# 安装编译依赖（cmake/gcc/make/g++/ninja/pip3）
bash tools/prepare_env.sh --quiet --type=deps --device-type=<a2|a3>

# 下载安装第三方源码包（json/libboundscheck）
bash tools/prepare_env.sh --quiet --type=third_party

# 下载 CANN 包
bash tools/prepare_env.sh --quiet --type=cann --only-download --device-type=<a2|a3> --install-path=$ASCEND_INSTALL_PATH

# 安装 CANN
bash tools/prepare_env.sh --quiet --type=cann --device-type=<a2|a3> --install-path=$ASCEND_INSTALL_PATH
```

> `--device-type` 根据芯片型号选择：910B → `a2`，910C → `a3`

### 3.2 环境配置

#### 加载 CANN 环境

```bash
# 设置 CANN 安装路径（默认 /usr/local/Ascend）
export ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
source ${ASCEND_INSTALL_PATH}/ascend-toolkit/set_env.sh
```

#### 设置 PTO-ISA 路径

```bash
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/pto-isa
```

> 如使用 CANN 自带 pto-isa：`export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH}/${arch}-linux`

#### 检查 NPU 卡并设置设备 ID

```bash
# 检查空闲 NPU 卡
bash .agents/skills/pypto-op-develop/scripts/list_idle_chip_ids.sh

# 设置空闲卡 ID（根据上一步结果填写）
export TILE_FWK_DEVICE_ID=<空闲 chip id>
```

#### 安装 torch 和 torch_npu

```bash
pip install torch==2.6.0 torch-npu==2.6.0.post3
```

#### 编译安装 PyPTO

```bash
pip uninstall pypto -y
rm -rf build_out
python3 -m pip install . --verbose
```

### 3.3 环境验证

```bash
# NPU 模式（有 NPU 环境时必须使用）
python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode npu

# SIM 模式（无 NPU 环境时使用）
# python3 examples/02_intermediate/operators/softmax/softmax.py --run_mode sim
```

> 通过标准：退出码 0，输出 `Softmax test passed`。

### 3.4 一键环境配置（可选）

将以上配置写入脚本，后续只需 source：

```bash
cat > env_setup.sh << 'EOF'
#!/bin/bash
export ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
source ${ASCEND_INSTALL_PATH}/ascend-toolkit/set_env.sh
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/pto-isa
export TILE_FWK_DEVICE_ID=0
echo "env_setup.sh 加载完成：TILE_FWK_DEVICE_ID=${TILE_FWK_DEVICE_ID}, PTO_TILE_LIB_CODE_PATH=${PTO_TILE_LIB_CODE_PATH}"
EOF
```

使用时：

```bash
source env_setup.sh
```

---

## 4. 测试执行

### 4.1 命令格式

```bash
source env_setup.sh

python3 ./tools/scripts/run_operation_test_with_config.py <OperationName> [选项]
```

### 4.2 参数说明

| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `op` | - | 必填 | - | 算子名称，首字母大写，如 `Transpose`、`Add`、`Matmul`、`Permute` |
| `--device` | `-d` | int | 0 | NPU 设备 ID（0-15） |
| `--start_index` | `-s` | int | 0 | 起始用例索引 |
| `--end_index` | `-e` | int | -1 | 结束用例索引（-1 表示全部） |
| `--input_file` | `-i` | str | 自动 | 自定义测试用例文件路径（默认读取 `test_case/{Op}_st_test_cases.csv`） |
| `--clean` | `-c` | flag | false | 清除编译结果后重新编译 |
| `--report` | - | str | test_result_report.xlsx | 测试报告文件名 |
| `--save_data` | - | flag | false | 保存 golden 和 plog 数据 |
| `--json_only` | - | flag | false | 仅将 CSV 转为 JSON，不执行测试 |
| `--golden_script` | - | str | 自动 | 自定义 golden 脚本路径 |
| `--python` | - | flag | false | 测试 Python 测试用例 |
| `--model` | - | flag | false | 使用 model 模式运行 |
| `--distributed_op` | - | flag | false | 分布式算子测试 |

### 4.3 用例索引说明

CSV 文件第 1 行为表头，第 2 行起为数据行。用例索引与 CSV 行号的关系：

**用例索引 = CSV 行号 - 2**

| CSV 行号 | 内容 | 用例索引 |
|----------|------|---------|
| 1 | 表头（column names） | - |
| 2 | 第 1 条用例 | 0 |
| 3 | 第 2 条用例 | 1 |
| 4 | 第 3 条用例 | 2 |
| ... | ... | ... |
| N | 第 N-1 条用例 | N-2 |

### 4.4 示例命令

```bash
# 测试 Transpose 算子，使用设备 0，执行全部用例
python3 ./tools/scripts/run_operation_test_with_config.py Transpose -d=0

# 测试 Transpose 算子，仅执行第 1 条用例（CSV 第 2 行，索引 0）
python3 ./tools/scripts/run_operation_test_with_config.py Transpose -d=0 -s=0 -e=0

# 测试 Transpose 算子，执行第 1~3 条用例（索引 0~2）
python3 ./tools/scripts/run_operation_test_with_config.py Transpose -d=0 -s=0 -e=2

# 测试 Transpose 算子，执行第 3~11 条用例（索引 2~10）
python3 ./tools/scripts/run_operation_test_with_config.py Transpose -d=0 -s=2 -e=10

# 使用 NPU 设备 1 测试 Add 算子
python3 ./tools/scripts/run_operation_test_with_config.py Add -d=1

# 测试 Matmul 算子并保存数据
python3 ./tools/scripts/run_operation_test_with_config.py Matmul -d=0 --save_data

# 使用自定义测试用例文件
python3 ./tools/scripts/run_operation_test_with_config.py Add -i path/to/custom_cases.csv

# 仅转换 CSV 为 JSON（不执行测试）
python3 ./tools/scripts/run_operation_test_with_config.py Add --json_only

# 清除编译缓存后重新测试
python3 ./tools/scripts/run_operation_test_with_config.py Transpose -d=0 -c
```

---

## 5. 可用算子列表

以下算子已有对应的 CSV 测试用例文件（`framework/tests/st/operation/test_case/`）：

| 算子名称 | 算子名称 | 算子名称 | 算子名称 |
|---------|---------|---------|---------|
| Abs | Add | Adds | Amax |
| Amin | ArgSort | BatchMatmul | BitwiseAnd |
| BitwiseAnds | BitwiseLeftShift | BitwiseLeftShifts | BitwiseNot |
| BitwiseOr | BitwiseOrs | BitwiseRightShift | BitwiseRightShifts |
| BitwiseXor | BitwiseXors | Cast | Cbrt |
| Ceil | CeilDiv | CeilDivs | Clip |
| Cmps | Compare | Concat | CopySign |
| CumSum | Div | Divs | Expand |
| ExpandExpDif | Exp | Exp2 | Expm1 |
| FillPad | Floor | Fmod | Fmods |
| Full | Gather | GatherElement | GatherMask |
| Gcd | Gcds | Hypot | IndexAdd |
| IndexPut_ | IsFinite | LReLU | Log |
| Log10 | Log1p | Log2 | LogicalAnd |
| LogicalNot | Matmul | Maximum | Minimum |
| Mul | Muls | Neg | OneHot |
| Pad | Permute | Pow | Pows |
| PReLU | Prod | Range | Reciprocal |
| Relu | Remainder | RemainderR | RemainderS |
| Round | Rsqrt | Scatter | ScatterTensor |
| ScatterUpdate | Sign | Signbit | SBitwiseLeftShift |
| SBitwiseRightShift | Sqrt | Sub | Subs |
| Sum | TopK | Transpose | TriL |
| TriU | Trunc | Var | Where |

---

## 6. 常见问题

### Q: 报错 `xxx is not exists`
CSV 文件路径不正确。检查 `framework/tests/st/operation/test_case/{OperationName}_st_test_cases.csv` 是否存在。

### Q: 报错 CANN 相关错误
确认已执行 `source ${ASCEND_INSTALL_PATH}/ascend-toolkit/set_env.sh`。

### Q: 报错 PTO-ISA 相关错误
确认已设置 `export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/pto-isa`。

### Q: NPU 设备繁忙
使用 `bash .agents/skills/pypto-op-develop/scripts/list_idle_chip_ids.sh` 查找空闲卡，并通过 `-d` 参数指定。

### Q: 编译失败
尝试使用 `-c` 参数清除编译缓存后重试：
```bash
python3 ./tools/scripts/run_operation_test_with_config.py Transpose -d=0 -c
```

### Q: torch_npu 版本不兼容
CANN 8.5.0 必须使用 `torch==2.6.0` + `torch-npu==2.6.0.post3`：
```bash
pip install torch==2.6.0 torch-npu==2.6.0.post3
```

### Q: `Error: Cant get any case to run when using...`
框架未读取到正确用例。请检查 C++ ST 执行源码 `framework/tests/st/operation/src/test_{operation}_operation.cpp` 中的配置，或者检查 Python golden 生成脚本 `vector_operator_golden.py`。

### Q: `ModuleNotFoundError: No module named 'openpyxl'`
环境中缺少表格读取依赖：
```bash
pip3 install openpyxl
```

### Q: `Error: Can't get deviceId`
命令缺少 device 参数。自动补充 `-d=0` 后重试。

---

## Part 2：Agent 自动化测试与修复指令

---

## 7. Agent 概述

本章节是专门提供给 AI Agent（例如 OpenCode / CANNBot）的标准化操作指导，用于指导 Agent 自动完成昇腾自定义算子的 ST 测试与循环修正工作。

### Agent 基本信息与路径上下文

作为 Agent，在开始测试前必须明确以下文件与环境上下文：

| 内容 | 路径 |
|------|------|
| 测试环境 Skill | `.agents/skills/pypto-environment-setup/SKILL.md` |
| 测试用例 CSV | `pypto/framework/tests/st/operation/test_case/{OperationName}_st_test_cases.csv` |
| ST 执行源码（C++） | `pypto/framework/tests/st/operation/src/test_{operation}_operation.cpp` |
| 算子核心逻辑源码 | `pto-isa/interface/operation/vector/`（例如 `permute.cpp`） |
| 默认 Golden 脚本 | `pypto/framework/tests/st/operation/python/vector_operator_golden.py` |

---

## 8. Agent 环境变量初始化规则

**【指令】** 每次执行测试脚本前，**必须**使用 `bash` 工具导出环境和源配置。建议将其链式组合到执行命令中：

```bash
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/pto-isa
source /mnt/workspace/Ascend/cann-9.0.0/bin/setenv.sh
cd /mnt/workspace/gitCode/pypto
```

> **注意**：路径需根据实际部署位置调整。如果当前仓库在 `/mnt/workspace/permute/pypto`，则 `cd /mnt/workspace/permute/pypto`。

---

## 9. Agent 执行测试命令语法

**【指令】** 使用 Python 脚本执行测试。脚本参数必须严格按照以下规则构造：

```bash
python3 ./tools/scripts/run_operation_test_with_config.py [OperationName] [-d=0] [-s=起始索引] [-e=结束索引]
```

**参数规则：**

1. **OperationName**：首字母大写的算子名称，例如 `Transpose`、`Permute`、`Add`、`Matmul`。
2. **-d**：务必指定 NPU 设备 ID（如 `-d=0`）。
3. **-s / -e**：索引值 = **对应的 CSV 用例行号 - 2**。
   - 例：执行第 0 条用例（对应 CSV 第 2 行） → `-s=0 -e=0`
   - 例：执行第 0 到 10 条用例 → `-s=0 -e=10`
   - 例：不指定 `-s` 和 `-e` 则执行全部用例

**完整链式命令示例：**

```bash
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/pto-isa && source /mnt/workspace/Ascend/cann-9.0.0/bin/setenv.sh && cd /mnt/workspace/permute/pypto && python3 ./tools/scripts/run_operation_test_with_config.py Permute -d=0 -s=0 -e=0
```

---

## 10. Agent 自动化测试与修复循环（Workflow）

作为 Agent，接管一个算子测试/修复任务时（如 `Permute`），必须严格执行以下**循环迭代流程**：

```
┌─────────────────────────────────────────────┐
│  Step 1: 运行单条用例                         │
│  （选择一条出错用例，或从索引 0 开始）            │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  Step 2: 解析输出结果                         │
│  Passed → Step 4                             │
│  Failed/Error → 记录错误日志 → Step 3         │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  Step 3: 修改算子源码并重新编译                 │
│  审查算子 C++ 源文件 → Edit 修改 → 回到 Step 1  │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  Step 4: 扩大用例验证范围                      │
│  单条通过后扩大范围（如 -s=0 -e=全部）           │
│  若新 Failed → 回到 Step 2 针对性修复          │
│  全部 Passed → 完成                           │
└─────────────────────────────────────────────┘
```

### Step 1: 运行单条用例

首先挑出单条出错用例进行测试。构造并执行 bash 组合命令：

```bash
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/pto-isa && source /mnt/workspace/Ascend/cann-9.0.0/bin/setenv.sh && cd /mnt/workspace/permute/pypto && python3 ./tools/scripts/run_operation_test_with_config.py Permute -d=0 -s=0 -e=0
```

### Step 2: 解析输出结果

读取 Bash 执行的输出（STDOUT/STDERR）：
- 如果返回 **Passed** → 进入 Step 4。
- 如果返回 **Failed/Error** → 记录错误日志，进入 Step 3。

### Step 3: 修改算子源码并重新编译

根据失败的输出，使用 `Read` 或 `Task` 工具审查该算子的 C++ 源文件：
- 文件路径示例：`pto-isa/interface/operation/vector/permute.cpp`
- 使用 `Edit` 工具对逻辑进行修改。
- 修改后，**回到 Step 1** 重新运行该测试用例，直到其完全通过为止。

### Step 4: 扩大用例验证范围

当该条用例测试通过后：
- 将用例范围扩大，例如改为 `-s=0 -e=<最大索引>` 执行全部用例。
- 若在此过程中发生新的 Failed，继续针对特定的行号重复 Step 2 和 Step 3。
- **直到所有的 case 都测试通过**。

---

## 11. Agent 错误排查知识库（Troubleshooting）

运行脚本时触发的常见基础设施异常，Agent 应按以下规则自动恢复，无需询问用户：

| 错误信息 | 原因 | Agent 应对 |
|---------|------|-----------|
| `Error: Cant get any case to run when using...` | 框架未读取到正确用例 | 检查 C++ ST 源码 `src/test_{op}_operation.cpp` 的配置，或检查 `vector_operator_golden.py` |
| `ModuleNotFoundError: No module named 'openpyxl'` | 缺少表格读取依赖 | 自动执行 `pip3 install openpyxl` 后重试测试命令 |
| `Error: Can't get deviceId` | 命令缺少 device 参数 | 自动补充 `-d=0` 后重试 |
| CANN 相关报错 | 未加载 CANN 环境 | 确认执行了 `source ${ASCEND_INSTALL_PATH}/ascend-toolkit/set_env.sh` |
| PTO-ISA 相关报错 | 未设置 PTO-ISA 路径 | 确认执行了 `export PTO_TILE_LIB_CODE_PATH=...` |
| NPU 设备繁忙 | 指定的 NPU 卡被占用 | 使用 `list_idle_chip_ids.sh` 查找空闲卡，更换 `-d` 参数 |
| 编译失败 | 缓存冲突或依赖缺失 | 添加 `-c` 参数清除编译缓存后重试 |
