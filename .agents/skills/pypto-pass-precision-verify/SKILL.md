---
name: pypto-pass-precision-verify
description: 验证PyPTO Pass精度问题，定位精度问题出现在哪个Pass，并尝试修复
trigger: 验证Pass精度问题、pass verify、Pass精度调试、定位Pass精度问题、Pass精度报错、Pass精度不一致
---

## 快速参考

| 场景 | 错误码 | 处理方法 |
|-----|-------|---------|
| 前端问题 | `0xB4001U` | 调用 `pypto-precision-compare` 技能 |
| OP 报错 | `0xB200FU` | 检查 IR 图，对比 Before/After |
| 精度问题 | `0xB4001U` | 使用 PreCheck/PostCheck → pass_compare.py → 上板结果比对 |
| 所有验证通过但精度异常 | 无报错 | 二分前端或二分 CCE |

---

## 常见场景速查表

| 场景 | 关键步骤 | 工具/命令 |
|-----|---------|----------|---------|
| 快速定位前端问题 | 查看tensor_graph验证结果 → 调用pypto-precision-compare技能 | `grep "tensor_graph Verify" log/*.log` |
| OP报错排查 | 检查IR图 → 对比Before/After → 查询OP详情 → 确认实际匹配| `get_op_info.py --op-magic <ID>` |
| Pass精度问题定位 | PreCheck/PostCheck → pass_compare → 上板比对 | 3步骤流程 |
| Codegen问题排查 | 二分前端 → 二分CCE → 打印验证 | `binary_cce.py --print-idx` |

**快速诊断流程**：
```
出现精度问题
    ↓
查看验证日志
    ↓
是否有报错码？
    ├─ 有 0xB4001U → tensor_graph FAIL → 前端问题（调用pypto-precision-compare）
    ├─ 有 0xB200FU → OP报错 → IR图分析
    ├─ 有 0xB4001U + Pass名称 → Pass精度问题 → PreCheck/PostCheck流程 + pass_compare → 上板比对
    ├─ 无报错但精度异常 → 二分前端或二分CCE
```

---

## 目录

1. [简介](#简介)
2. [环境与配置](#环境与配置)
3. [操作步骤](#操作步骤)
4. [错误码速查表](#错误码速查表)
5. [问题处理流程](#问题处理流程)
6. [二分CCE方法](#二分cce方法)
7. [IR图分析方法](#ir图分析方法)
8. [常见错误案例库](#常见错误案例库)
9. [注意事项](#注意事项)

---

## 简介

本技能用于验证 PyPTO Pass 侧的精度问题。当 PyPTO 算子上板执行后输出数据与 torch 输出不一致时，用于排查问题是否出现在 Pass 处理阶段。

> **前端问题处理**：若日志显示 `tensor_graph Verify FAIL`，说明问题在前端代码，请直接调用 `pypto-precision-compare` 技能进行定位。

---

## 环境与配置

环境依赖检查、环境变量配置和配置说明请参考以下文档：

| 文档 | 内容 |
|------|------|
| [environment_setup.md](./environment_setup.md) | 环境依赖检查、环境变量配置 |
| [config_guide.md](./config_guide.md) | verify_options配置、tile_fwk_config.json配置、配置备份与恢复 |

**快速配置检查**：
```bash
# 验证环境变量
echo "ASCEND_WORK_PATH: $ASCEND_WORK_PATH"
echo "ASCEND_GLOBAL_LOG_LEVEL: $ASCEND_GLOBAL_LOG_LEVEL"

# 验证PyPTO安装
python3 -c "import pypto; print('PyPTO installed')"
```

---

## 操作步骤

### 步骤一：配置校验开关

> **开始前**：请评估当前测试用例规模。如果数据量较大（如 Shape 参数 T>1000），建议：
> - 缩小 shape（如 T=64）以加快验证
> - 确认是否有最小可用用例
> - 防止编译/运行时间过长导致卡死
> - **删除 LOOP 中的 unrolllist 参数**：使计算图变为按照1的力度展开，减少编译复杂度

**缩小用例方法详解**：

| 方法 | 适用场景 | 操作说明 | 效果 |
|------|---------|---------|------|
| 缩小 shape | Shape 参数较大 | 将 T、B 等参数缩小至最小值（如 T=64） | 减少数据量，加快编译运行 |
| 删除 unrolllist | LOOP 级联复杂 | 移除 LOOP 的 unrolllist 参数，使用默认力度展开 | 计算图按力度1展开，减少计算图数量 |

**删除 unrolllist 参数示例**：

```python
# 原代码（有 unrolllist）
for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx", unroll_list=[64, 32, 16]):
    # ... 复杂计算图 ...

# 缩小后（删除 unrolllist）
for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx"):  # 删除 unrolllist，默认按1力度展开
    # ... 计算图按最小力度展开，减少计算图数量，便于调试 ...
```

> **说明**：删除 `unrolllist` 参数后，LOOP 将按照最小力度（1）展开，生成单张计算图，便于快速定位问题。当问题定位完成后，可恢复 `unrolllist` 参数验证完整场景。

详细配置请参考：[config_guide.md](./config_guide.md)

### 步骤二：编译并运行

```bash
# 编译安装（加 --no-build-isolation）
python3 -m pip install . --verbose --no-build-isolation

# 运行测试
python3 your_test_case.py
```

**输出目录位置说明**：

| 目录类型 | 位置 | 内容 |
|---------|------|------|
| **验证数据目录** | `./output/output_*` | Pass 验证结果、tensor 数据、IR 图 |
| **组件日志目录** | `$ASCEND_WORK_PATH/log/` | Pass 侧日志、Machine 上板日志 |

### 步骤三：分析验证结果与定位问题

运行后会打印验证结果，错误码统一定义于 `framework/src/interface/interpreter/verify_error.h`。

根据日志中的错误码和验证阶段，参考[问题处理流程](#问题处理流程)进行处理。

> **Pass 精度判断标准**：
> 
> Pass 侧精度是否通过，**只看最后一个 Pass（CodegenPreproc）是否正确**。
> 
> - 如果 CodegenPreproc Pass 验证结果为 PASS → Pass 侧整体通过
> - 如果中间 Pass 报错但 CodegenPreproc PASS → 中间 Pass 错误可忽略（可能是工具误报）
> - 如果 CodegenPreproc FAIL → 需要定位具体 Pass 问题

---

## 错误码速查表

| 错误码 | 名称 | 阶段 | 处理方法 |
|-------|------|------|---------|
| `0xB4001U` | VERIFY_RESULT_MISMATCH | 前端/Pass | 参考[问题处理流程](#问题处理流程) |
| `0xB200FU` | RUNTIME_EXCEPTION | Pass | 检查 OP 属性，参考 IR 图 |
| `0xB0001U` | VERIFY_NOT_ENABLE | 环境 | 检查 `torch >= 2.1.0` |
| 其他 | — | 未知 | 联系开发人员 |

---

## 问题处理流程

### 情况一：tensor_graph Verify FAIL（前端问题）

| 项目 | 说明 |
|-----|------|
| **错误码** | `0xB4001U`（VERIFY_RESULT_MISMATCH） |
| **日志特征** | `[VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED` |
| **处理** | 调用 `pypto-precision-compare` 技能 |

---

### 情况二：Pass 级别 Verify FAIL

#### 2.1 OP 报错

| 项目 | 说明 |
|-----|------|
| **错误码** | `0xB200FU`（RUNTIME_EXCEPTION） |
| **日志特征** | `[operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB ...` |
| **处理** | 查看对应 Pass 的 IR 图，分析该 OP 是否缺失属性 |

> **重要**：遇到 OP 报错时，请先**对比 Before 和 After 两个 IR 文件**，确认是否为工具误报。

#### 2.2 精度问题

| 项目 | 说明 |
|-----|------|
| **错误码** | `0xB4001U`（VERIFY_RESULT_MISMATCH） |
| **日志特征** | `[VERIFY]:ErrCode: FB4001! pass_06_SplitReshape Verify result FAILED` |

**Pass 侧精度问题定位流程**：

1. **首先使用 PreCheck/PostCheck 方法**：打开 `tile_fwk_config.json` 中对应 Pass 的开关
2. **其次利用 pass_compare.py**：对比失败 Pass 与前置 Pass 的每个 OP 节点
3. **最后使用上板结果比对**：将 Machine 上板后的结果与 verify_result 中各 Pass 保存的输出逐一比对

**pass_compare.py 使用方法**：

```bash
python3 tools/verifier/pass_compare.py --p <FailedPass> <GoldenPass> --verify_path=/path/to/verify_data
```

---

### 情况三：所有验证都 PASS 但仍有精度问题

| 项目 | 说明 |
|-----|------|
| **错误码** | 无报错，但上板结果与 golden 不一致 |
| **原因** | 问题可能在 Codegen 或 Machine 执行阶段 |

**排查决策流程**：

```
所有验证PASS但精度异常
        ↓
┌─ 用户输入精度报错现象？
│
│   是 → 调用 Pass侧常见错误分析
│       → 查阅 error_cases.md 匹配常见问题
│       → 根据现象特征定位可能原因
│
│   否（默认） → 采用二分前端方法
│       → 调用 pypto-precision-compare 技能
│       → 定位问题 Op
│
│   用户告知某Op错误 / 需打印上板数据？
│       → 采用二分CCE方法
│       → 参考 binary_cce.md
│       → 打印真实上板输出验证
│
└───────────────────────────────────────────
```

**三种排查方向详解**：

| 排查方向 | 适用场景 | 触发条件 | 操作方法 |
|---------|---------|---------|---------|
| **Pass侧常见错误分析** | 精度报错现象可描述 | 用户输入精度报错现象 | 查阅 error_cases.md，根据现象匹配常见错误 |
| **二分前端** | 定位前端代码问题 | 默认采用，或无明显现象特征 | 调用 `pypto-precision-compare` 技能，定位问题 Op |
| **二分 CCE** | 打印真实上板输出 | 用户告知某Op错误，或需打印上板数据 | 参考 [binary_cce.md](./binary_cce.md) |

> **建议顺序**：优先尝试常见错误分析（成本最低） → 二分前端 → 二分CCE（成本依次递增）

---

## 二分CCE方法

详细使用方法请参考：**[binary_cce.md](./binary_cce.md)**

**快速入口**：

| 场景 | 方法 | 说明 |
|------|------|------|
| 多CCE场景 | 方式一/二 | 目标定位或二分搜索 |
| 单CCE场景 | 方式三/四 | 单CCE直接二分或手动修改 |

### ⚠️ 配置与打印语句确认环节

**在使用二分CCE技能时，完成配置和添加打印语句后，必须向用户展示以下信息供确认**：

> **打印方法详解**：参见 [binary_cce.md - 打印方法概述](./binary_cce.md#打印方法概述)

#### 确认模板

```markdown
## 二分CCE配置确认

请确认以下配置和打印语句是否正确：

### 1. 关键配置检查

| 配置项 | 文件位置 | 当前值 | 正确值 | 状态 |
|--------|---------|--------|--------|------|
| `ENABLE_AICORE_PRINT` | `aicore_print.h` | [展示当前值] | 1 | ✅/❌ |
| `fixed_output_path` | `tile_fwk_config.json` | [展示当前值] | true | ✅/❌ |
| `force_overwrite` | `tile_fwk_config.json` | [展示当前值] | false | ✅/❌ |
| `parallel_compile` | `tile_fwk_config.json` | [展示当前值] | 1 | ✅/❌ |

### 2. CCE文件信息

- **CCE文件路径**: `./kernel_aicore/[文件名].cpp`
- **目标CCE索引**: [idx]
- **CCE文件数量**: [count]

### 3. 打印方法与语句

**打印类型**: [GM数据打印 / UB数据打印 / Shape打印 / Offset打印]

#### GM数据打印（最常用）
```cpp
#include "tilefwk/aicore_print.h"
AiCorePrintGmTensor(param->ctx, (__gm__[dtype]*)[tensor名].GetAddr(), [末尾偏移量], [起始偏移量]);
```

#### UB数据打印
```cpp
#include "tilefwk/aicore_print.h"
AiCorePrintUbTensor(param->ctx, (__ub__[dtype]*)[tensor名].GetAddr(), [末尾偏移量], [起始偏移量]);
```

#### Shape打印
```cpp
#include "tilefwk/aicore_print.h"
AiCorePrintShape(param->ctx, Shape2Dim(sym_15_dim_0, sym_15_dim_1));
```

#### Offset打印
```cpp
#include "tilefwk/aicore_print.h"
AiCorePrintShape(param->ctx, Coord2Dim(...));
```

**打印参数说明**：
- 数据类型(dtype): float / bfloat16_t / half / int32_t
- 偏移量范围: [末尾偏移量], [起始偏移量]（元素数量 = 末尾-起始+1 ≤ 80）
- Shape变量: sym_XX_dim_Y（从CCE中查找）
- 打印位置: kernel_start / kernel_end / [具体Op前后]

### 4. 预期日志位置

运行后打印数据将出现在：
```
$ASCEND_WORK_PATH/log/debug/device-[id]/device-[id].log
```

### 请确认

1. 配置是否正确？
2. 打印类型是否选择正确？
3. 偏移量范围是否合理（末尾-起始+1 ≤ 80）？
4. 是否可以继续运行测试？
```

#### 必须展示的内容

| 项目 | 说明 |
|------|------|
| **配置值** | 从文件中读取并展示实际值 |
| **CCE文件名** | 展示完整文件路径 |
| **打印类型** | GM数据打印 / UB数据打印 / Shape打印 |
| **打印语句** | 展示实际添加的代码（含行号） |
| **打印参数** | tensor名、dtype、偏移量范围 |
| **日志位置** | 告知用户运行后查看日志的位置 |

#### 示例输出

```markdown
## 二分CCE配置确认

请确认以下配置和打印语句是否正确：

### 1. 关键配置检查

| 配置项 | 文件位置 | 当前值 | 正确值 | 状态 |
|--------|---------|--------|--------|------|
| `ENABLE_AICORE_PRINT` | `aicore_print.h` | 1 | 1 | ✅ |
| `fixed_output_path` | `tile_fwk_config.json` | true | true | ✅ |
| `force_overwrite` | `tile_fwk_config.json` | false | false | ✅ |
| `parallel_compile` | `tile_fwk_config.json` | 1 | 1 | ✅ |

### 2. CCE文件信息

- **CCE文件路径**: `./kernel_aicore/TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11_0_aiv.cpp`
- **目标CCE索引**: 0
- **CCE文件数量**: 1

### 3. 打印方法与语句

**打印类型**: GM数据打印

**文件**: `./kernel_aicore/TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11_0_aiv.cpp`

```cpp
// 第 156 行添加的打印语句：
#include "tilefwk/aicore_print.h"

AiCorePrintGmTensor(param->ctx, (__gm__float*)gmTensor_output.GetAddr(), 63, 0);
```

**打印参数说明**：
- Tensor名称: gmTensor_output
- 数据类型: float
- 偏移量范围: 0 ~ 63（共64个元素，≤80符合要求）
- 打印位置: kernel 函数开始处（kernel_start）

### 4. 预期日志位置

运行后打印数据将出现在：
```
$ASCEND_WORK_PATH/log/debug/device-0/device-0.log
```

### 请确认

所有配置正确，打印语句已添加，可以继续运行测试。
```

**后续步骤**：用户确认后，执行测试并查看日志中的打印数据。

---

## IR图分析方法

用于判断是否为工具误报或真实错误，以及辅助CCE二分定位。

详细 IR 分析指南请参考：`pypto/.agents/skills/pypto-pass-error-locator/references/ir-analysis-guide.md`

IR分析实战示例请参考：**[ir_analysis_examples.md](./ir_analysis_examples.md)**

### IR分析工具使用

```bash
# 查询指定 OP 的详细信息
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName.tifwkgr \
    --op-magic 10003

# 列出所有 OP
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName.tifwkgr \
    --list-ops
```

---

## 常见错误案例库

遇到常见错误时，请查阅 **[error_cases.md](./error_cases.md)** 查看具体案例的诊断与解决方案。

**案例索引**：
- **案例01**：Reshape操作导致Pass验证报错 → reshape后添加 `+ 0.0` 规避
- **案例02**：精度对比通过但Pass验证报错 → 以精度对比结果为准

---

## 注意事项

1. **Pass 精度判断标准**：只看 CodegenPreproc Pass 是否通过
2. **前端问题**：tensor_graph FAIL 时调用 pypto-precision-compare
3. **工具误报**：IR 图显示 shape 实际匹配时可能是误报
4. **二分CCE配置**：必须设置 `fixed_output_path=true`, `force_overwrite=false`
5. **打印限制**：AiCorePrint 偏移量范围对应的元素数量（末尾-起始+1）不能超过 80
6. **配置恢复**：调试完成后恢复原始配置
7. **缩小用例恢复**：删除 unrolllist 参数定位问题后，需恢复原始 unrolllist 验证完整场景

---

## 相关文档索引

| 文档 | 内容 |
|------|------|
| [environment_setup.md](./environment_setup.md) | 环境依赖检查、环境变量配置 |
| [config_guide.md](./config_guide.md) | 配置说明、配置备份与恢复 |
| [binary_cce.md](./binary_cce.md) | 二分CCE完整指南 |
| [ir_analysis_examples.md](./ir_analysis_examples.md) | IR分析实战示例 |
| [error_cases.md](./error_cases.md) | 常见错误案例库 |