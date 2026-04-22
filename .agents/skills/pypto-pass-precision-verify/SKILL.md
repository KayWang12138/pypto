---
name: pypto-pass-precision-verify
description: 验证PyPTO Pass侧精度问题，定位问题来源（前端/Pass/Codegen/Machine），并指导修复。当出现以下情况时使用：(1) PyPTO算子上板结果与torch不一致；(2) 验证日志显示精度报错（tensor_graph FAIL、Pass Verify FAIL）；(3) OP报错涉及动态shape/validshape需打印上板数据验证；(4) 所有验证PASS但精度异常；(5) 用户提到Pass精度调试、精度问题定位、验证报错分析。
---

## 快速诊断

| 场景 | 错误码 | 处理方法 |
|-----|-------|---------|
| 前端问题 | `0xB4001U` | 调用 `pypto-precision-compare` 技能 |
| OP 报错 | `0xB200FU` | 检查 IR 图，动态shape需打印验证 |
| Pass精度问题 | `0xB4001U` | PreCheck/PostCheck → pass_compare → 上板比对 |
| 无报错但精度异常 | 无 | 调用 pypto-precision-compare 二分前端 |

```
精度问题 → 查看验证日志 → 按错误码选择处理流程：
├─ 0xB4001U (tensor_graph) → 前端问题 → pypto-precision-compare
├─ 0xB200FU (OP报错) → IR图分析 → 动态shape则打印验证
├─ 0xB4001U + Pass名 → Pass精度 → PreCheck/PostCheck
└─ 无报错 → 二分前端 → pypto-precision-compare
```

---

## 目录

1. [简介](#简介)
2. [环境与配置](#环境与配置)
3. [操作步骤](#操作步骤)
4. [错误码速查表](#错误码速查表)
5. [问题处理流程](#问题处理流程)
6. [打印上板信息](#打印上板信息)
7. [IR图分析](#ir图分析)
8. [常见错误案例](#常见错误案例)
9. [注意事项](#注意事项)

---

## 简介

验证 PyPTO Pass 侧精度问题，定位问题来源（前端/Pass/Codegen/Machine）。

> `tensor_graph Verify FAIL` → 前端问题，直接调用 `pypto-precision-compare` 技能。

---

## 环境与配置

环境依赖检查、环境变量配置和配置说明请参考：

| 文档 | 内容 |
|------|------|
| [references/environment_setup.md](./references/environment_setup.md) | 环境依赖检查、环境变量配置 |
| [references/config_guide.md](./references/config_guide.md) | verify_options配置、tile_fwk_config.json配置 |

---

## 操作步骤

### 步骤一：配置校验开关

配置详情请参考 [references/config_guide.md](./references/config_guide.md)。

> 大数据量时（Shape T>1000）：缩小shape参数、删除LOOP的unrolllist参数。

### 步骤二：编译运行

```bash
python3 -m pip install . --verbose --no-build-isolation
python3 your_test_case.py
```

输出目录：`./output/output_*`（验证数据）、`$ASCEND_WORK_PATH/log/`（日志）

### 步骤三：分析验证结果

错误码定义：`framework/src/interface/interpreter/verify_error.h`

> **判断标准**：只看 CodegenPreproc Pass 是否通过。中间 Pass 报错但 CodegenPreproc PASS → 可忽略。

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

### 情况一：tensor_graph FAIL → 调用 `pypto-precision-compare`

### 情况二：Pass级别FAIL

> **前置配置**（必须）：在 `verify_options` 中配置以下参数，重新运行用例：
> ```json
> "pass_verify_pass_filter": "all",
> "pass_verify_save_tensor": true
> ```
> 详见 [references/config_guide.md](./references/config_guide.md)。

**2.1 OP报错**：对比 Before/After IR，确认是否误报。

动态shape场景：IR显示符号变量（如 `sym_15_dim_0`）→ 参考 [references/print_npu_data.md](./references/print_npu_data.md) 打印验证。

**2.2 精度问题**：

```
配置PreCheck/PostCheck → 编译运行 → 观察日志报错
    ├─ 有报错 → 终止，告知用户
    └─ 无报错 → pass_compare.py对比
        ├─ 能定位 → 解决
        └─ 无法定位 → 上板比对
```

**步骤详解**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | PreCheck/PostCheck | 打开 `tile_fwk_config.json` 对应Pass开关 |
| 2 | pass_compare.py | 对比失败Pass与前置Pass的OP节点 |
| 3 | 上板比对 | 打印前端pypto输出 vs 精度工具Pass输出 |

```bash
python3 tools/verifier/pass_compare.py --p <FailedPass> <GoldenPass> --verify_path=/path/to/verify_data
```

> 上板比对：若精度工具保存的Pass输出与上板结果一致 → 问题在该Pass。

### 情况三：无报错但精度异常 → 调用 `pypto-precision-compare`

定位问题Op后，如需打印上板数据验证 → 参考 [references/print_npu_data.md](./references/print_npu_data.md)

---

## 打印上板信息

**前置条件**：已通过 `pypto-precision-compare` 定位到具体Op。

用于：打印上板tensor数据、验证动态shape/offset值。

详细方法请参考：**[references/print_npu_data.md](./references/print_npu_data.md)**

### 可打印内容

| 内容 | 方法 | 说明 |
|-----|------|------|
| GM tensor数据 | `AiCorePrintGmTensor` | DDR/GM上的tensor |
| UB tensor数据 | `AiCorePrintUbTensor` | UB上的tensor |
| Shape变量值 | `AicoreLogF` | 动态shape实际值 |
| Offset值 | `AicoreLogF` | 动态offset实际值 |

### 脚本工具

```bash
# 初始化配置
python3 scripts/print_npu_data.py --init --work-path /path/to/work

# 列出CCE文件
python3 scripts/print_npu_data.py --work-path /path/to/work --list-cce

# 打印tensor数据
python3 scripts/print_npu_data.py --work-path /path/to/work --print-idx 0 --tensor gmTensor_4

# 打印shape值
python3 scripts/print_npu_data.py --work-path /path/to/work --print-idx 0 --print-shape sym_15_dim_0
```

详见：[scripts/print_npu_data.py](./scripts/print_npu_data.py)

---

## IR图分析

判断误报、辅助定位。详见 `pypto/.agents/skills/pypto-pass-error-locator/references/ir-analysis-guide.md`

```bash
# 查询OP详情
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file <IR文件> --op-magic <ID>

# 列出所有OP
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file <IR文件> --list-ops
```

---

## 常见错误案例

查阅 **[references/error_cases.md](./references/error_cases.md)**：
- 案例01：Reshape报错 → 添加 `+ 0.0` 规避
- 案例02：精度通过但验证报错 → 以精度结果为准

---

## 注意事项

1. Pass精度判断：只看 CodegenPreproc 是否通过
2. tensor_graph FAIL → 调用 pypto-precision-compare
3. 无报错但精度异常 → 调用 pypto-precision-compare
4. 动态shape验证：参考 references/print_npu_data.md
5. 打印配置：`fixed_output_path=true`, `force_overwrite=false`
6. 打印限制：元素数量 ≤ 80

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [references/config_guide.md](./references/config_guide.md) | 配置说明 |
| [references/print_npu_data.md](./references/print_npu_data.md) | 打印上板信息指南 |
| [references/error_cases.md](./references/error_cases.md) | 常见错误案例 |
| [scripts/print_npu_data.py](./scripts/print_npu_data.py) | 打印上板信息脚本 |