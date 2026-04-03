---
name: pypto-pass-precision-verify
description: 验证PyPTO Pass精度问题，定位精度问题出现在哪个Pass，并尝试修复
trigger: 验证Pass精度问题、pass verify、Pass精度调试、定位Pass精度问题、Pass精度报错、Pass精度不一致
---

## 简介

本技能用于验证 PyPTO Pass 侧的精度问题。当 Tensor Graph 校验通过但精度仍不对时，用于排查问题是否出现在 Pass 处理阶段。

**前置条件**：请先完成 SKILL.md 中的"精度校验流程"，确认 Tensor Graph 校验通过后再进入 Pass 校验。

## PyPTO 执行流程

```
前端代码书写 -> Function构图 -> Pass处理优化计算图 -> Codegen生成 -> Machine执行
     ↓              ↓              ↓                ↓           ↓
  Tensor Graph   Tensor Graph   Pass优化处理      代码生成      上板运行
     ↓              ↓              ↓                ↓           ↓
  前端代码问题    构图问题      Pass处理问题     Codegen问题   Machine问题
```

## 操作步骤

### 步骤一：查看验证数据

执行结束后，在最新的 `{work_path}/output/output_*/verify_*` 目录下生成验证数据：

```
├── verify_exception.log    # 验证异常日志
├── verify_result.csv       # 验证结果报告
├── tensor_graph/           # 前端初始计算图数据
│   ├── *.data
├── {FUNC}.pass_XX_NAME/   # 各 Pass 中间数据
│   └── *.data
├── tensor/                 # 人工保存的数据（如使用 pass_verify_save）
│   ├── *.data
│   └── *.csv
```

### 步骤二：分析验证结果与定位问题

#### 情况一：Pass 级别 Verify FAIL

Pass 验证失败分为两种类型：

**（1）具体 OP 报错**

| 项目 | 说明 |
|-----|------|
| **错误码** | `0xB200FU`（RUNTIME_EXCEPTION） |
| **日志特征** | `[operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB ...` 日志会输出出错 OP 的 `<shape/validshape>` 及属性信息 |
| **原因** | Operation 模拟执行失败，OP 上存在错误或缺失的属性 |
| **处理** | 查看对应 Pass 的 IR 图，分析该 OP 是否缺失属性或属性值不正确 |

**（2）精度问题**

| 项目 | 说明 |
|-----|------|
| **错误码** | `0xB4001U`（VERIFY_RESULT_MISMATCH） |
| **日志特征** | `[VERIFY]:ErrCode: FB4001! pass_06_SplitReshape Verify result FAILED` |
| **原因** | Pass 处理阶段引入的精度偏差 |

**Pass 阶段精度问题定位手段**：

| 方法 | 说明 |
|-----|------|
| **PreCheck / PostCheck** | 打开 `framework/src/interface/configs/tile_fwk_config.json` 中对应 Pass 的 PreCheck 及 PostCheck 开关，观察是否存在报错 |
| **pass_compare.py 脚本** | 对比失败 Pass 与前置 Pass 的每个 OP 节点，定位第一个出错的节点 |
| **上板结果比对** | 将 Machine 上板后的结果与 verify_result 中各 Pass 保存的输出逐一比对，第一个输出与上板输出一致的 Pass 即为出错的 Pass |

pass_compare.py 使用方法：

```bash
python3 pass_compare.py --p <FailedPass> <GoldenPass> --verify_path=/path/to/verify_data
```

- `--p` 参数后是对比的两个 Pass，空格隔开，前者为精度对比失败的 Pass，后者为作为 golden 的 Pass
- `--verify_path` 参数为精度工具 dump 数据文件目录的绝对路径
- 对比结果会生成类似 `verify_pass@SplitK@ExpandFunction@1773821696834386.csv` 的文件，记录每个 OP 节点的对比结果，未能匹配的节点会标注 skip

#### 情况二：所有验证都 PASS 但仍有精度问题

| 项目 | 说明 |
|-----|------|
| **错误码** | 无报错，但上板结果与 golden 不一致 |
| **原因** | 问题可能在 Codegen 或 Machine 执行阶段，也可能 Pass 侧未能覆盖校验 |
| **处理** | 参考 SKILL.md 中"情况 C"，使用二分对比方法进行上板二分定位 |

#### 其他错误码

| 错误码 | 名称 | 说明 |
|-------|------|------|
| `0xB0001U` | VERIFY_NOT_ENABLE | PyTorch 版本不支持，请检查本地 `torch >= 2.1.0` |
| 其他 | — | 通常是 PyPTO 内部缺陷导致，请在社区联系开发人员解决 |

## Pass 常见错误类型及修复

### 错误1：ValidShape 不匹配

| 项目 | 说明 |
|-----|------|
| **现象** | Pass 验证时报 ValidShape 错误 |
| **原因** | Pass 处理过程中 shape 计算错误 |
| **修复** | 检查该 Pass 的 shape 推导逻辑；检查 tile shapes 设置是否正确 |

### 错误2：Offset 越界

| 项目 | 说明 |
|-----|------|
| **现象** | Pass 验证时报 Offset 错误 |
| **原因** | Pass 处理过程中内存偏移计算错误 |
| **修复** | 检查该 Pass 的索引计算逻辑；检查 tile 切分是否正确 |

### 错误3：数据类型精度损失

| 项目 | 说明 |
|-----|------|
| **现象** | float16/bfloat16 计算结果与 float32 差异大 |
| **原因** | Pass 处理过程中类型转换不当或低精度溢出 |
| **修复** | 检查 Pass 是否引入了不必要的类型转换；检查中间计算是否需要提升精度 |

### 错误4：累积误差

| 项目 | 说明 |
|-----|------|
| **现象** | 长计算路径的算子误差大 |
| **原因** | Pass 优化后计算路径变化，多次计算累积舍入误差 |
| **修复** | 检查 Pass 是否改变了计算顺序；考虑使用更高精度中间变量 |

### 错误5：边界值处理

| 项目 | 说明 |
|-----|------|
| **现象** | 特定边界值（0、inf、nan）结果错误 |
| **原因** | Pass 优化后边界条件判断或处理不当 |
| **修复** | 检查 Pass 是否正确处理了边界值场景；确认 inf/nan 传播逻辑 |
