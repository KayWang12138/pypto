---
name: pypto-pass-precision-verify
description: 验证PyPTO Pass精度问题，定位精度问题出现在哪个Pass，并尝试修复
trigger: 验证Pass精度问题、pass verify、Pass精度调试、定位Pass精度问题、Pass精度报错、Pass精度不一致
---

## 简介

本技能用于验证 PyPTO Pass 侧的精度问题。当 PyPTO 算子上板执行后输出数据与 torch 输出不一致时，用于排查问题是否出现在 Pass 处理阶段。

> **前端问题处理**：若日志显示 `tensor_graph Verify FAIL`，说明问题在前端代码，请直接调用 `pypto-precision-compare` 技能进行定位。

## PyPTO 执行流程

```
前端代码书写 -> Function构图 -> Pass处理优化计算图 -> Codegen生成 -> Machine执行
     ↓              ↓              ↓                ↓           ↓
  Tensor Graph   Tensor Graph   Pass优化处理      代码生成      上板运行
     ↓              ↓              ↓                ↓           ↓
  前端代码问题    构图问题      Pass处理问题     Codegen问题   Machine问题
```

## 操作步骤

### 步骤一：配置校验开关

在 PyPTO 算子实现文件中，配置 `verify_options`：

```python
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,  # 保存中间tensor数据用于分析
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
) -> pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    return input0 + input1
```

设置 Golden 数据（**必须在算子运行前设置**）：

```python
# 计算 golden
torch_output = torch.add(input_data0, input_data1)

# 设置 golden（必须放在 pypto 算子运行前设置！）
pypto.set_verify_golden_data(goldens=[None, None, torch_output])

# 执行 pypto 算子
pypto_output = your_kernel(input_data0, input_data1)
```

**关键注意事项**：
- `pypto.set_verify_golden_data()` 必须在 `your_kernel()` 调用**之前**执行
- 如果放在 pypto 算子运行之后设置，仅能校验 Pass 阶段，无法校验 Tensor Graph 阶段
- goldens 列表顺序对应算子的输入输出数量：`[input0, input1, ..., output]`
- goldens 列表中的输入项不参与校验，统一设置为 `None`

### 步骤二：编译并运行

```bash
# 编译安装（加 --no-build-isolation）
python3 -m pip install . --verbose --no-build-isolation

# 运行测试
python3 your_test_case.py
```

### 步骤三：分析验证结果与定位问题

运行后会打印验证结果，错误码统一定义于 `framework/src/interface/interpreter/verify_error.h` 与 `framework/src/interface/interpreter/calculator/calc_error.h` 文件。

执行结束后，在 `{work_path}/output/output_*/` 目录下生成验证数据：

```
├── tensor_graph/           # 前端初始计算图数据
│   ├── *.data
├── verify_result.csv       # 验证结果报告
├── {FUNC}.pass_XX_NAME/   # 各 Pass 中间数据
│   └── *.data
├── tensor/                 # 人工保存的数据（如使用 pass_verify_save）
│   ├── *.data
│   └── *.csv
```

根据日志中的错误码和验证阶段，按以下三种情况处理：

#### 情况一：tensor_graph Verify FAIL（前端问题）

| 项目 | 说明 |
|-----|------|
| **错误码** | `0xB4001U`（VERIFY_RESULT_MISMATCH） |
| **日志特征** | `[VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED` |
| **原因** | 前端代码书写问题 |
| **处理** | 调用 `pypto-precision-compare` 技能，发送指令："使用 pypto-precision-compare 技能，定位 xxx.py 的精度问题" |

#### 情况二：Pass 级别 Verify FAIL

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

#### 情况三：所有验证都 PASS 但仍有精度问题

| 项目 | 说明 |
|-----|------|
| **错误码** | 无报错，但上板结果与 golden 不一致 |
| **原因** | 问题可能在 Codegen 或 Machine 执行阶段，也可能 Pass 侧未能覆盖校验 |
| **处理** | 1. 先回顾常见 Pass 错误类型，逐一排除<br>2. 若仍未定位，向用户确认是否继续排查，并提供两种二分定位方式供选择（默认推荐二分前端）<br>3. **二分前端**：调用 `pypto-precision-compare` 技能，自动插入检查点定位出错的代码行<br>4. **二分 CCE**：通过二分 CCE 中间产物，定位出错的 Op |

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

