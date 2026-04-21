# Binary CCE 二分定位指南

本指南用于通过二分打印法定位 CCE 中哪个 Op 出现精度问题，帮助获取真实上板数据（DDR/GM 和 UB）。

---

## 目录

1. [概述](#概述)
2. [打印方法概述](#打印方法概述)
3. [前置配置](#前置配置)
4. [执行流程](#执行流程)
5. [场景判断](#场景判断)
6. [多CCE场景处理](#多cce场景处理)
7. [单CCE场景处理](#单cce场景处理)
8. [精确定位具体Op的方法](#精确定位具体op的方法)
9. [打印语句添加后的确认环节](#打印语句添加后的确认环节)
10. [打印不出来上板数据时的诊断检查](#打印不出来上板数据时的诊断检查)
11. [Shape打印与ValidShape问题诊断](#shape打印与validshape问题诊断)
12. [脚本工具使用](#脚本工具使用)

---

## 概述

当问题定位到 Codegen/Machine 阶段时，使用二分法或目标定位法定位具体出错的 Op。

**定位原理**：
1. 找到所有 CCE 文件（按生成顺序排列）
2. 在某个 CCE 中添加打印语句，打印其中的 tensor 数据
3. 运行测试，对比打印数据与 golden
4. 通过二分法不断缩小范围，最终定位到出错的 Op

---

## 打印方法概述

### 四种打印方法总览

CCE 打印方法分为四类，用于获取不同类型的数据：

| 打印类型 | 函数名称 | 适用场景 | 内存位置 |
|---------|---------|---------|---------|
| **GM数据打印** | `AiCorePrintGmTensor` | 打印DDR/GM上的tensor数据（最常用） | 全局内存（DDR） |
| **UB数据打印** | `AiCorePrintUbTensor` | 打印UB上的tensor数据 | 统一缓冲区（UB） |
| **Shape打印** | `AiCorePrintShape` | 打印shape调试信息 | 参数/符号信息 |
| **Offset打印** | `AiCorePrintShape` | 打印offset调试信息 | 参数/符号信息 |

### 打印方法选择指南

```
需要打印什么数据？
        ↓
┌───────────────────────────────────────────────┐
│                                               │
│ tensor数据（GM/DDR） → AiCorePrintGmTensor     │
│   • 输入/输出tensor验证                        │
│   • 大数据量打印                               │
│                                               │
│ tensor数据（UB） → AiCorePrintUbTensor         │
│   • UB内部计算验证                             │
│   • CopyIn/CopyOut中间数据                    │
│                                               │
│ shape信息 → AiCorePrintShape                   │
│   • 动态shape验证                              │
│   • validshape问题定位                         │
│                                               │
│ offset信息 → AiCorePrintShape                  │
│   • offset/dynoffset验证                       │
│   • 越界访问诊断                               │
│                                               │
└───────────────────────────────────────────────┘
```

---

### GM数据打印（AiCorePrintGmTensor）

**用途**：打印全局内存（GM/DDR）上的 tensor 数据，是最常用的打印方法。

**函数原型**：
```cpp
AiCorePrintGmTensor(param->ctx, (__gm__[dtype]*)[tensor名].GetAddr(), [末尾偏移量], [起始偏移量]);
```

**参数说明**：

| 参数 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| `param->ctx` | context | AiCore上下文 | `param->ctx` |
| `(__gm__[dtype]*)` | 指针类型 | GM内存指针类型 | `(__gm__float*)`、`(__gm__bfloat16_t*)` |
| `[tensor名].GetAddr()` | 地址 | tensor数据地址 | `gmTensor_output.GetAddr()` |
| `[末尾偏移量]` | int | 打印结束位置（含） | 63（打印第0~63个元素） |
| `[起始偏移量]` | int | 打印起始位置 | 0 |

> **元素数量计算**：元素数量 = 末尾偏移量 - 起始偏移量 + 1
> 例如：`末尾偏移=63, 起始偏移=0` → 打印64个元素

**使用示例**：
```cpp
#include "tilefwk/aicore_print.h"

// 打印GM上的float数据（64个元素，从第0个开始）
AiCorePrintGmTensor(param->ctx, (__gm__float*)gmTensor_output.GetAddr(), 63, 0);

// 打印GM上的bfloat16数据（80个元素，从第0个开始）
AiCorePrintGmTensor(param->ctx, (__gm__bfloat16_t*)gmTensor_input.GetAddr(), 79, 0);

// 打印GM上的数据（从第10到第73，共64个元素）
AiCorePrintGmTensor(param->ctx, (__gm__float*)gmTensor_temp.GetAddr(), 73, 10);
```

**常见dtype类型对照表**：

| C++类型 | PyPTO DataType | 说明 |
|--------|----------------|------|
| `float` | DT_FP32 | 单精度浮点 |
| `bfloat16_t` | DT_BF16 | Brain Float16 |
| `half` | DT_FP16 | 半精度浮点 |
| `int32_t` | DT_INT32 | 32位整数 |

---

### UB数据打印（AiCorePrintUbTensor）

**用途**：打印统一缓冲区（UB）上的 tensor 数据，用于验证UB内部计算。

**函数原型**：
```cpp
AiCorePrintUbTensor(param->ctx, (__ub__[dtype]*)[tensor名].GetAddr(), [末尾偏移量], [起始偏移量]);
```

**适用场景**：
- 验证UB内部数据是否正确
- 定位 CopyIn/CopyOut 过程中的问题
- 检查 UB 计算中间结果

**使用示例**：
```cpp
#include "tilefwk/aicore_print.h"

// 打印UB上的float数据（64个元素，从第0个开始）
AiCorePrintUbTensor(param->ctx, (__ub__float*)ubTensor_temp.GetAddr(), 63, 0);

// 打印UB上的bfloat16数据（80个元素，从第0个开始）
AiCorePrintUbTensor(param->ctx, (__ub__bfloat16_t*)ubTensor_calc.GetAddr(), 79, 0);
```

---

### Shape打印（AiCorePrintShape）

**用途**：打印 shape 调试信息，用于验证动态shape计算。

**函数原型**：
```cpp
AiCorePrintShape(param->ctx, ShapeXDim(sym_XX_dim_0, sym_XX_dim_1, ...));
```

**参数说明**：
- `ShapeXDim`：打印的维度数量模板
  - `Shape2Dim(...)`：打印2维shape
  - `Shape3Dim(...)`：打印3维shape
- `sym_XX_dim_Y`：shape变量名（从CCE中查找）

**适用场景**：
- 验证动态shape计算是否正确
- 检查validshape与实际shape是否匹配
- 定位越界访问问题

**使用示例**：
```cpp
#include "tilefwk/aicore_print.h"

// 打印2维shape
AiCorePrintShape(param->ctx, Shape2Dim(sym_15_dim_0, sym_15_dim_1));

// 打印3维shape
AiCorePrintShape(param->ctx, Shape3Dim(sym_10_dim_0, sym_10_dim_1, sym_10_dim_2));

// 打印4维shape
AiCorePrintShape(param->ctx, Shape4Dim(sym_20_dim_0, sym_20_dim_1, sym_20_dim_2, sym_20_dim_3));
```

---

### Offset打印（AiCorePrintShape - Coord2Dim）

**用途**：打印 offset 调试信息，用于验证offset/dynoffset配置。

**函数原型**：
```cpp
AiCorePrintShape(param->ctx, CoordXDim(offset_var0, offset_var1, ...));
```

**参数说明**：
- `CoordXDim`：打印的offset数量模板
  - `Coord2Dim(...)`：打印2个offset值
  - `Coord3Dim(...)`：打印3个offset值
- `offset_var`：offset变量，通常使用 `RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST` 宏

**适用场景**：
- 验证offset/dynoffset配置是否正确
- 定位内存访问越界问题
- 检查dynvalidshape与offset关系

**使用示例**：
```cpp
#include "tilefwk/aicore_print.h"

// 打印2个offset值
AiCorePrintShape(param->ctx, Coord2Dim(
    (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 19, 0)),
    (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 1, 1))
));

// 打印3个offset值
AiCorePrintShape(param->ctx, Coord3Dim(
    (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 10, 0)),
    (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 11, 0)),
    (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 12, 0))
));
```

**适用场景**：
- 验证UB内部数据是否正确
- 定位 CopyIn/CopyOut 过程中的问题
- 检查 UB 计算中间结果

**使用示例**：
```cpp
#include "tilefwk/aicore_print.h"

// 打印UB上的float数据
AiCorePrintUbTensor(param->ctx, (__ub__float*)ubTensor_temp.GetAddr(), 64, 0);

// 打印UB上的bfloat16数据
AiCorePrintUbTensor(param->ctx, (__ub__bfloat16_t*)ubTensor_calc.GetAddr(), 80, 1);
```

---

### 打印注意事项

1. **元素数量限制**：打印元素数量 = 末尾偏移量 - 起始偏移量 + 1，**不能超过80**
2. **打印时机**：打印必须在 Op 执行**之后**，否则数据未写入内存
3. **偏移量设置**：多次打印不同范围数据时，设置不同的起始/末尾偏移量
4. **头文件**：必须引入 `#include "tilefwk/aicore_print.h"`
5. **日志位置**：打印数据出现在 `$ASCEND_WORK_PATH/log/debug/device-*/device-*.log`
6. **Shape/Offset函数**：使用 `AiCorePrintShape`，注意参数格式为模板函数调用

## 前置配置

在使用二分 CCE 方法前，需要确保以下配置：

### 1. 配置 tile_fwk_config.json

修改 `framework/src/interface/configs/tile_fwk_config.json`：

```json
"codegen": {
    "fixed_output_path": true,
    "force_overwrite": false,
    "parallel_compile": 1
}
```

| 配置项 | 正确值 | 说明 |
|-------|-------|------|
| `fixed_output_path` | `true` | 固定CCE输出路径，生成在 `./kernel_aicore/` |
| `force_overwrite` | `false` | 不覆盖已修改的CCE文件 |
| `parallel_compile` | `1` | 单线程编译，便于调试 |

### 2. 配置打印开关

确保 `framework/src/interface/machine/device/tilefwk/aicore_print.h` 中：

```c
#define ENABLE_AICORE_PRINT 1   // 必须为 1
```

---

## 执行流程

### 标准执行步骤

| 步骤 | 操作 | 命令 |
|------|------|------|
| 1 | 配置固定输出路径 | 修改 `tile_fwk_config.json` |
| 2 | 编译安装 | `python3 -m pip install . -v` |
| 3 | 运行用例一次 | 生成 CCE 文件 |
| 4 | 在 CCE 中添加打印 | 修改 `./kernel_aicore/*.cpp` |
| 5 | 再次运行用例 | 获取打印数据 |
| 6 | 查看对应日志 | 根据场景选择日志位置 |

### CCE 文件生成位置

配置 `fixed_output_path: true` 后，CCE 文件固定生成在：

```
./kernel_aicore/
├── TENSOR_xxx_xxx_0_aiv.cpp
├── TENSOR_xxx_xxx_1_aiv.cpp
└── ...
```

**注意**：
- 未配置时，CCE 每次生成在新的 `output/output_*/kernel_aicore/` 目录
- 配置后，CCE 固定在 `./kernel_aicore/`，修改后再次运行不会被覆盖

---

## 场景判断

### 日志位置说明

| 场景 | 日志位置 | 查看内容 |
|-----|---------|---------|
| **精度工具验证** | `output/output_*/verify_*/verify_graph_result_brief.log` | Pass验证结果、EXCEPTION报错、op信息 |
| **二分CCE调试** | `$ASCEND_WORK_PATH/log/debug/device-*/device-*.log` | AiCorePrint打印的上板tensor数据 |

### 场景判断标准

| 判断条件 | 多CCE场景 | 单CCE场景 |
|---------|-----------|-----------|
| CCE文件数量 | > 1个 | = 1个 |
| 函数复杂度 | 包含多个子图 | 单一计算图 |
| 问题范围 | 跨多个CCE | 单个CCE内 |
| 推荐方法 | 方式一/二 | 方式三/四 |

### 场景判断方法

```bash
# 1. 统计CCE文件数量
cce_count=$(ls kernel_aicore/*.cpp 2>/dev/null | wc -l)
echo "CCE文件数量: $cce_count"

# 2. 确定场景
if [ "$cce_count" -gt 1 ]; then
    echo "当前场景: 多CCE场景"
    echo "推荐方法: 方式一（目标定位）或 方式二（二分搜索）"
elif [ "$cce_count" -eq 1 ]; then
    echo "当前场景: 单CCE场景"
    echo "推荐方法: 方式三（单CCE直接二分）或 方式四（手动修改CCE）"
else
    echo "当前场景: 中等规模CCE场景"
    echo "推荐方法: 根据问题复杂度选择方式一或方式三"
fi
```

### 决策树

```
开始二分CCE
       ↓
判断CCE场景
       ↓
  ┌─────────────────────────┐
  │                         │
多CCE场景               单CCE场景
  │                         │
  ↓                         ↓
能否直接定位？        方式三：单CCE直接二分
  │                         │
  是                        方式四：手动修改CCE
  ↓                         │
方式一：目标定位             │
  ↓                         │
方式二：二分搜索             │
  │                         │
  └─────────────────────────┘
```

---

## 多CCE场景处理

### 方式一：目标定位（推荐多CCE场景）

根据错误日志或输出分析，定位可疑的 CCE 文件，直接打印验证：

#### 步骤1：分析可疑 CCE

- 查看验证日志中的计算图名（如 `TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11`）
- 根据输出差异的 pattern 推测可能出问题的算子
- 检查 IR 图中的异常操作

#### 步骤2：找到对应 CCE 文件

```bash
# 根据函数名搜索
ls kernel_aicore/*TENSOR_Loop_S_Unroll1*.cpp

# 根据 funcHash 搜索
grep -l "funcHash: 11899060959657268680" kernel_aicore/*.cpp

# 根据操作类型搜索
grep -l "TILE_MATMUL" kernel_aicore/*.cpp
```

#### 步骤3：手动添加打印

```cpp
#include "tilefwk/aicore_print.h"

// 在 kernel 函数对应的位置打印输入
AiCorePrintGmTensor(param->ctx, (__gm__bfloat16_t*)gmTensor_X.GetAddr(), element_count, 0);
```

#### 步骤4：运行验证

```bash
python3 your_test.py
# 查看 $ASCEND_WORK_PATH/log/debug/device-*/device-*.log 获取打印结果
```

---

### 方式二：二分搜索（多CCE场景）

当存在多个 CCE 文件，且无法直接定位可疑文件时，推荐**先使用 pypto-precision-compare 二分前端**。

#### 第一步：先用 pypto-precision-compare 定位问题范围

调用 `pypto-precision-compare` 技能，在前端代码中插入检查点，二分定位问题出现在哪个检查点区间。

#### 第二步：在问题范围内的 CCE 文件中二分

```bash
# 列出所有 CCE 文件，确认问题范围
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/output/output_latest/ \
    --list-cce
```

找到问题范围对应的 CCE 索引（如 20-50），然后在该范围内二分：

```bash
# 从范围中间开始打印
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/output/output_latest/ \
    --print-idx 35 \
    --pos kernel_start \
    --rebuild
```

#### 第三步：根据结果判断方向

运行后检查输出：
- 输出正常或差异变小 → 问题在后半部分 → 继续打印后半区间的 CCE
- 输出仍然异常 → 问题在前半部分 → 继续打印前半区间的 CCE

重复二分，逐渐缩小范围。

---

## 单CCE场景处理

### 方式三：单CCE直接二分

当确定问题在单个 CCE 文件中时，直接在该 CCE 内二分查找具体 Op：

#### 第一步：初始化配置（仅需一次）

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --init \
    --work-path /path/to/output/output_latest/
```

#### 第二步：列出CCE文件信息

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/output/output_latest/ \
    --list-cce
```

输出示例：
```
找到 1 个 CCE 文件:
  [0] TENSOR_Loop_S_TND_Unroll1_PATH0_xxx.cpp
  ...
```

#### 第三步：选择打印位置

基于以下原则选择：
- 如果知道大概范围（如 0-93），从中间开始
- 如果完全不确定，从约 1/3 位置开始

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/output/output_latest/ \
    --print-idx 60 \
    --pos kernel_start \
    --rebuild
```

---

### 方式四：手动修改 CCE（高精度定位）

当二分定位到单个 CCE 后，需要精确定位具体 Op：

#### 1. 查看 CCE 内容

```bash
# 查看 CCE 文件
cat kernel_aicore/XXX.cpp

# 查看CCE中的关键操作
grep -n "TILE_MATMUL\|TILE_ADD\|TILE_MUL" kernel_aicore/XXX.cpp
```

#### 2. 根据IR/错误日志信息定位CCE中的具体操作

参见下节：[精确定位具体Op的方法](#精确定位具体op的方法)

#### 3. 在可疑操作前后添加打印

> **打印方法详解**：参见 [打印方法概述](#打印方法概述)

```cpp
#include "tilefwk/aicore_print.h"

// 在目标操作之前打印输入tensor（GM）
AiCorePrintGmTensor(param->ctx, (__gm__float*)inputTensor.GetAddr(), 64, 0);

// 执行目标操作
...

// 在目标操作之后打印输出tensor（GM）
AiCorePrintGmTensor(param->ctx, (__gm__float*)outputTensor.GetAddr(), 64, 1);
```

**打印位置选择建议**：
- 输入验证：在操作前打印输入数据，确认输入是否正确
- 输出验证：在操作后打印输出数据，确认输出是否正确
- 双向验证：前后都打印，对比分析数据变化

---

## 精确定位具体Op的方法

当需要精确定位到 CCE 文件中某个具体 Op 时，可以通过以下方法：

### 方法一：根据 param 索引定位

从错误日志或 IR 中提取 param 索引信息，直接搜索：

```bash
# 错误日志中 tensor索引 %23@485#(41) 对应 GET_PARAM_ADDR(param, 41, 485)
grep -n "GET_PARAM_ADDR(param, 41" kernel_aicore/*.cpp
grep -n "param, 40, 378" kernel_aicore/*.cpp
```

### 方法二：根据 Op 类型关键字定位

| IR中的Op类型 | CCE中的关键字 | 搜索命令 |
|-------------|--------------|----------|
| TILE_RESHAPE | `DynReshapeCopyIn\|DynReshapeCopyOut` | `grep -n "DynReshape" *.cpp` |
| TILE_MATMUL | `Matmul` | `grep -n "Matmul" *.cpp` |
| TILE_ADD | `Add\|TAdd` | `grep -n "Add" *.cpp` |
| TILE_LOAD | `TLoad\|CopyIn` | `grep -n "TLoad" *.cpp` |
| TILE_STORE | `TStore\|CopyOut` | `grep -n "TStore" *.cpp` |

### 方法三：根据相邻 Op 关系辅助定位（推荐）

**当直接搜索不到某个 Op 时**，可以通过相邻 Op 关系辅助定位：

#### 步骤一：从 IR 分析获取相邻 Op 信息

```bash
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName.tifwkgr \
    --op-magic 10115 \
    --list-ops
```

#### 步骤二：在 CCE 中搜索相邻 Op

IR 中 Op 的执行顺序与 CCE 中的代码顺序相对应：

```bash
# 假设问题Op前一个Op是TILE_LOAD，后一个Op是TILE_STORE
grep -n "TLoad" kernel_aicore/target.cpp    # 前Op
grep -n "TStore" kernel_aicore/target.cpp   # 后Op

# 问题Op就在这两个相邻Op之间的代码行
```

#### 步骤三：分析相邻关系定位目标 Op

```
IR中的Op顺序：
  TILE_LOAD (op_id=10010)  → 前Op
  TILE_RESHAPE (op_id=10115) → 目标Op（搜索不到）
  TILE_STORE (op_id=10020)  → 后Op

CCE中的代码顺序：
  第315行: TLoad(...)        → 对应前Op
  第319行: DynReshapeCopyOut → 对应目标Op（在前后Op之间）
  第325行: TStore(...)       → 对应后Op
```

### 方法四：结合 IR 数据流分析定位

```bash
# 分析IR文件中的数据流
python3 .agents/skills/pypto-pass-error-locator/scripts/computation_graph_analyzer.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName.tifwkgr

# 根据tensor名称在CCE中搜索
grep -n "gmTensor_1\|ubTensor_0" kernel_aicore/*.cpp
```

### 定位示例

假设错误日志显示：
- func hash: 1865806080813139654
- op magic: 10115 (RESHAPE)
- param索引: 41, 400

```bash
# 1. 找到对应的CCE文件
grep -l "funcHash: 1865806080813139654" kernel_aicore/*.cpp

# 2. 在CCE中搜索reshape操作
grep -n "DynReshapeCopyIn\|DynReshapeCopyOut" TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11*.cpp

# 3. 定位到具体行（如第322行）
# TileOp::DynReshapeCopyIn<float, 1, 1, 52, 32>((__ubuf__ float*)UB_S6656_E13312, (__gm__ float*)GET_PARAM_ADDR(param, 41, 400), ...)

# 4. 如果直接搜索不到，通过相邻Op定位
grep -n "TLoad" TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11*.cpp
grep -n "TStore" TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11*.cpp

# 5. 在前后Op之间定位问题Op
sed -n '315,325p' TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11*.cpp
```

---

## 打印语句添加后的确认环节

**⚠️ 重要：在配置完成并添加打印语句后，必须向用户展示配置和打印语句供确认。**

> **打印方法详解**：参见 [打印方法概述](#打印方法概述)

### 确认输出模板

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

**文件**: `./kernel_aicore/[目标CCE文件名].cpp`

```cpp
// 第 [行号] 行添加的打印语句：
#include "tilefwk/aicore_print.h"

// GM数据打印（末尾偏移量, 起始偏移量）
AiCorePrintGmTensor(param->ctx, (__gm__[dtype]*)[tensor名].GetAddr(), [末尾偏移量], [起始偏移量]);

// 或 UB数据打印
AiCorePrintUbTensor(param->ctx, (__ub__[dtype]*)[tensor名].GetAddr(), [末尾偏移量], [起始偏移量]);

// 或 Shape打印
AiCorePrintShape(param->ctx, Shape2Dim(sym_XX_dim_0, sym_XX_dim_1));

// 或 Offset打印
AiCorePrintShape(param->ctx, Coord2Dim(...));
```

**打印参数说明**（详见[打印方法概述](#打印方法概述)）：
- Tensor名称: [tensor名]
- 数据类型(dtype): float / bfloat16_t / half / int32_t
- 偏移量范围: [末尾偏移量], [起始偏移量]（元素数量 = 末尾 - 起始 + 1 ≤ 80）
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

---

## 打印不出来上板数据时的诊断检查

**当打印语句添加后但日志中找不到上板数据时**，必须按以下步骤展示配置和打印语句供用户确认：

### 诊断输出模板

```markdown
## 问题诊断：配置与打印语句检查

### 1. aicore_print.h 关键配置
// 文件: framework/src/interface/machine/device/tilefwk/aicore_print.h
// 第28行 - 打印功能开关（必须为1）
#define ENABLE_AICORE_PRINT 1   // ✅ 正确 / ❌ 错误

### 2. tile_fwk_config.json 关键配置
{
  "codegen": {
    "parallel_compile": [当前值],    // ❌ 应为 1
    "fixed_output_path": [当前值],   // ❌ 应为 true
    "force_overwrite": [当前值]      // ❌ 应为 false
  }
}

### 3. CCE打印语句
- 头文件引入: [已引入/缺失]
- 打印位置: [行号]
- 偏移量范围: [末尾偏移量], [起始偏移量]

### 4. 问题总结
| 配置项 | 当前值 | 正确值 | 状态 |
|--------|--------|--------|------|
| ENABLE_AICORE_PRINT | [值] | 1 | ✅/❌ |
| fixed_output_path | [值] | true | ✅/❌ |
| force_overwrite | [值] | false | ✅/❌ |
| parallel_compile | [值] | 1 | ✅/❌ |

**是否需要修改配置并重新运行测试？**
```

### 常见问题及解决方案

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 日志无打印数据 | `ENABLE_AICORE_PRINT=0` | 改为 `1` |
| 打印数据不变化 | `force_overwrite=true` | 改为 `false` |
| 找不到CCE文件 | `fixed_output_path=false` | 改为 `true` |
| 打印数据错误 | 偏移量范围超出限制 | 调整偏移量（末尾-起始+1≤80） |
| 打印时机错误 | 打印在Op执行前 | 移到Op执行后 |

---

## Shape打印与ValidShape问题诊断

> **Shape打印方法详解**：参见 [打印方法概述](#打印方法概述)中的 **Shape打印（AiCorePrintShape）** 小节

### 何时使用 Shape 打印

当怀疑 validshape 有问题时，可以使用 shape 打印功能：

1. **数据异常但代码逻辑正确**：可能是 shape/validshape 不匹配导致
2. **越界访问错误**：可能是 validshape 设置错误
3. **数据维度不匹配**：需要确认运行时 shape 信息

### Shape 打印示例

在 CCE 文件中添加 shape 打印：

```cpp
#include "tilefwk/aicore_print.h"

// 打印2维shape
AiCorePrintShape(param->ctx, Shape2Dim(sym_15_dim_0, sym_15_dim_1));

// 打印3维shape
AiCorePrintShape(param->ctx, Shape3Dim(sym_10_dim_0, sym_10_dim_1, sym_10_dim_2));

// 打印offset值
AiCorePrintShape(param->ctx, Coord2Dim(
    (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 19, 0)),
    (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 1, 1))
));
```

或者使用脚本自动添加：

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/work \
    --print-idx 0 \
    --print-shape sym_15_dim_0,sym_15_dim_1
```

### ValidShape 问题检测

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/work \
    --ir-file path/to/ir_file.tifwkgr
```

检测内容包括：
- 输入 tensor 的 shape 和 validshape 是否一致
- offset/dynoffset 配置是否正确
- dynvalidshape 是否缺失

---

## 脚本工具使用

### 方式一：初始化配置（推荐先执行）

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --init \
    --work-path /path/to/work
```

这会：
1. 修改 `tile_fwk_config.json` 开启 `fixed_output_path` 和 `parallel_compile`
2. 修改 `aicore_print.h` 将 `ENABLE_AICORE_PRINT` 设为 1

### 方式二：列出CCE信息

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/work \
    --list-cce
```

### 方式三：指定CCE添加打印

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/work \
    --print-idx 0 \
    --tensor gmTensor_001
```

### 参数说明

| 参数 | 必填 | 说明 |
|-----|------|------|
| `--work-path` | 是 | ASCEND_WORK_PATH 工作目录 |
| `--pypto-root` | 否 | PyPTO源码根目录（默认当前目录） |
| `--init` | 否 | 仅初始化配置，不执行其他操作 |
| `--list-cce` | 否 | 仅列出CCE文件及其中包含的Op |
| `--print-idx` | 否 | 指定打印哪个CCE(0-based) |
| `--print-type` | 否 | 打印类型：GM 或 UB（默认GM） |
| `--tensor` | 否 | 指定要打印的tensor名称 |
| `--print-shape` | 否 | 打印 shape 变量（多个用逗号分隔） |

---

## 注意事项

> **打印方法注意事项**：参见 [打印方法概述](#打印方法概述)中的 **打印注意事项** 小节

1. **CCE 文件是 .cpp 格式**：每个 cpp 对应一张子图（kernel）
2. **打印会重新编译**：修改 cpp 后需要重新运行测试
3. **查看日志位置**：`$ASCEND_WORK_PATH/log/debug/device-*/device-*.log`
4. **恢复原文件**：调试完成后记得删除打印语句
5. **配置修改**：初始化时会将 `codegen.parallel_compile` 设为 1 以控制编译并发

---

## 相关文档

- 主流程文档：[SKILL.md](./SKILL.md)
- 配置指南：[config-guide.md](./config-guide.md)
- IR分析示例：[ir-analysis-examples.md](./ir-analysis-examples.md)