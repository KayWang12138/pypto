# PyPTO Pass 精度验证 - IR分析实战示例

本文档提供 IR 分析的实战示例，帮助用户快速定位 Pass 精度问题。

---

## 示例1：操作丢失问题

### 场景描述

Pass验证报错，IR对比显示 After 中缺少 TILE_ADD 操作。

### IR对比结果

```bash
# Before IR
!10003 TILE_ADD(g:-1, s:-1) %6@10#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR, %8@12#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR

# After IR - 缺少TILE_ADD操作
# !10003 TILE_ADD 操作不存在
```

### 问题定位步骤

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 确认操作丢失 | 对比 Before 和 After IR 的操作 ID |
| 2 | 查找影响 Pass | 检查 Pass 日志，找到删除该操作的 Pass |
| 3 | 定位对应 CCE | 根据函数名查找 CCE 文件 |

### 解决方法

```bash
# 1. 找到包含TILE_ADD的CCE文件
cce_file=$(grep -l "TILE_ADD" output/output_*/kernel_aicore/*.cpp 2>/dev/null | head -1)
echo "问题CCE: $cce_file"

# 2. 在CCE中添加打印验证
# 在TILE_ADD操作前后添加打印

# 3. 使用IR分析工具对比
python3 .agents/skills/pypto-pass-error-locator/scripts/computation_graph_analyzer.py \
    --before-file output/output_*/Pass_XX_Name/Before_XX_PassName.tifwkgr \
    --after-file output/output_*/Pass_XX_Name/After_XX_PassName.tifwkgr \
    --output-dir ir_analysis_result
```

### 预期结果

- IR 分析报告显示操作丢失
- 定位到删除操作的 Pass
- 修复后重新验证通过

---

## 示例2：Shape 不匹配问题

### 场景描述

Pass验证报 shape 错误，IR分析显示 shape 计算错误。

### IR分析结果

```bash
# Before IR
<16 x 128 x DT_FP32 / 16 x 128 x DT_FP32> %6@10#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR

# After IR - shape不匹配
<8 x 128 x DT_FP32 / 8 x 128 x DT_FP32> %6@10#(-1)MEM_DEVICE_DDR::MEM_DEVICE_DDR
# 注意：shape从 16x128 变成了 8x128
```

### 问题定位步骤

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 确认 shape 不匹配 | 对比 Before 和 After IR 的 shape |
| 2 | 分析 shape 变化原因 | 检查 shape 计算相关操作 |
| 3 | 定位对应 CCE | 根据 shape 变化定位 CCE |

### 解决方法

```bash
# 1. 使用 get_op_info.py 查询报错 OP 的详细信息
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --op-magic <报错的OP_ID> \
    --format json

# 2. 对比 Before 和 After IR 中的 OP 信息
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/Before_XX_PassName_funcname.tifwkgr \
    --op-magic <报错的OP_ID> \
    --format json > before_op.json

python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_XX_Name/After_XX_PassName_funcname.tifwkgr \
    --op-magic <报错的OP_ID> \
    --format json > after_op.json

# 3. 比对两个 JSON 文件的差异
diff before_op.json after_op.json

# 4. 查看生成的分析报告
cat ir_analysis_result/analysis_report.txt

# 5. 在相关CCE中添加打印验证
```

### 预期结果

- JSON 对比显示 shape 字段差异
- 定位到修改 shape 的 Pass

---

## 示例3：Reshape 数据流分析（结合CCE定位）

### 场景描述

验证日志显示 reshape 操作 EXCEPTION，需要定位具体 CCE 代码行。

### 错误日志信息

```
[VERIFY:EXCEPTION:OP] Pass_28_InferParamIndex, ..., 10115, RESHAPE
func: TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_leaf41
funcHash: 1865806080813139654
<52 x 32 x DT_FP32 ...> %23@485#(41)MEM_DEVICE_DDR::MEM_DEVICE_DDR::IsDummy = !10115 TILE_RESHAPE ...
```

### 关键信息提取

| 字段 | 值 | 用途 |
|------|-----|------|
| `op_magic` | 10115 | RESHAPE 操作标识 |
| `funcHash` | 1865806080813139654 | 定位 CCE 文件 |
| `tensor索引` | %23@485#(41) | 定位 CCE 代码行 |
| `内存类型` | MEM_DEVICE_DDR | 确认打印方法 |

### 问题定位步骤

```bash
# 1. 根据 funcHash 找到对应 CCE 文件
grep -l "funcHash: 1865806080813139654" kernel_aicore/*.cpp

# 2. 在 CCE 中搜索 reshape 操作（IR中使用 TILE_RESHAPE，CCE使用 DynReshape）
grep -n "DynReshapeCopyIn\|DynReshapeCopyOut" TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11*.cpp

# 3. 如果直接搜索不到，通过相邻 Op 定位
# 从IR分析得知reshape前后是 TLoad 和 TStore
grep -n "TLoad" TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11*.cpp
grep -n "TStore" TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11*.cpp

# 4. 在前后 Op 之间定位问题 Op
sed -n '315,325p' TENSOR_Loop_S_Unroll1_PATH0_hiddenfunc0_11*.cpp
```

### 添加打印验证

```cpp
#include "tilefwk/aicore_print.h"

// reshape 输入（DynReshapeCopyOut 输出）
TileOp::DynReshapeCopyOut<float, 1, 4, 13, 32>(...);
AiCorePrintGmTensor(param->ctx, (__gm__ float*)GET_PARAM_ADDR(param, 40, 378), 80, 0);

// reshape 输出（DynReshapeCopyIn 输入）
AiCorePrintGmTensor(param->ctx, (__gm__ float*)GET_PARAM_ADDR(param, 38, 356), 80, 1);
TileOp::DynReshapeCopyIn<float, 1, 1, 52, 32>(...);
```

---

## IR分析与CCE定位对照表

| IR中的Op类型 | CCE中的关键字 | 搜索命令 |
|-------------|--------------|----------|
| TILE_RESHAPE | `DynReshapeCopyIn\|DynReshapeCopyOut` | `grep -n "DynReshape" *.cpp` |
| TILE_MATMUL | `Matmul` | `grep -n "Matmul" *.cpp` |
| TILE_ADD | `Add\|TAdd` | `grep -n "Add" *.cpp` |
| TILE_MUL | `Mul\|TMul` | `grep -n "Mul" *.cpp` |
| TILE_LOAD | `TLoad\|CopyIn` | `grep -n "TLoad" *.cpp` |
| TILE_STORE | `TStore\|CopyOut` | `grep -n "TStore" *.cpp` |
| TILE_EXP | `Exp` | `grep -n "Exp" *.cpp` |
| TILE_SIGMOID | `Sigmoid` | `grep -n "Sigmoid" *.cpp` |
| TILE_VIEW | `View` | `grep -n "View" *.cpp` |
| TILE_ASSEMBLE | `Assemble\|TAssemble` | `grep -n "Assemble" *.cpp` |

---

## 相关文档

- 主流程文档：[SKILL.md](./SKILL.md)
- IR分析详细指南：`pypto/.agents/skills/pypto-pass-error-locator/references/ir-analysis-guide.md`
- 二分CCE方法：[binary_cce.md](./binary_cce.md)