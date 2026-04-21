
# Binary CCE 二分定位脚本

## 概述

本脚本用于通过二分打印法定位CCE中哪个Op出现精度问题，帮助获取真实上板数据（DDR/GM和UB）。

## 使用方法

### 方式一：初始化配置（推荐先执行）

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --init \
    --work-path /path/to/work
```

这会：
1. 修改 `tile_fwk_config.json` 开启 `fixed_output_path`
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

### 方式四：完整流程

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/work \
    --test-cmd python3 your_test.py
```

### 方式五：打印 Shape 变量（用于诊断 validshape 问题）

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/work \
    --print-idx 0 \
    --print-shape sym_15_dim_0,sym_15_dim_1
```

### 方式六：检测 ValidShape 问题

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/work \
    --ir-file path/to/ir_file.tifwkgr
```

## 参数说明

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
| `--check-validshape` | 否 | 检测 validshape 问题（指定 IR 文件路径） |
| `--ir-file` | 否 | 指定 IR 文件路径（用于 validshape 检测） |
| `--test-cmd` | 否 | 测试命令（多个参数用引号包裹） |

## 手动流程（参考）

如果不想使用脚本，可以手动操作：

### 1. 配置开关

修改 `framework/src/interface/configs/tile_fwk_config.json`:
```json
"codegen": {
    "fixed_output_path": true,
    "force_overwrite": false
}
```

修改 `framework/src/interface/machine/device/tilefwk/aicore_print.h`:
```c
#define ENABLE_AICORE_PRINT 1
```

### 2. 重新编译
```bash
python3 -m pip install . -v
```

### 3. 修改CCE添加打印

在生成的 `.cce` 文件中添加：
```c
#include "tilefwk/aicore_print.h"

// 打印GM tensor
AiCorePrintGmTensor(param->ctx, (__gm__bfloat16_t*)gmTensor_999.GetAddr(), 1, 0);

// 打印UB tensor  
AiCorePrintUbTensor(param->ctx, (__ub__bfloat16_t*)ubTensor_999.GetAddr(), 1, 0);

// 打印Shape接口
AiCorePrintShape(param->ctx, Shape2Dim(sym_15_dim_0, sym_15_dim_1));
```



### 4. 配置环境变量
```bash
export ASCEND_WORK_PATH=/path/to/work
export ASCEND_GLOBAL_LOG_LEVEL=0
```

### 5. 运行并查看日志

运行测试后，在以下位置查找日志：
```
/path/to/work/log/debug/device-*/DumpAicoreLog*
```

## 定位原理

1. 找到所有CCE文件（按生成顺序排列）
2. 在某个CCE中添加打印语句，打印其中的tensor数据
3. 运行测试，对比打印数据与golden
4. 通过二分法不断缩小范围，最终定位到出错的Op

## 注意事项

- 打印buffer默认16KB，可修改 `aicpu_common.h` 中的 `PRINT_BUFFER_SIZE`
- 打印语句会影响性能，仅用于调试
- 调试完成后记得删除打印语句

## Shape 打印与 ValidShape 问题诊断

### 何时使用 Shape 打印

当怀疑 validshape 有问题时，可以使用 shape 打印功能：

1. **数据异常但代码逻辑正确**：可能是 shape/validshape 不匹配导致
2. **越界访问错误**：可能是 validshape 设置错误
3. **数据维度不匹配**：需要确认运行时 shape 信息

### Shape 打印方法

在 CCE 文件中添加 shape 打印：

```c
#include "tilefwk/aicore_print.h"

// 打印 shape 信息
AiCoreLogF(param->ctx, "shape sym_15=[%ld,%ld]\n", sym_15_dim_0, sym_15_dim_1);
```

或者使用脚本自动添加：

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/work \
    --print-idx 0 \
    --print-shape sym_15_dim_0,sym_15_dim_1
```

### ValidShape 问题检测

自动检测 IR 文件中的 validshape 问题：

```bash
python3 .agents/skills/pypto-pass-precision-verify/binary_cce.py \
    --work-path /path/to/work \
    --ir-file path/to/ir_file.tifwkgr
```

检测内容包括：
- 输入 tensor 的 shape 和 validshape 是否一致
- offset/dynoffset 配置是否正确
- dynvalidshape 是否缺失

### 常见 ValidShape 问题

#### 1. Shape 和 ValidShape 不一致

**现象**：
```
Shape: [1, 4, 1, 64, 1, 4, 1, 64]
ValidShape: [1, 4, 1, 64]
```

**原因**：
- View 操作没有正确设置 validshape
- Offset 配置错误导致 validshape 计算错误

**解决方法**：
- 检查 offset/dynoffset 配置
- 确认 dynvalidshape 设置正确
- 查看 IR 文件中的操作信息

#### 2. Offset/Dynoffset 存在但 Dynvalidshape 缺失

**现象**：
```
Offset: [0, 0, 0, 0]
Dynoffset: [0, 0, 0, 0]
Dynvalidshape: []
```

**原因**：
- Pass 没有正确生成 dynvalidshape 信息

**解决方法**：
- 检查 Pass 实现
- 确认是否需要添加 dynvalidshape 计算

#### 3. 内存越界错误

**现象**：
```
RuntimeError: memory access out of bounds
```

**原因**：
- Validshape 设置错误导致访问越界

**解决方法**：
- 使用 shape 打印确认运行时 shape
- 对比 IR 文件中的 shape 和 validshape
- 检查 offset 配置

### 完整诊断流程

1. **检测问题**：
   ```bash
   python3 binary_cce.py --work-path /path/to/work --ir-file path/to/ir.tifwkgr
   ```

2. **添加 Shape 打印**：
   ```bash
   python3 binary_cce.py --work-path /path/to/work --print-idx 0 --print-shape
   ```

3. **运行测试并查看日志**：
   ```bash
   # 运行测试
   python3 test.py
   
   # 查看日志
   cat /path/to/work/log/debug/device-*/DumpAicoreLog*
   ```

4. **分析结果**：
   - 对比打印的 shape 和 IR 文件中的 shape
   - 检查 validshape 是否正确
   - 确认 offset 配置

### 日志分析示例

**Shape 打印日志**：
```
shape sym_15=[1,4,1,64]
shape sym_16=[1,4,1,64]
```

**ValidShape 检测报告**：
```
=== ValidShape 问题检测报告 ===
文件: After_005_MergeViewAssemble_TENSOR_default_loop_1_Unroll1_PATH0_hiddenfunc0_5.tifwkgr
总操作数: 4
发现问题数: 2

### 操作 10004 (TILE_VIEW) - 第 17 行
  输入 Tensor 问题:
    Tensor ID: 17
    Shape: [1, 4, 1, 64, 1, 4, 1, 64]
    ValidShape: [1, 4, 1, 64]
    问题: shape != valid_shape
    建议: 检查 offset/dynoffset 配置是否正确

### 操作 10007 (TILE_ASSEMBLE) - 第 21 行
  Offset 问题:
    问题: offset/dynoffset 存在但 dynvalidshape 缺失
    Offset: [0, 0, 0, 0]
    Dynoffset: [0, 0, 0, 0]
    建议: 检查是否需要添加 dynvalidshape
```

### 最佳实践

1. **先检测后打印**：先使用 validshape 检测功能，了解问题概况
2. **针对性打印**：根据检测结果选择需要打印的 shape 变量
3. **对比分析**：将打印结果与 IR 文件信息对比
4. **逐步定位**：结合二分法逐步缩小问题范围
5. **及时清理**：调试完成后删除打印语句