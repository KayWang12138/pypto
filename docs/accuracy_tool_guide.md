# PyPTO 精度工具使用说明

## 概述

PyPTO 精度验证工具是一套完整的算子精度问题定位和验证系统，支持前端模拟计算验证和 NPU 上板数据 dump，帮助开发者快速定位功能错误和计算误差。

当 PyPTO 算子执行后无功能告警或报错，但输出数据不符合预期时，可基于以下方法进行精度问题的定位和定位。精度问题主要来源于两个方面：

- **功能错误**：硬件静默故障、软件静默功能问题、公式实现错误引起的明显数据错误或误差
- **计算误差**：数据类型、算法（切分、累积、公式近似）差异等引起的明显数据误差

## 整体流程

精度调试的整体流程如下：

1. 规避已知问题
2. 缩小问题规模
3. 工具自检
4. 上板 dump（可选）
5. 分析验证结果

## 规避已知问题

精度调试前，应确保已规避当前软件存在的已知问题，详细请参见已知问题文档。

## 缩小问题规模

缩小问题规模通常是一个可选步骤，旨在简化问题，提高复现和定位的效率。

- 缩小问题规模后需能复现同样问题，然后继续进行后续的工具自检或人工调试
- 对于缩小后出现新问题的情况，建议尝试其它缩小方法以复现原始问题，不建议将新问题纳入关键定位流程

然而，在某些情况下，缩小问题规模是必选步骤，例如对于较大的模型：
- 主机内存不足导致自检工具无法执行
- 文件存储空间过小导致自检工具无法保存中间计算数据
- 其它分析流程或工具的耗时超过主观容忍范围，甚至无法执行等阻塞式情况

通常通过以下方法缩小问题规模：

- **减少子图数量和大小**：例如减少 loop 的次数或减少 cube/vector Tiling 块的个数（即增大 TileShape 的大小）
- **裁剪模型**：例如调小模型的 Shape 规格，如 batch_size、seq_len 等
- **采用二分法移除尾部计算**：
  1. 按模型计算的顺序，采用二分法移除靠近尾部的计算，并将断开的输出加入算子的输出列表
  2. 执行算子并观察、分析新的输出列表
     - 如果数据正常（无 inf/nan、无主观认为随机的值、或与参考基准数据误差较小）则返回上步继续二分操作
     - 如果数据存在异常，则将代码恢复到本次移除前的状态，作为最新的候选问题场景
     - 如果裁剪后的模型规模已经很小，可以停止二分操作并选择最新的候选问题场景进行后续的定位

## 工具自检

### 工具简介

PyPTO 在计算图编译的各 Pass 阶段拥有完整的中间表示，可翻译成第三方计算代码，并在其它计算单元（例如 Host CPU）上模拟计算过程。该工具通过模拟计算结果与基准数据的误差对比，可以检测算子异常或者某个 Pass 的处理结果是否存在异常，并定位首个出现异常的计算节点。

主要特性及使用场景：

- **粗检模式（TENSOR_GRAPH 模式）**：用于粗检算子代码、框架前端处理的正确性。基于用户提供的基准（golden）输入输出数据，与 Tensor Graph 模拟计算的最终结果对比检测整体计算的正确性。常用于以下情况：
  - 当用户存在可用的算子基准（golden）输入、输出数据时，可先使用粗检特性粗略排除算子代码、框架前端处理是否引入差异
  - 快速验证算子代码、框架前端处理是否引入差异
  - 验证算子的最终输出精度
  - 端到端的精度验证

- **自检模式（PASS 模式）**：用于自检 Pass 的正确性。基于各 Pass 模拟计算的结果，对比检测 Pass 正确性及异常计算节点。常用于以下情况：
  - 当用户算子精度刚刚出现问题且没有明确方向，可先使用自检特性排除 Pass 处理阶段是否引入潜在错误
  - 当用户大致明确某个 Pass 出问题时，使用自检特性获取该 Pass 及前序 Pass 的模拟计算中间数据，对比数据找出潜在出问题的计算操作
  - 定位精度问题出现的具体 pass
  - 调试中间结果
  - 验证 pass 级别的精度

- **人工调试**：指定单个计算结果，保存到文件或者以可读形式打印到输出、日志。
  - 当用户的数据异常为典型的连续非法值、连续异常 0 值时，可使用自检开关并使用 pass_verify_print/pass_verify_save 特性打印、保存可疑的模拟数据，逐步人工检查找到首个发生异常的计算代码行
  - 当用户存在可用的算子基准（golden）中间数据、且中间数据可与 PyPTO 算子的相应计算直接对应时，可使用自检开关并使用 pass_verify_print/pass_verify_save 特性打印、保存可对应的数据，人工对比检查找到首个发生异常的计算代码行

### 使用约束

当前精度调试工具存在以下限制（完整计算流表示仅保存在 Pass 运行上下文中），无法使用检测功能：

- 不支持上板执行的中间数据检查，仅支持前端及 Pass 的检查
- 不支持集合通信场景
- 不支持特定 Pass，特定 Pass（例如 SubgraphToFunction）属于中间的优化过程缺少完整计算信息，工具内部做自动跳过处理
- 不支持 Pass 间的自动对比校验（需人工进行数据数据对比）
- 不支持程序退出后在任意运行环境构造并模拟计算。需在算子编译期间，所对应的主机 CPU 及进程上构造并模拟计算
- 不支持基于昇腾 AI 处理器调用 Ascend C 构造并模拟计算
- 不支持基于 GPU 构造并模拟计算

### 环境要求

最新 master 分支代码及 0.1.1 之后版本（不含 0.1.1 版本）支持在运行时在线编译精度工具所需 C++ 二进制，不需重新编译安装 PyPTO，但需确认在线编译所需的构建工具符合以下要求：

- cmake >= 3.16.3
- make
- g++ >= 9.4.0

### 自检操作步骤

1. 开启精度调试开关。参考样例为：hello_world.py

```python
...
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    ...
}

@pypto.frontend.jit(verify_options=verify_options)
def add_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    ) -> pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    return input0 + input1
...
```

2. 设置 golden 数据（可选）

如果需要进行粗检模式（TENSOR_GRAPH 模式）验证，需要设置 golden 数据：

```python
import torch

# 准备输入数据
input0 = torch.randn((1, 4, 1, 64), dtype=torch.float32)
input1 = torch.randn((1, 4, 1, 64), dtype=torch.float32)

# 计算 golden 结果
golden = input0 + input1

# 设置 golden 数据（golden 必须在 CPU 上）
# 参数说明：goldens 列表对应算子的输出参数，None 表示跳过该参数的验证
pypto.set_verify_golden_data(goldens=[golden.cpu()])
```

3. 执行修改后用例

```bash
python3 examples/00_hello_world/hello_world.py
```

打印类似以下输出，指示对应的自检结果为通过（PASS）、未通过（FAIL(ED)）或跳过校验（NO_COMPARE）：

```text
2025-mm-dd HH:MM:SS:xxx V | tensor_graph Verify NO_COMPARE
2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_00_RemoveRedundantReshape Verify result PASS
2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_01_AutoCast Verify result PASS
2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_02_InferMemoryConflict Verify result PASS
...
2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_34_InsertSync Verify result PASS
...
```

3. 执行结束后，在 `${work_path}/output/output_*/` 目录（`*` 代表时间戳）下生成 `verify_*` 目录，存放检测结果文件

```text
├── tensor_graph/              # 保存前端初始计算图模拟计算后的中间数据，作为基础数据
│   ├── *.data
│   └── ...
├── verify_result.csv          # 结果报告，用于保存结果报告
    ├── {FUNC_NAME}.pass_{PASS_SEQ}_{PASS_NAME}/  # 保存中间 pass 计算图模拟计算后的中间数据，作为待测数据
│   ├── *.data
│   └── ...
```

4. 查看结果报告文件 verify_result.csv

**表 1** 结果报告文件参数说明

| 参数 | 说明 |
|------|------|
| No. | 单个 Pass 内的数据顺序编号，例如 1 |
| rootFuncID | Root Function 节点的唯一标识 |
| funcID | Function 节点的唯一标识 |
| verifyType | 算子信息及数据对应的 pass 名称，<br>格式为：* {FUNC_NAME}.pass_{PASS_SEQ}_{PASS_NAME}<br>示例为：function_TENSOR_LOOP_L1_Unroll1_PATH1_7.pass_06_SplitReshape |
| LoopInfo | 控制流信息，例如 s_idx=2@b_idx=1 |
| opCode | Tile 算子名称 |
| opMagic | Operation 节点的唯一标识 |
| tensorMagic | Tensor 的唯一标识 |
| rawTensorMagic | Tensor 节点所属物理内存区域 Raw Tensor 的唯一标识 |
| offset | 当前 Tensor 在 rawTensor 内存中的偏移量，为整数数组 |
| inputShape | 输入 Tensor 节点的形状信息，为整数数组 |
| inputValidShape | 输入 Tensor 节点实际数据大小的形状信息，为整数数组 |
| inputDtype | 输入数据类型 |
| inputTensors | Tile 算子的输入 tile tensor 对应的数据文件名列表，当前不支持文件落盘 |
| outputShape | 输出 Tensor 节点的形状信息，为整数数组 |
| outputValidShape | 输出 Tensor 节点实际数据大小的形状信息，为整数数组 |
| outputDynValidShape | 输出 Tensor 节点中实际数据大小的动态形状信息，为整数数组 |
| outputDtype | 输出数据类型 |
| outputTensor | tile 算子的输出 tile tensor 对应的数据文件名，<br>例如：5~10003~s_idx=2@b_idx=1~7~10007~MUL~16~17~1765890242967069.data<br>该列表生成在 ${work_path}/output/output_*/ 目录下。<br>文件名格式定义如下：{ROOT_FN_ID}~{CALL_OP_ID}~{LOOP_INFO}~{LEAF_FN_ID}~{OP_ID}~{OP_NAME}~{TENSOR_ID}~{TILE_TENSOR_ID}~{TIMESTAMP}.data<br>文件内容格式如下：Tensor 数据的直接内存转储（endianness 遵循主机 CPU 所使用的规则，例如 x86 CPU 通常为 little-endian） |
| verifyResult | 对比检测的结果：<br>粗检模式：用户 golden 与模拟计算最终输出的结果比较。<br>自检模式：tensor graph 中间模拟输出与其它 pass 中间模拟输出的结果比较。 |
| maxAbsDiff | 误差信息：最大绝对误差 |
| maxRelDiff | 误差信息：最大相对误差 |
| errorCount | 误差信息：实际错误数量 N_err |
| errorRatio | 误差信息：实际错误数占比 R<br>判定阈值 T：R<=T 为通过<br>粗检模式：T==1e-2<br>自检模式：T==1e-3 |

当前工具基于以下方式生成检测结果。

1. 统计待测数据（`*pass_*/\*.data`）中绝对值不大于 1e-6 但基准数据（`tensor_graph/*.data`）大于 1e-6 的数量，记为 N_zero
2. 给定误差阈值 T，对于两组数据中绝对值均大于 1e-6 的值，逐点（elementwise）统计相对误差、绝对误差均大于 T 的数值占总数据量的比例，记为 R
3. 如果 N_zero <= 1000，且 R <= T，则判定误差在接受范围内

5. 后续处理建议

建议收集相关结果信息，并提交 ISSUE 进行处理。

## 上板 Leaf Func Dump 功能

### 1. 功能概述

上板 Leaf Func Dump 功能支持在 NPU 设备上执行算子时，实时 dump 中间张量数据，用于对比前端模拟计算结果和 NPU 实际执行结果，定位精度问题。

### 2. 启用方式

```python
import os

# 设置环境变量启用上板 dump
os.environ["PTO_DATADUMP_ENABLE"] = "true"

# 配置验证选项
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True
    }
)
def kernel(...):
    ...
```

### 3. Dump 数据格式

**Dump 数据结构：**
```cpp
struct DumpTensorInfo {
    uint32_t headSize;          // 头部大小
    uint32_t funcId;            // 函数 ID
    uint32_t taskId;            // 任务 ID
    uint32_t callopMagic;       // 操作 magic
    int32_t coreId;            // 核心 ID
    int32_t dataType;           // 数据类型
    int32_t rawMagic;           // 原始 tensor magic
    int32_t dims;               // 维度数
    int64_t exeStart;           // 执行开始时间
    int64_t exeEnd;             // 执行结束时间
    uint64_t rootHash;          // root hash
    uint64_t funcHash;          // 函数 hash
    uint64_t timeStamp;          // 时间戳
    uint64_t shape[DEV_SHAPE_DIM_MAX];     // shape 信息
    uint64_t offset[DEV_SHAPE_DIM_MAX];    // offset 信息
    uint64_t rawShape[DEV_SHAPE_DIM_MAX];  // 原始 shape
    uint64_t tensorAddr;        // tensor 地址
};
```

**支持的数据类型：**
| 类型值 | 数据类型 | Python 类型 |
|--------|---------|------------|
| 0 | DT_INT4 | ml_dtypes.int4 |
| 1 | DT_INT8 | np.int8 |
| 2 | DT_INT16 | np.int16 |
| 3 | DT_INT32 | np.int32 |
| 4 | DT_INT64 | np.int64 |
| 5 | DT_FP8 | ml_dtypes.float8_e4m3fn |
| 6 | DT_FP16 | np.float16 |
| 7 | DT_FP32 | np.float32 |
| 8 | DT_BF16 | ml_dtypes.bfloat16 |
| 11 | DT_UINT8 | np.uint8 |
| 12 | DT_UINT16 | np.uint16 |
| 13 | DT_UINT32 | np.uint32 |
| 14 | DT_UINT64 | np.uint64 |
| 15 | DT_BOOL | np.bool_ |
| 16 | DT_DOUBLE | np.float64 |

### 4. Dump 数据输出路径

```
output/dump_tensor{hostPid}/device_{deviceId}/
└── {taskId}_{seqNo}_{callopMagic}_{rootHash}_{funcHash}_{rawMagic}_{timeStamp}_{dataType}_{input/output}{index}.tdump
```

**文件名组成：**
- taskId: 任务 ID
- seqNo: 序列号
- callopMagic: 操作 magic
- rootHash: root hash
- funcHash: 函数 hash
- rawMagic: 原始 tensor magic
- timeStamp: 时间戳
- dataType: 数据类型
- input/output{index}: 输入/输出标记及索引

### 5. 数据处理工具

**工具位置：** `tools/verifier/parse_dump_tensors.py`

**主要功能：**
- 解析 dump 的二进制数据（.tdump 文件）
- 提取 tensor 数据并保存为 .data 文件
- 与 verify_result.csv 进行数据验证对比
- 合并多个 tensor 为完整的 raw tensor
- 支持多进程并行处理
- 生成 tensor_info.csv 报告文件

**使用方法：**
```bash
# 基本用法（不进行验证）
python3 tools/verifier/parse_dump_tensors.py \
    --dump_tensor_path output/dump_tensor/device_0

# 带验证的用法
python3 tools/verifier/parse_dump_tensors.py \
    --dump_tensor_path output/dump_tensor/device_0 \
    --verify_path output/output_*/verify_*/
```

**参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dump_tensor_path` | dump 数据目录路径 | `output/dump_tensor/device_0` |
| `--verify_path` | verify_result.csv 所在目录 | `""`（不验证） |

**输出文件：**
```
output/dump_tensor/device_0/
├── tensor_info.csv              # 解析结果报告
├── *.data                       # 提取的 tensor 数据文件
└── raw_{rawMagic}_{dataType}_{ioflag}.data  # 合并后的 raw tensor
```

**tensor_info.csv 字段说明：**
| 字段 | 说明 |
|------|------|
| headSize | 头部大小 |
| funcId | 函数 ID |
| taskId | 任务 ID |
| callopMagic | 操作 magic |
| coreId | 核心 ID |
| dataType | 数据类型（数值） |
| dataTypeStr | 数据类型（字符串） |
| rawMagic | 原始 tensor magic |
| dims | 维度数 |
| exeStart | 执行开始时间 |
| exeEnd | 执行结束时间 |
| exeDuration | 执行时长 |
| rootHash | root hash |
| funcHash | 函数 hash |
| timeStamp | 时间戳 |
| shape | tensor shape |
| offset | 在 raw tensor 中的偏移 |
| rawShape | 原始 tensor shape |
| tensorAddr | tensor 地址 |
| ioflag | 输入/输出标记 |
| seqNo | 序列号 |
| bin_file | 数据文件路径 |
| verify_tensor_file | 验证数据文件路径（如果启用验证） |
| cmp_res | 对比结果（True/False/"NO_CMP"） |
| loop_info | 循环信息 |

### 6. 数据验证功能

当提供 `--verify_path` 参数时，工具会自动进行数据验证：

**验证流程：**
1. 读取 verify_result.csv 文件
2. 按 callopMagic 分组处理 tensor
3. 匹配 CodegenPreproc pass 中的 COPY_IN/COPY_OUT 操作
4. 对比 dump 数据与 verify 数据
5. 记录对比结果到 tensor_info.csv

**验证逻辑：**
- 对于输入 tensor：匹配 COPY_IN 操作的输出
- 对于输出 tensor：匹配 COPY_OUT 操作的输入
- 按 offset 和 loop_info 进行精确匹配
- 使用 np.allclose 进行数据对比（容差 1e-3）

### 7. Raw Tensor 合并功能

工具会自动将属于同一个 rawMagic 的多个 tensor 合并成一个完整的 raw tensor：

**合并规则：**
- 按 offset 排序所有 tensor
- 将每个 tensor 的数据放置到对应 offset 位置
- 生成 `raw_{rawMagic}_{dataType}_{ioflag}.data` 文件
- 如果 tensor shape 等于 rawShape，则跳过合并

**合并后的验证：**
- 如果存在 tensor_graph pass 的验证数据
- 自动对比合并后的 raw tensor 与验证数据
- 记录对比结果

### 8. 使用限制

- 单个 tensor 数据大小限制：2MB
- 需要 IDE 环境支持（IdeDumpStart、IdeDumpData、IdeDumpEnd 接口）
- 仅支持上板执行，不支持模拟模式
- 需要设置 hostPid 才能启用
- DT_HF4 和 DT_HF8 数据类型暂不支持解析

## 配置选项

### jit 装饰器的 verify_options 参数

通过 jit 装饰器的 `verify_options` 参数配置验证选项：

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "./verify_output",
        "pass_verify_pass_filter": ["all"],
        "pass_verify_error_tol": [1e-3, 1e-3]
    }
)
def my_kernel(x, y, out):
    pass
```

**参数说明**：

| `参数` | 类型 | 说明 |
|--------|------|------|
| `enable_pass_verify` | bool | 是否启用 pass 验证 |
| `pass_verify_save_tensor` | bool | 是否保存 tensor 数据到文件 |
| `pass_verify_save_tensor_dir` | str | tensor 保存路径 |
| `pass_verify_pass_filter` | List[str] | 过滤要验证的 pass，`["all"]` 表示验证所有 pass |
| `pass_verify_error_tol` | List[float] | 容差配置 `[rt`ol, atol]`，默认 `[1e-3, 1e-3]` |

**配置示例**：

```python
# 基本配置
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True
    }
)
def my_kernel(x, y, out):
    pass

# 完整配置
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "./verify_output",
        "pass_verify_pass_filter": ["all"],
        "pass_verify_error_tol": [1e-3, 1e-3]
    }
)
def my_kernel(x, y, out):
    pass

# 验证特定 pass
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_pass_filter": ["PadLocalBuffer", "FuseOps"]
    }
)
def my_kernel(x, y, out):
    pass
```

### set_verify_golden_data

设置 golden 数据：

```python
def set_verify_golden_data(in_out_tensors=None, goldens=None):
    """
    设置验证用的 golden 数据
    
    Args:
        in_out_tensors: 输入和输出 tensor 列表 [input1, input2, ..., output1, output2, ...]
        goldens: golden 数据列表 [golden1, golden2, ...]
    
    功能特性：
        - 设置用户执行算子时实际的输入、输出列表
        - 设置计算基准数据（golden）用于对比检测
        - 支持跳过特定位置的数据对比（设置为 None）
    """
```

**使用示例**：
```python
# 仅设置 golden 输出数据
input1 = torch.randn((32, 32), dtype=torch.float16, device='npu:0')
input2 = torch.randn((32, 32), dtype=torch.float16, device='npu:0')
output = torch.zeros((32, 32), dtype=torch.float16, device='npu:0')
golden = input1 + input2

# 设置 golden 数据（golden 必须在 CPU 上）
pypto.set_verify_golden_data(goldens=[None, None, golden.cpu()])

# 同时设置实际输入和 golden 输出
pypto.set_verify_golden_data([input1.cpu(), input2.cpu(), output.cpu()], [None, None, golden.cpu()])
```

## 调试工具

### 1. pass_verify_print

在 pass 验证过程中条件性打印 tensor 和标量值。

**函数签名**：
```python
def pass_verify_print(*values, cond: Union[int, SymbolicScalar] = 1) -> None:
    """
    在 pass 验证过程中条件性打印 tensor 和标量值
    
    Parameters
    ----------
    *values :
        混合类型的值列表：
        - Tensor / pypto.Tensor: 以紧凑格式打印 tensor
        - int / SymbolicScalar: 作为符号标量打印
        - 其他 Python 对象: 直接转换为字符串打印
    cond : int or SymbolicScalar, optional
        打印条件。当 `cond` 为 0 时，不打印任何内容。
        默认为 1（总是打印）。
    
    Notes
    -----
    此 API 仅用于 **pass 验证**，不影响数值计算结果。
    它只影响日志输出。
    
    功能特性：
        - 支持混合打印：张量、符号标量、Python 对象
        - 张量以 Tensor 格式打印
        - 可通过 `cond` 参数控制打印条件（符号标量或整数）
        - 仅影响日志输出，不影响数值结果
    """
```

**使用示例**：

**示例 1：打印 tensor 和标量**
```python
import pypto

@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={"enable_pass_verify": True}
)
def my_kernel(x, y, out):
    # 打印输入 tensor
    pypto.pass_verify_print("Input x:", x)
    pypto.pass_verify_print("Input y:", y)
    
    # 计算
    temp = x + y
    
    # 打印中间结果
    pypto.pass_verify_print("Temp:", temp)
    
    out[:] = temp * 2
```

**示例 2：条件性打印**
```python
import pypto

@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={"enable_pass_verify": True}
)
def my_kernel(x, out):
    for idx in pypto.loop(10):
        # 仅在 idx > 5 时打印
        pypto.pass_verify_print("Processing idx=", idx, cond=(idx > 5))
        
        # 计算逻辑
        result = x * idx
        pypto.assemble(result, [idx], out)
```

**示例 3：打印多个值**
```python
import pypto

@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={"enable_pass_verify": True}
)
def my_kernel(x, y, z, out):
    # 打印多个 tensor 和标量
    pypto.pass_verify_print("x=", x, ", y=", y, ", z=", z, ", step=", 10)
    
    out[:] = x + y + z
```

### 2. pass_verify_save

在 pass 验证过程中条件性保存 tensor 到文件。

**函数签名**：
```python
def pass_verify_save(
    tensor: Tensor,
    fname: Union[str, SymbolicScalar, int],
    cond: Union[int, SymbolicScalar] = 1,
    **kwargs: Union[int, SymbolicScalar, pypto_impl.SymbolicScalar],
) -> None:
    """
    在 pass 验证过程中条件性保存 tensor 到文件
    
    Parameters
    ----------
    tensor : Tensor
        要保存的 tensor
    fname : str or SymbolicScalar or int
        文件名模板。可以使用动态占位符 ``$name``,
        并从 ``kwargs`` 中获取对应的值进行替换。
        实际保存路径由后端验证配置决定。
    cond : int or SymbolicScalar, optional
        保存条件。当 `cond` 为 0 时，忽略此调用。
        默认为 1（总是保存）。
    **kwargs :
        从占位符名称到标量值的映射，例如
        ``pass_verify_save(t, "tensor_$idx", idx=loop_idx)``。
    
    Notes
    -----
    此 API 仅用于 **pass 验证**，不影响计算结果。
    主要用于定位和比较特定 Pass / 迭代的中间 tensor。
    
    功能特性：
        - 保存中间张量到文件
        - 支持动态文件名模板（使用 `$name` 占位符）
        - 可通过 `cond` 参数控制保存条件
        - 用于定位和对比特定 Pass / 迭代的中间张量
    """
```

**使用示例**：

**示例 1：保存 tensor 到固定文件名**
```python
import pypto

@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "./verify_output"
    }
)
def my_kernel(x, y, out):
    temp = x + y
    
    # 保存中间结果
    pypto.pass_verify_save(temp, "temp_tensor")
    
    out[:] = temp * 2
```

**示例 2：使用循环索引保存多个 tensor**
```python
import pypto

@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "./verify_output"
    }
)
def my_kernel(x, out):
    for idx in pypto.loop(10):
        tile = pypto.view(x, [32], [idx * 32])
        result = tile * idx
        
        # 保存每个循环迭代的 tensor
        # 文件名格式: tensor_out_0.data, tensor_out_1.data, ...
        pypto.pass_verify_save(result, "tensor_out_$idx", idx=idx)
        
        pypto.assemble(result, [idx * 32], out)
```

**示例 3：条件性保存**
```python
import pypto

@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "./verify_output"
    }
)
def my_kernel(x, out):
    for idx in pypto.loop(10):
        tile = pypto.view(x, [32], [idx * 32])
        result = tile * idx
        
        # 仅在 idx > 5 时保存
        pypto.pass_verify_save(result, "tensor_out_$idx", idx=idx, cond=(idx > 5))
        
        pypto.assemble(result, [idx * 32], out)
```

**示例 4：使用多个占位符**
```python
import pypto

@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "./verify_output"
    }
)
def my_kernel(x, out):
    for batch_idx in pypto.loop(2):
        for seq_idx in pypto.loop(5):
            tile = pypto.view(x, [32], [batch_idx * 160 + seq_idx * 32])
            result = tile * (batch_idx + seq_idx)
            
            # 使用多个占位符
            # 文件名格式: tensor_batch_0_seq_0.data, tensor_batch_0_seq_1.data, ...
            pypto.pass_verify_save(
                result, 
                "tensor_batch_$batch_idx_seq_$seq_idx", 
                batch_idx=batch_idx, 
                seq_idx=seq_idx
            )
            
            pypto.assemble(result, [batch_idx * 160 + seq_idx * 32], out)
```

### 3. pass_verify_print 和 pass_verify_save 配合使用

**示例：综合调试**
```python
import pypto

@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "./verify_output",
        "pass_verify_pass_filter": ["all"]
    }
)
def complex_kernel(x, y, out):
    # 打印输入
    pypto.pass_verify_print("Start processing")
    pypto.pass_verify_print("Input x shape:", x.shape)
    pypto.pass_verify_print("Input y shape:", y.shape)
    
    # 第一步计算
    temp1 = x + y
    pypto.pass_verify_print("After add:", temp1)
    pypto.pass_verify_save(temp1, "temp1")
    
    # 第二步计算
    temp2 = temp1 * x
    pypto.pass_verify_print("After mul:", temp2)
    pypto.pass_verify_save(temp2, "temp2")
    
    # 循环处理
    for idx in pypto.loop(10):
        tile = pypto.view(temp2, [32], [idx * 32])
        result = tile * idx
        
        # 打印和保存循环中间结果
        pypto.pass_verify_print("Loop idx=", idx, ", result=", result, cond=(idx % 2 == 0))
        pypto.pass_verify_save(result, "loop_result_$idx", idx=idx, cond=(idx % 2 == 0))
        
        pypto.assemble(result, [idx * 32], out)
    
    pypto.pass_verify_print("Processing completed")
```

## 误差判定机制

### 判定逻辑

1. **零值检测**：统计待测数据中绝对值 ≤ 1e-6 但基准数据 > 1e-6 的数量，记为 N_zero
2. **误差检测**：对于两组数据中绝对值均 > 1e-6 的值，统计相对误差和绝对误差均 > 阈值 T 的数值占比，记为 R
3. **判定条件**：`N_zero <= 1000` 且 `R <= T`

### 误差阈值

- **Golden 对比模式**：T == 1e-2
- **Pass 间对比模式**：T == 1e-3
- **自定义**：通过 `pass_verify_error_tol` 设置

### 验证报告字段 (`verify_result.csv`)

| 字段 | 说明 |
|------|------|
| No. | Pass 内数据顺序编号 |
| verifyType | 算子信息及 Pass 名称 |
| opCode | Tile 算子名称 |
| inputShape/outputShape | 输入输出形状 |
| inputDtype/outputDtype | 输入输出数据类型 |
| verifyResult | 验证结果（PASS/FAIL/NO_COMPARE） |
| maxAbsDiff | 最大绝对误差 |
| maxRelDiff | 最大相对误差 |
| errorCount | 实际错误数量 |
| errorRatio | 实际错误数占比 |

## 容差计算机制

### 容差公式

PyPTO 精度工具使用以下容差计算公式：

```cpp
// 相对误差计算
relDiff = |output - golden| / (|output| + |golden|) * 2

// 容差阈值
tol_attn = (|output| + |golden|) * rtol / 2 + atol

// 失败阈值（严格模式）
tol_fail = tol_attn * 128

// 判断是否超出容差
if (|output - golden| > tol_attn) {
    // 记录误差
}

// 判断是否失败
if (|output - golden| > tol_fail) {
    // 标记为失败
}
```

### 容差配置建议

根据算子类型和数据类型选择合适的容差：

| 算子类型 | 数据类型 | rtol | atol | 说明 |
|---------|---------|------|------|------|
| **复杂算子** | BF16 | `0.0078125` | `0.0001` | Attention、FFN、涉及 Softmax、除法等不稳定运算 |
| **简单算子** | BF16 | `5e-3` | `5e-3` | Gate、MatMul、简单矩阵运算 |
| **通用算子** | FP32 | `1e-3` | `1e-3` | 默认容差 |
| **高精度要求** | FP32 | `1e-5` | `1e-5` | 对精度要求极高的场景 |
| **精确匹配** | INT | `0` | `0` | 整数运算、精确匹配场景 |

**配置示例**：
```python
# BF16 复杂算子
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_error_tol": [0.0078125, 0.0001]
    }
)
def my_kernel(x, y, out):
    pass

# FP32 通用算子
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_error_tol": [1e-3, 1e-3]
    }
)
def my_kernel(x, y, out):
    pass

# 整数精确匹配
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_error_tol": [0, 0]
    }
)
def my_kernel(x, y, out):
    pass
```

## 支持的数据类型

精度工具支持以下数据类型：

| 数据类型 | Python 类型 |
|---------|------------|
| DT_INT8 | torch.int8 |
| DT_INT16 | torch.int16 |
| DT_INT32 | torch.int32 |
| DT_INT64 | torch.int64 |
| DT_UINT8 | torch.uint8 |
| DT_UINT16 | torch.uint16 |
| DT_UINT32 | torch.uint32 |
| DT_UINT64 | torch.uint64 |
| DT_FP16 | torch.float16 |
| DT_FP32 | torch.float32 |
| DT_BF16 | torch.bfloat16 |
| DT_DOUBLE | torch.float64 |
| DT_BOOL | torch.bool |

## 验证结果输出

### 1. 控制台输出

验证失败时，工具会输出详细的错误信息：

```
Error rtol=0.001 atol=0.001
  index:0 golden:1.234567 output:1.234568 absDiff:0.000001 relDiff:0.000000
  index:1 golden:2.345678 output:2.345679 absDiff:0.000001 relDiff:0.000000
  ...
  
All size:163840 failNum:0 maxAbsDiff:0.000001 maxRelDiff:0.000000 errorCount:0 errorRatio:0.0000 zeroCount:0 zeroRatio:0.0000
maxAbs-> index:0 golden:1.234567 output:1.234568 absDiff:0.000001 relDiff:0.000000
maxRel-> index:0 golden:1.234567 output:1.234568 absDiff:0.000001 relDiff:0.000000
```

### 2. CSV 文件输出

当启用 `pass_verify_save_tensor` 时，工具会生成 CSV 文件记录验证结果：

**Golden 对比模式输出结果：**
```
output/output_*/verify_*/
├── tensor_graph/              # 输入输出数据
│   ├── tensor_Incast_0.data
│   ├── tensor_OUT_0.data
└── verify_result.csv          # 验证报告
```

**Pass 间对比模式输出结果：**
```
output/output_*/verify_*/
├── tensor_graph/              # 基准数据
│   ├── *.data
├── verify_result.csv          # 验证报告
└── {FUNC_NAME}.pass_{SEQ}_{NAME}/  # 各 Pass 数据
    ├── *.data
```

**CSV 文件路径：**
```
{pass_verify_save_tensor_dir}/verify_{timestamp}/verify_result.csv
```

**CSV 文件列**：
- No.
- rootFuncID
- funcID
- passName
- verifyType
- callopMagic
- loopInfo
- opMagic
- opCode
- rawTensorMagic
- tensorMagic
- callopRawMagic
- offset
- inputShape
- inputValidShape
- inputDtype
- inputTensors
- outputShape
- tensorOffset
- outputValidShape
- outputDynValidShape
- outputDtype
- outputTensor
- verifyResult
- maxAbsDiff
- maxRelDiff
- errorCount
- errorRatio

### 3. Tensor 数据文件

当启用 `pass_verify_save_tensor` 时，工具会保存 tensor 数据到二进制文件：

**文件命名格式：**
```
{tensorName}~{index}~{timestamp}.data
```

**示例**：
```
tensor~my_kernel~PadLocalBuffer~0~1234567890.data
```

## 完整使用示例

### 示例 1：基本精度验证（粗检）

```python
import os
import torch
import torch_npu
import pypto

# 设置环境
device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
device = f'npu:{device_id}'
torch.npu.set_device(device_id)

# 定义算子
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_error_tol": [1e-3, 1e-3]
    }
)
def add_kernel(x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
               y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
               out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16)):
    out[:] = x + y

# 准备数据
shape = (32, 32)
input1 = torch.randn(shape, dtype=torch.float16, device=device)
input2 = torch.randn(shape, dtype=torch.float16, device=device)
output = torch.zeros(shape, dtype=torch.float16, device=device)
golden = input1 + input2

# 设置 golden 数据
pypto.set_verify_golden_data(goldens=[None, None, golden.cpu()])

# 执行算子
add_kernel(input1, input2, output)

# 验证结果
assert torch.allclose(output, golden, rtol=1e-3, atol=1e-3)
print("✓ 精度验证通过")
```

### 示例 2：Pass 级别验证与调试（自检）

```python
import os
import torch
import torch_npu
import pypto

# 设置环境
device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
device = f'npu:{device_id}'
torch.npu.set_device(device_id)

# 定义算子
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_pass_filter": ["all"],
        "pass_verify_error_tol": [1e-3, 1e-3]
    }
)
def complex_kernel(x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
                 y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
                 out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16)):
    # 打印输入
    pypto.pass_verify_print("Input x:", x)
    pypto.pass_verify_print("Input y:", y)
    
    # 第一步计算
    temp = x * y
    pypto.pass_verify_print("After mul:", temp)
    pypto.pass_verify_save(temp, "temp_mul")
    
    # 第二步计算
    out[:] = temp + x
    pypto.pass_verify_print("Output:", out)
    pypto.pass_verify_save(out, "output")

# 准备数据
shape = (32, 32)
input1 = torch.randn(shape, dtype=torch.float16, device=device)
input2 = torch.randn(shape, dtype=torch.float16, device=device)
output = torch.zeros(shape, dtype=torch.float16, device=device)
golden = input1 * input2 + input1

# 设置 golden 数据
pypto.set_verify_golden_data(goldens=[None, None, golden.cpu()])

# 执行算子
complex_kernel(input1, input2, output)

# 验证结果
assert torch.allclose(output, golden, rtol=1e-3, atol=1e-3)
print("✓ 精度验证通过")
print("✓ 验证结果已保存到 ./verify_output/")
```

### 示例 3：循环调试

```python
import os
import torch
import torch_npu
import pypto

# 设置环境
device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
device = f'npu:{device_id}'
torch.npu.set_device(device_id)

# 定义算子
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_pass_filter": ["all"],
        "pass_verify_error_tol": [1e-3, 1e-3]
    }
)
def loop_kernel(x: pypto.Tensor([pypto.STATIC], pypto.DT_FP16),
               out: pypto.Tensor([pypto.STATIC], pypto.DT_FP16)):
    pypto.pass_verify_print("Start loop processing")
    
    for idx in pypto.loop(10):
        tile = pypto.view(x, [32], [idx * 32])
        result = tile * idx
        
        # 调试循环中间结果
        pypto.pass_verify_print("Loop idx=", idx, ", result=", result, cond=(idx % 2 == 0))
        pypto.pass_verify_save(result, "loop_result_$idx", idx=idx, cond=(idx % 2 == 0))
        
        pypto.assemble(result, [idx * 32], out)
    
    pypto.pass_verify_print("Loop processing completed")

# 准备数据
shape = (320,)
input_data = torch.randn(shape, dtype=torch.float16, device=device)
output = torch.zeros(shape, dtype=torch.float16, device=device)
golden = torch.zeros(shape, dtype=torch.float16, device=device)
for idx in range(10):
    golden[idx * 32:(idx + 1) * 32] = input_data[idx * 32:(idx + 1) * 32] * idx

# 设置 golden 数据
pypto.set_verify_golden_data(goldens=[None, None, golden.cpu()])

# 执行算子
loop_kernel(input_data, output)

# 验证结果
assert torch.allclose(output, golden, rtol=1e-3, atol=1e-3)
print("✓ 精度验证通过")
print("✓ 验证结果已保存到 ./verify_output/")
```

### 示例 4：完整使用流程

```python
import os
import torch
import torch_npu
import pypto

# 1. 启用上板 dump 功能
os.environ["PTO_DATADUMP_ENABLE"] = "true"

# 2. 设置 golden 数据
golden_output = torch.rand((64, 64))
pypto.set_verify_golden_data(goldens=[None, None, golden_output])

# 3. 定义算子
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_error_tol": [1e-3, 1e-3]
    }
)
def add_kernel(input0, input1):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    
    # 人工调试：保存中间结果
    pypto.pass_verify_save(input1, "input1_debug")
    pypto.pass_verify_print(input0)
    
    return input0 + input1

# 4. 执行并查看结果
output = add_kernel(input0, input1)
# 验证报告自动生成在 output/output_*/verify_*/verify_result.csv
# 上板 dump 数据生成在 output/dump_tensor{pid}/device_{deviceId}/
```

## 调试技巧建议

### 1. 定位精度问题

当精度验证失败时，使用 PASS 模式定位问题：

```python
# 验证所有 pass
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_pass_filter": ["all"],
        "pass_verify_error_tol": [1e-3, 1e-3]
    }
)
def my_kernel(x, y, out):
    pass

# 执行后检查 CSV 文件，找到第一个失败的 pass
# 然后可以只验证该 pass
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_pass_filter": ["PassName"]
    }
)
def my_kernel(x, y, out):
    pass
```

### 2. 使用 pass_verify_print 调试

在算子中添加调试打印：

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={"enable_pass_verify": True}
)
def debug_kernel(x, y, out):
    # 打印输入
    pypto.pass_verify_print("Input x:", x)
    pypto.pass_verify_print("Input y:", y)
    
    # 中间结果
    temp = x + y
    pypto.pass_verify_print("Temp:", temp)
    
    out[:] = temp * 2
    pypto.pass_verify_print("Output:", out)
```

### 3. 使用 pass_verify_save 保存中间结果

保存中间结果用于离线分析：

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "./debug_output"
    }
)
def debug_kernel(x, y, out):
    temp1 = x + y
    pypto.pass_verify_save(temp1, "temp1")  # 保存 temp1
    
    temp2 = temp1 * x
    pypto.pass_verify_save(temp2, "temp2")  # 保存 temp2
    
    out[:] = temp2
```

### 4. 逐步放宽容差

如果精度验证失败，可以逐步放宽容差：

```python
# 首先使用严格容差
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_error_tol": [1e-5, 1e-5]
    }
)
def my_kernel(x, y, out):
    pass

# 如果失败，使用中等容差
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_error_tol": [1e-3, 1e-3]
    }
)
def my_kernel(x, y, out):
    pass

# 如果仍然失败，使用宽松容差
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_error_tol": [1e-2, 1e-2]
    }
)
def my_kernel(x, y, out):
    pass
```

### 5. 综合调试建议

1. **优先使用 Golden 对比验证**：存在 golden 时快速验证整体计算正确性
2. **使用 Pass 间对比验证**：当精度问题刚出现时，排除 Pass 处理阶段
3. **结合人工调试**：使用 `pass_verify_print` 和 `pass_verify_save` 定位具体问题
4. **逐步缩小范围**：通过 `pass_verify_pass_filter` 过滤特定 Pass
5. **分析误差报告**：查看 `verify_result.csv` 中的误差统计信息
6. **启用上板 dump**：需要对比 NPU 实际执行结果时，设置 `PTO_DATADUMP_ENABLE`

## 常见问题

### 问题 1：验证失败

**可能原因**：
- 算子实现逻辑错误
- 数据类型转换错误
- 容差设置过严
- Golden 数据计算错误

**排查方法**：
1. 检查 Golden 数据是否正确
2. 使用 PASS 模式定位问题 pass
3. 逐步放宽容差
4. 使用 pass_verify_print 和 pass_verify_save 检查中间结果

### 问题 2：没有生成验证结果

**可能原因**：
- `enable_pass_verify` 未设置
- `pass_verify_pass_filter` 配置错误
- Golden 数据未设置

**排查方法**：
```python
# 确保启用了验证
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_pass_filter": ["all"]
    }
)
def my_kernel(x, y, out):
    pass

# 确保设置了 golden 数据
pypto.set_verify_golden_data(goldens=[None, None, golden.cpu()])
```

### 问题 3：CSV 文件为空

**可能原因**：
- `pass_verify_save_tensor` 未设置
- `pass_verify_save_tensor_dir` 权限问题

**排查方法**：
```python
# 确保启用了 tensor 保存
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "./verify_output"  # 确保目录可写
    }
)
def my_kernel(x, y, out):
    pass
```

### 问题 4：pass_verify_print 没有输出

**可能原因**：
- `enable_pass_verify` 未设置
- `cond` 条件不满足

**排查方法**：
```python
# 确保启用了验证
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={"enable_pass_verify": True}
)
def my_kernel(x, y, out):
    pass

# 检查 cond 条件
pypto.pass_verify_print("Debug info", cond=1)  # 总是打印
```

### 问题 5：pass_verify_save 没有生成文件

**可能原因**：
- `pass_verify_save_tensor` 未设置
- `pass_verify_save_tensor_dir` 未设置
- `cond` 条件不满足

**排查方法**：
```python
# 确保启用了 tensor 保存
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "./verify_output"
    }
)
def my_kernel(x, y, out):
    pass

# 检查 cond 条件
pypto.pass_verify_save(tensor, "debug", cond=1)  # 总是保存
```

## 最佳实践

### 1. 容差选择

- **优先检查实现逻辑**：首先确认算法实现是否正确
- **检查数据类型**：确认数据类型转换是否正确
- **根据算子类型选择容差**：复杂算子使用较宽松的容差

### 2. 验证策略

- **开发阶段**：使用 PASS 模式验证所有 pass，快速定位问题
- **测试阶段**：使用 TENSOR_GRAPH 模式验证最终结果
- **调试阶段**：启用 tensor 保存，分析中间结果

### 3. 调试策略

- **使用 pass_verify_print**：快速查看中间值
- **使用 pass_verify_save**：保存中间结果用于离线分析
- **条件性调试**：使用 `cond` 参数减少输出
- **循环调试**：使用占位符保存不同迭代的结果

### 4. 性能考虑

- **生产线上**：禁用 tensor 保存，只进行精度验证
- **调试时**：启用 tensor 保存，但注意不过磁盘空间
- **大规模测试**：使用合理的 pass 过滤，避免过多的验证

## 参考资料

- **Python 接口**：
  - `python/pypto/config.py`
  - `python/pypto/runtime.py`
  - `python/pypto/op/verify.py`

- **测试示例**：
  - `python/tests/st/interface/test_verify_frontend_jit.py`
  - `python/tests/st/interface/test_verify_jit.py`
  - `python/tests/ut/interface/test_verify.py`

- **数据处理工具**：
  - `tools/dump_tensor/process_dump_tensor.py`

## 总结

PyPTO 精度工具提供了完整的精度验证能力：

1. **两种验证模式**：TENSOR_GRAPH（粗检）、PASS（自检）
2. **完整的调试流程**：确认问题合法性 → 缩小问题规模 → 工具自检 → 上板 dump
3. **灵活的配置选项**：通过 jit 装饰器的 verify_options 参数配置
4. **强大的调试工具**：pass_verify_print 和 pass_verify_save
5. **上板 dump 功能**：支持 NPU 实际执行结果 dump
6. **详细的结果输出**：控制台输出、CSV 文件、tensor 数据文件
7. **多种数据类型支持**：支持所有常见的数据类型
8. **完整的误差判定机制**：零值检测、误差检测、阈值判定

通过合理使用这些功能，可以有效地验证算子精度，快速定位和解决精度问题，是算子开发过程中的重要辅助工具。
