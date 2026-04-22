# PyPTO 通信算子精度定位问题指南

## 概述

PyPTO 通信算子用于实现多卡间的数据传输与同步，是分布式计算场景的核心组件。与普通计算算子相比，通信算子涉及跨进程数据交换、信号量同步、切块策略等复杂机制，其精度问题定位具有独特的挑战性。

本指南提供系统化的通信算子精度问题定位方法，帮助开发者快速定位和解决精度异常。

## 通信算子精度问题特点

与普通算子相比，通信算子精度问题具有以下特殊性：

| 特性             | 普通算子     | 通信算子                                            |
| ---------------- | ------------ | --------------------------------------------------- |
| **执行环境**     | 单卡执行     | 多卡并行执行                                        |
| **数据流**       | 单向数据传递 | 多向数据交换（Put/Get）                             |
| **同步机制**     | 不需要同步   | 需信号同步（signal/wait_until）                     |
| **内存管理**     | 普通内存     | 共享内存（shmem）                                   |
| **问题来源**     | 计算逻辑错误 | 计算逻辑错误 + 同步问题 + 数据切分问题              |

## 精度定位整体流程

```
精度问题发现
    │
    ├─ 第一阶段：基础预检
    │   ├─ 问题确认与可复现性验证
    │   ├─ 环境与版本检查
    │   └─ 算子代码逻辑审查
    │
    ├─ 第二阶段：通信配置检查
    │   ├─ 切块策略检查
    │   ├─ 通信依赖关系检查
    │   ├─ 信号量配置检查
    │   └─ valid_shape 检查
    │
    ├─ 第三阶段：中间结果定位
    │   ├─ 上板执行 tensor dump
    │   ├─ 数据对比分析
    │   └─ 问题定位到具体算子
    │
    └─ 第四阶段：问题修复与验证
        ├─ 修复方案实施
        ├─ 精度验证
        └─ 回归测试
```

## 第一阶段：基础预检

### 1.1 问题确认与可复现性验证

**检查项**：

- [ ] 精度误差阈值是否合理（考虑 BF16 等低精度类型）
- [ ] 问题能否稳定复现（多次执行结果一致）
- [ ] 不同环境下问题表现是否一致
- [ ] Device 日志无 error 报错，排除功能问题（确保是精度问题而非功能异常）

**操作示例**：

**检查 Device 日志**：

```bash
# 检查 Device 日志（日志路径由 ASCEND_PROCESS_LOG_PATH 环境变量指定）
grep -i "error" $ASCEND_PROCESS_LOG_PATH/debug/device*/device*.log
```

**说明**：

- Device 日志中无 error 报错表明算子功能执行正常，问题为纯精度问题
- 若 Device 日志存在 error（如 AICore error、AICPU error），需先定位并修复功能问题


### 1.2 环境与版本检查

**检查项**：

- [ ] CANN 软件版本正确
- [ ] PyPTO 版本正确
- [ ] NPU 硬件状态正常（使用 `npu-smi info` 检查）
- [ ] MPI 环境配置正确（仅 C++ 通信用例需要，Python 通信用例无需 MPI）
- [ ] 通信 st 用例能正常执行且精度无异常（排除环境问题）

### 1.3 算子代码逻辑审查

**重点检查**：

- [ ] 通信流程是否符合标准范式（create → put → signal → wait → get）；多轮通信场景需在首轮前执行 clear 和 barrier
- [ ] Golden 计算逻辑是否与算子实现一致，特别是低精度类型（BF16/FP16）的累加操作：golden 是直接累加还是先 cast 到 FP32 再累加

**标准通信流程示例**：

```python
def standard_communication_flow(input_tensor, group_name, world_size):
    shmem_shape = [row_size, col_size]
    
    shmem_tensor = pypto.distributed.create_shmem_tensor(
        group_name, world_size, pypto.DT_FP32, shmem_shape)
    shmem_barrier_signal = pypto.distributed.create_shmem_signal(group_name, world_size)
    my_pe = pypto.distributed.my_symbolic_pe(group_name)
    
    data_clear_out = pypto.distributed.shmem_clear_data(
        shmem_tensor, shmem_shape, [0, 0])
    signal_clear_out = pypto.distributed.shmem_clear_signal(shmem_tensor)
    barrier_out = pypto.distributed.shmem_barrier_all(
        shmem_barrier_signal, [data_clear_out, signal_clear_out])
    
    for dyn_idx in range(world_size):
        put_out = pypto.distributed.shmem_put(
            input_tensor, [0, 0], shmem_tensor, dyn_idx,
            put_op=pypto.AtomicType.ADD, pred=[barrier_out])
        pypto.distributed.shmem_signal(
            shmem_tensor, dyn_idx, 1, shmem_shape, [0, 0],
            target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out])
    
    wait_until_out = pypto.distributed.shmem_wait_until(
        shmem_tensor, my_pe, world_size, shmem_shape, [0, 0],
        cmp=pypto.OpType.EQ, clear_signal=True)
    
    output = pypto.distributed.shmem_get(
        shmem_tensor, my_pe, shmem_shape, [0, 0],
        pred=[wait_until_out], valid_shape=shmem_shape)
    
    return output
```

## 第二阶段：通信配置检查

### 2.1 切块策略检查

切块配置是通信算子精度问题的关键因素。错误的切块配置会导致信号量语义不匹配、数据读写时机错误等问题。

#### 检查项一：shmem_signal 与 shmem_put 切块语义匹配

**约束**：`shmem_signal` 写入的信号量标识的数据块大小必须与 `shmem_put` 实际写入的数据块大小匹配。

**问题示例**：

```python
pypto.set_vec_tile_shapes(16, 64)
put_dummy = pypto.distributed.shmem_put(input_tensor, [0, 0], shmem_tensor, 0,
              put_op=pypto.AtomicType.ADD, pred=[input_tensor])
pypto.set_vec_tile_shapes(33, 64)
pypto.distributed.shmem_signal(shmem_tensor, 0, 1, shmem_shape,
              [0, 0], target_pe=0, sig_op=pypto.AtomicType.ADD, pred=[put_dummy])
```

**问题分析**：

- `shmem_put` TileShape 为 [16, 64]，数据被切分为 4 块
- `shmem_signal` TileShape 为 [33, 64]，信号量被切分为 2 块
- `tile_shmem_signal0` 对应 `tile_shmem_put0` 和 `tile_shmem_put1`
- `tile_shmem_signal0` 标识写入 [33, 64] 数据，实际只写入 [32, 64] 数据
- **信号量语义不匹配，可能导致精度异常**

**正确配置示例**：

```python
input_shape = [64, 64]

pypto.set_vec_tile_shapes(16, 64)
put_dummy = pypto.distributed.shmem_put(input_tensor, [0, 0], shmem_tensor, 0,
              put_op=pypto.AtomicType.ADD, pred=[input_tensor])

pypto.set_vec_tile_shapes(32, 64)
pypto.distributed.shmem_signal(shmem_tensor, 0, 1, input_shape,
              [0, 0], target_pe=0, sig_op=pypto.AtomicType.ADD, pred=[put_dummy])
```

**验证逻辑**：

- `shmem_put` 将数据切分为 4 块：每块 [16, 64]
- `shmem_signal` 将信号量切分为 2 块：每块对应 [32, 64]
- 第 1 个 signal 块覆盖前 2 个 put 块（共 [32, 64]）
- **信号量语义正确匹配**

#### 检查项二：shmem_get 与 shmem_wait_until 切块语义匹配

**约束**：`shmem_get` 读取的数据块大小必须与 `shmem_wait_until` 等待的信号量对应的数据块大小匹配。

**问题示例**：

```python
pypto.set_vec_tile_shapes(16, 64)
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0], cmp=pypto.OpType.EQ)

pypto.set_vec_tile_shapes(33, 64)
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out])
```

**问题分析**：

- `shmem_wait_until` TileShape 为 [16, 64]，等待 4 个信号量块
- `shmem_get` TileShape 为 [33, 64]，读取 2 个数据块
- `tile_shmem_get0` 对应等待前 2 个 wait_until 块
- 前两个 wait_until 块对应的数据区域为 [32, 64]
- **但 `tile_shmem_get0` 试图读取 [33, 64] 数据，超出实际写入范围**

**正确配置示例**：

```python
pypto.set_vec_tile_shapes(16, 64)
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0], cmp=pypto.OpType.EQ)

pypto.set_vec_tile_shapes(32, 64)
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out], valid_shape=shmem_shape)
```

### 2.2 通信依赖关系检查

通信算子间的执行依赖关系通过 `pred` 参数和 dummy tensor 控制，正确的依赖配置确保算子按语义顺序执行：
- **写信号量必须在写数据之后执行**：确保数据写入完成后才通知远端 rank
- **读数据必须在等待信号量之后执行**：确保远端 rank 数据写入完成后再读取
- **多轮通信需在首轮前执行 clear 和 barrier**：确保共享内存和信号量初始状态正确

#### 检查项一：pred 参数正确传递

**标准依赖关系**：

```
shmem_clear → shmem_barrier_all → shmem_put → shmem_signal → shmem_wait_until → shmem_get
```

**检查方法**：

逐个检查每个通信算子的 `pred` 参数：

```python
data_clear_out = pypto.distributed.shmem_clear_data(
    shmem_tensor, shmem_shape, [0, 0], pred=[input_tensor])

signal_clear_out = pypto.distributed.shmem_clear_signal(
    shmem_tensor, pred=[input_tensor])

barrier_out = pypto.distributed.shmem_barrier_all(
    shmem_barrier_signal, [data_clear_out, signal_clear_out])

put_out = pypto.distributed.shmem_put(
    input_tensor, [0, 0], shmem_tensor, dyn_idx,
    put_op=pypto.AtomicType.ADD, pred=[barrier_out])

pypto.distributed.shmem_signal(
    shmem_tensor, dyn_idx, 1, shmem_shape, [0, 0],
    target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out])

wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0],
    cmp=pypto.OpType.EQ, clear_signal=True, pred=[barrier_out])

output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0],
    pred=[wait_until_out], valid_shape=shmem_shape)
```

### 2.3 信号量配置检查

#### 检查项一：信号量比较值正确

`shmem_wait_until` 的比较值应与 `shmem_signal` 累加后的期望值一致。

**正确配置**：

```python
for dyn_idx in range(world_size):
    pypto.distributed.shmem_signal(
        shmem_tensor, dyn_idx, 1, shmem_shape, [0, 0],
        target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD)

wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0],
    cmp=pypto.OpType.EQ, clear_signal=True)
```

**说明**：

- 每个 rank 执行一次 `shmem_signal`，信号量值累加 1
- `world_size` 个 rank 执行完成后，信号量值为 `world_size`
- `shmem_wait_until` 比较值为 `world_size`，使用 `EQ` 比较操作

#### 检查项二：clear_signal 参数正确

`shmem_wait_until` 的 `clear_signal=True` 会在等待完成后清零信号量，为下一次通信做准备。

**检查方法**：

- 如果通信流程在循环中执行，确保 `clear_signal=True`
- 如果是单次通信，可以设置 `clear_signal=False`

### 2.4 valid_shape 检查

动态 shape 场景下，最后一块可能小于固定块大小，需正确设置 `valid_shape` 参数。

**问题示例**：

```python
batch_size = pypto.DYNAMIC
view_row_shape = 8

for bs_idx in range((batch_size + view_row_shape - 1) // view_row_shape):
    in_tensor_tile = pypto.view(
        in_tensor, (view_row_shape, hidden_size), [bs_idx * view_row_shape, 0])
    
    output = pypto.distributed.shmem_get(
        shmem_tensor, my_pe, shmem_shape, [0, 0])
```

**问题分析**：

- 最后一块的 `batch_size - bs_idx * view_row_shape` 可能小于 `view_row_shape`
- 未设置 `valid_shape`，可能读取越界数据

**正确配置**：

```python
for bs_idx in range((batch_size + view_row_shape - 1) // view_row_shape):
    valid_row = (batch_size - bs_idx * view_row_shape).min(view_row_shape)
    in_tensor_tile = pypto.view(
        in_tensor, (view_row_shape, hidden_size), [bs_idx * view_row_shape, 0],
        valid_shape=[valid_row, hidden_size])
    
    output = pypto.distributed.shmem_get(
        shmem_tensor, my_pe, shmem_shape, [0, 0], valid_shape=[valid_row, hidden_size])
```

## 第三阶段：中间结果定位

### 重要说明

**精度调试工具的限制**：

PyPTO 的 `enable_pass_verify` 等前端模拟计算工具**不支持集合通信场景**。通信算子的精度定位需依赖上板执行和 tensor dump。

### 3.1 上板执行 tensor dump

#### 启用方式

```python
import os

os.environ["PTO_DATADUMP_ENABLE"] = "true"

@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True
    }
)
def communication_kernel(...):
    ...
```

#### Dump 数据输出路径

```
output/output_*/dump_tensor_*/device_{deviceId}/
└── {taskId}_{seqNo}_{callopMagic}_{rootHash}_{funcHash}_{rawMagic}_{timeStamp}_{dataType}_{input/output}{index}.tdump
```

#### 数据处理工具

```bash
python3 tools/verifier/parse_dump_tensors.py \
    --dump_tensor_path output/output_*/dump_tensor_*/device_0
```

#### 输出文件

```
output/output_*/dump_tensor_*/device_0/
├── tensor_info.csv              # 解析结果报告
├── *.data                       # 提取的 tensor 数据文件
└── raw_{rawMagic}_{dataType}_{ioflag}.data  # 合并后的 raw tensor
```

### 3.2 数据对比分析

#### 单卡数据对比

对于单卡参与的计算部分（如 matmul、add、rmsnorm），可以与 golden 数据对比：

```python
import torch
import numpy as np

def compare_with_golden(dump_data_path, golden_data):
    dump_data = torch.from_file(dump_data_path)
    golden = golden_data.cpu()
    
    diff = torch.abs(dump_data - golden)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print(f"Elements with inf/nan: {(torch.isinf(dump_data) | torch.isnan(dump_data)).sum()}")
    
    return max_diff < 1e-3
```

#### 多卡数据一致性检查

对于通信结果，检查不同 rank 的数据一致性：

```python
import torch
import torch_npu
import os

def check_multi_rank_consistency(rank_outputs):
    for i in range(1, len(rank_outputs)):
        diff = torch.abs(rank_outputs[i] - rank_outputs[0]).max()
        print(f"Rank {i} vs Rank 0 max diff: {diff}")
        
        if diff > 1e-3:
            print(f"WARNING: Rank {i} has significant difference from Rank 0")
```

### 3.3 问题定位到具体算子

#### 定位方法一：二分法

逐步移除尾部计算，定位首个出现精度问题的算子：

```python
def binary_search_precision_issue():
    original_code = """
        matmul_result = pypto.matmul(...)
        shmem_put(...)
        shmem_signal(...)
        shmem_wait_until(...)
        output = shmem_get(...)
        add_result = pypto.add(output, residual)
        norm_result = rms_norm(add_result)
    """
    
    test_points = [
        ("matmul", "matmul_result = pypto.matmul(...); return matmul_result"),
        ("allreduce", "...output = shmem_get(...); return output"),
        ("add", "...add_result = pypto.add(...); return add_result"),
        ("rmsnorm", "...norm_result = rms_norm(...); return norm_result"),
    ]
    
    for name, code in test_points:
        result = execute_modified_kernel(code)
        if is_precision_issue(result):
            print(f"Precision issue found at: {name}")
            return name
```

#### 定位方法二：中间数据保存

使用 `pypto.pass_verify_save` 保存中间计算结果（仅适用于非通信部分）：

```python
@pypto.frontend.jit(verify_options={"enable_pass_verify": True, "pass_verify_save_tensor": True})
def kernel_with_save(...):
    matmul_result = pypto.matmul(in_tensor, weight)
    pypto.pass_verify_save(matmul_result, "matmul_result")
    
    add_result = pypto.add(allreduce_out, residual)
    pypto.pass_verify_save(add_result, "add_result")
```

**注意**：`pass_verify_save` 不支持通信算子的中间结果保存。

## 第四阶段：问题修复与验证

### 4.1 常见问题修复方案

#### 问题 1：切块语义不匹配

**修复方案**：调整 TileShape 配置，确保信号量与数据块的语义匹配

```python
input_shape = [64, 64]

# 方案 A：统一切块大小
pypto.set_vec_tile_shapes(16, 64)
put_out = pypto.distributed.shmem_put(...)
pypto.distributed.shmem_signal(...)

pypto.set_vec_tile_shapes(16, 64)
wait_out = pypto.distributed.shmem_wait_until(...)
output = pypto.distributed.shmem_get(...)

# 方案 B：信号量切块覆盖整数倍数据块
pypto.set_vec_tile_shapes(16, 64)
put_out = pypto.distributed.shmem_put(...)

pypto.set_vec_tile_shapes(32, 64)
pypto.distributed.shmem_signal(...)

pypto.set_vec_tile_shapes(32, 64)
wait_out = pypto.distributed.shmem_wait_until(...)
output = pypto.distributed.shmem_get(...)
```

#### 问题 2：信号量等待超时

**修复方案**：确保 `shmem_wait_until` 和 `shmem_signal` TileShape 一致

```python
pypto.set_vec_tile_shapes(32, 64)
pypto.distributed.shmem_signal(...)

pypto.set_vec_tile_shapes(32, 64)
wait_out = pypto.distributed.shmem_wait_until(...)
```

#### 问题 3：valid_shape 未设置

**修复方案**：为动态 shape 的通信算子设置 valid_shape

```python
valid_row = (batch_size - bs_idx * view_row_shape).min(view_row_shape)
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0],
    valid_shape=[valid_row, hidden_size])
```

#### 问题 4：数据竞争

**修复方案**：增加屏障同步或调整偏移配置

```python
barrier_out = pypto.distributed.shmem_barrier_all(shmem_barrier_signal, pred=[...])

for dyn_idx in range(world_size):
    put_out = pypto.distributed.shmem_put(
        ..., pred=[barrier_out])
```

### 4.2 精度验证

#### 验证步骤

1. 执行修复后的算子
2. 与 golden 数据对比
3. 验证多卡一致性
4. 验证可复现性

#### 验证代码示例

```python
import torch
import torch_npu

def validate_precision():
    golden = compute_golden_reference()
    
    for _ in range(5):
        torch.npu.synchronize()
        output = execute_kernel()
        torch.npu.synchronize()
        
        max_diff = torch.abs(output.cpu() - golden).max()
        print(f"Max difference: {max_diff}")
        
        if max_diff > threshold:
            print("FAILED: Precision issue persists")
            return False
    
    print("PASSED: Precision validation successful")
    return True
```

### 4.3 回归测试

#### 测试范围

- [ ] 原问题场景验证
- [ ] 不同 shape 规格测试
- [ ] 不同 world_size 测试
- [ ] 多轮执行稳定性测试

#### 测试脚本示例

```python
import torch
import torch_npu

def regression_test():
    test_cases = [
        {"shape": [64, 64], "world_size": 2},
        {"shape": [128, 256], "world_size": 4},
        {"shape": [256, 512], "world_size": 8},
    ]
    
    for case in test_cases:
        for _ in range(10):
            output = execute_kernel(**case)
            if not validate_output(output):
                print(f"FAILED for case: {case}")
                return False
    
    print("PASSED: All regression tests successful")
    return True
```

## 常见问题分类与案例

### 问题分类速查表

| 问题类型           | 典型现象                   | 定位方法                   | 修复方案                   |
| ------------------ | -------------------------- | -------------------------- | -------------------------- |
| 切块语义不匹配     | 特定位置数据错误           | TileShape 配置检查         | 调整切块大小               |
| 信号量等待超时     | 0xA3000 错误               | signal/wait TileShape 检查 | 统一切块大小               |
| valid_shape 未设置 | 尾块数据错误               | valid_shape 参数检查       | 添加 valid_shape 参数      |
| 数据竞争           | 结果随机异常               | pred 参数检查              | 增加屏障同步               |
| 依赖关系错误       | 数据覆盖、时序错误         | 依赖关系流程检查           | 修正 pred 参数传递         |
| 数据类型不一致     | 精度损失                   | dtype 检查                 | 统一数据类型               |

### 案例 1：AllReduce 精度异常

**问题描述**：

MatmulAllReduce 算子在特定 shape 下出现精度异常，部分元素误差超过阈值。

**定位过程**：

1. 基础预检：环境正常，问题可复现
2. 代码审查：发现 `shmem_signal` 与 `shmem_put` TileShape 不匹配
   - `shmem_put` TileShape [16, 64]，数据切分为 4 块
   - `shmem_signal` TileShape [33, 64]，信号量切分为 2 块
3. 问题定位：第 2 个 signal 块标识数据区域 [33-64, 64]，但实际只写入 [32-64, 64] 数据

**修复方案**：

调整 `shmem_signal` TileShape 为 [32, 64]，确保信号量语义与实际写入数据匹配。

**验证结果**：

精度验证通过，回归测试通过。

### 案例 2：信号量等待超时

**问题描述**：

执行 `shmem_wait_until` 时出现错误码 0xA3000 AICPU_TASK_TIMEOUT。

**定位过程**：

1. 错误码查询：0xA3000 表示 AICPU 等待超时
2. 配置检查：发现 `shmem_signal` TileShape [32, 64]，`shmem_wait_until` TileShape [16, 64]
3. 问题定位：信号量切块数量与等待切块数量不匹配，导致等待操作无法找到对应的信号量块

**修复方案**：

统一 `shmem_signal` 和 `shmem_wait_until` 的 TileShape 为 [32, 64]。

**验证结果**：

超时错误消除，通信流程正常执行。

### 案例 3：动态 shape 尾块精度异常

**问题描述**：

动态 batch_size 场景下，最后一块数据出现精度异常。

**定位过程**：

1. 数据分析：发现问题集中在 batch_size 的尾块位置
2. 配置检查：发现 `shmem_get` 未设置 `valid_shape` 参数
3. 问题定位：尾块实际数据量小于固定块大小，未设置 `valid_shape` 导致读取越界数据

**修复方案**：

为 `shmem_get` 添加 `valid_shape` 参数：

```python
valid_row = (batch_size - bs_idx * view_row_shape).min(view_row_shape)
output = pypto.distributed.shmem_get(
    ..., valid_shape=[valid_row, hidden_size])
```

**验证结果**：

尾块精度问题解决，整体精度验证通过。

## 注意事项

### 1. 工具限制

- `enable_pass_verify` 等前端模拟计算工具**不支持集合通信场景**
- 通信算子精度定位需依赖上板执行和 tensor dump
- `pass_verify_save` 不支持通信算子的中间结果保存

### 2. 切块配置原则

- `shmem_signal` 与 `shmem_put` 切块需满足语义匹配
- `shmem_wait_until` 与 `shmem_signal` TileShape 必须一致
- `shmem_get` 与 `shmem_wait_until` 切块需满足语义匹配

### 3. 通信流程规范

- 遵循标准通信流程：create → clear → barrier → put → signal → wait → get
- 确保依赖关系正确传递（pred 参数）
- 多 rank 通信使用屏障同步避免数据竞争

### 4. 数据类型一致性

- 通信算子数据类型需与计算算子一致
- 低精度类型（BF16）需使用合理的误差阈值
- 注意累加操作的数据类型影响

## 参考资料

### 官方文档

| 文档名称                                        | 路径                                                         | 说明                       |
| ----------------------------------------------- | ------------------------------------------------------------ | -------------------------- |
| 分布式算子开发指南                              | [docs/tutorials/distributed](../tutorials/distributed)      | 通信算子开发教程           |
| 通信算子切块设置指南                            | [distributed_operatoion_tiling_guide.md](../tutorials/distributed/distributed_operatoion_tiling_guide.md) | 切块策略详细说明           |
| MatmulAllReduce 融合算子示例                    | [matmul_allreduce_rmsnorm.md](../tutorials/distributed/matmul_allreduce_rmsnorm.md) | AllReduce 融合算子示例     |
| 精度调试指南                                    | [precision.md](../tutorials/debug/precision.md)             | 通用精度调试方法           |
| DISTRIBUTED 错误码                               | [distributed.md](distributed.md)                             | 分布式错误码定义           |

### API 文档

| API 名称                                         | 文档路径                                                     | 说明                       |
| ----------------------------------------------- | ------------------------------------------------------------ | -------------------------- |
| pypto.distributed.create_shmem_tensor           | [docs/api/distributed/pypto-distributed-create_shmem_tensor.md](../api/distributed/pypto-distributed-create_shmem_tensor.md) | 创建共享数据缓冲区         |
| pypto.distributed.shmem_put                     | [docs/api/distributed/pypto-distributed-shmem_put.md](../api/distributed/pypto-distributed-shmem_put.md) | 数据写入                   |
| pypto.distributed.shmem_get                     | [docs/api/distributed/pypto-distributed-shmem_get.md](../api/distributed/pypto-distributed-shmem_get.md) | 数据读取                   |
| pypto.distributed.shmem_signal                  | [docs/api/distributed/pypto-distributed-shmem_signal.md](../api/distributed/pypto-distributed-shmem_signal.md) | 信号量写入                 |
| pypto.distributed.shmem_wait_until              | [docs/api/distributed/pypto-distributed-shmem_wait_until.md](../api/distributed/pypto-distributed-shmem_wait_until.md) | 信号量等待                 |
| pypto.set_vec_tile_shapes                       | [docs/api/config/pypto-set_vec_tile_shapes.md](../api/config/pypto-set_vec_tile_shapes.md) | Vector 算子切块设置        |

### 已知问题

参见 [docs/tutorials/appendix/issue.md](../tutorials/appendix/issue.md)。