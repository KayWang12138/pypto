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
    │   └─ 基础预检
    │
    ├─ 第二阶段：常见场景排查
    │   ├─ 切块语义不匹配
    │   ├─ valid_shape 未设置
    │   ├─ 依赖关系配置错误
    │   ├─ 通信流程不规范
    │   ├─ Golden 计算逻辑不一致
    │   └─ 数据类型不一致
    │
    ├─ 第三阶段：精度定位方法
    │   ├─ 数据对比分析
    │   └─ 问题定位技巧
    │
    └─ 第四阶段：验证与回归
        ├─ 精度验证
        └─ 回归测试
```

## 第一阶段：基础预检

### 1.1 基础预检

**检查项**：

- [ ] 精度误差阈值是否合理（考虑 BF16 等低精度类型）
- [ ] 问题能否稳定复现（多次执行结果一致）
- [ ] Device 日志无 error 报错，排除功能问题（确保是精度问题而非功能异常）
- [ ] 通信 st 用例能正常执行且精度无异常（排除环境问题）

**操作示例**：

**检查 Device 日志**：

```bash
# 检查 Device 日志（日志路径由 ASCEND_PROCESS_LOG_PATH 环境变量指定）
grep -i "error" $ASCEND_PROCESS_LOG_PATH/debug/device*/device*.log
```

**说明**：

- Device 日志中无 error 报错表明算子功能执行正常，问题为纯精度问题
- 若 Device 日志存在 error（如 AICore error、AICPU error），需先定位并修复功能问题
- 通信 st 用例正常执行可排除环境配置问题

## 第二阶段：通信算子精度问题常见场景

### 2.1 切块语义不匹配

#### 2.1.1 shmem_signal 与 shmem_put 切块语义不匹配

**问题现象**：特定位置数据错误，部分元素精度异常。

**原因分析**：`shmem_signal` 写入的信号量标识的数据块大小与 `shmem_put` 实际写入的数据块大小不匹配。

**错误示例**：

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
- **信号量语义不匹配，导致精度异常**

**正确配置**：

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

#### 2.1.2 shmem_get 与 shmem_wait_until 切块语义不匹配

**问题现象**：读取数据区域超出实际写入范围，导致精度异常。

**原因分析**：`shmem_get` 读取的数据块大小与 `shmem_wait_until` 等待的信号量对应的数据块大小不匹配。

**错误示例**：

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

**正确配置**：

```python
pypto.set_vec_tile_shapes(16, 64)
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0], cmp=pypto.OpType.EQ)

pypto.set_vec_tile_shapes(32, 64)
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out], valid_shape=shmem_shape)
```

### 2.2 valid_shape 未设置

**问题现象**：动态 shape 场景下，尾块数据精度异常。

**原因分析**：最后一块可能小于固定块大小，未设置 `valid_shape` 导致读取越界数据。

**错误示例**：

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

### 2.3 依赖关系配置错误

#### 2.3.1 pred 参数传递错误

**问题现象**：数据覆盖、时序错误，结果异常。

**原因分析**：`pred` 参数传递错误，导致通信算子执行顺序不符合语义要求。

**标准依赖关系**：

```
shmem_clear → shmem_barrier_all → shmem_put → shmem_signal → shmem_wait_until → shmem_get
```

**关键原则**：

- **写信号量必须在写数据之后执行**：确保数据写入完成后才通知远端 rank
- **读数据必须在等待信号量之后执行**：确保远端 rank 数据写入完成后再读取

**正确配置示例**：

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

#### 2.3.2 缺少屏障同步导致数据竞争

**问题现象**：结果随机异常，多次执行结果不一致。

**原因分析**：多 rank 通信缺少屏障同步，导致数据竞争。

**正确配置**：

```python
barrier_out = pypto.distributed.shmem_barrier_all(shmem_barrier_signal, pred=[...])

for dyn_idx in range(world_size):
    put_out = pypto.distributed.shmem_put(
        ..., pred=[barrier_out])
```

### 2.4 通信流程不规范

#### 2.4.1 多轮通信缺少 clear/barrier

**问题现象**：多轮通信场景下，首轮数据影响后续轮次，精度异常。

**原因分析**：多轮通信场景需在首轮前执行 clear 和 barrier，确保共享内存和信号量初始状态正确。

**标准通信流程**：

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

### 2.5 Golden 计算逻辑不一致

#### 2.5.1 低精度类型累加处理差异

**问题现象**：精度误差超过预期阈值，累加结果与 golden 不一致。

**原因分析**：Golden 计算逻辑与算子实现不一致，特别是低精度类型（BF16/FP16）的累加操作处理方式不同。

**关键检查点**：

- Golden 是直接累加还是先 cast 到 FP32 再累加
- 算子实现中的累加操作数据类型处理

**建议**：

- 对于低精度类型（BF16/FP16）的累加操作，Golden 和算子实现需保持一致的数据类型处理方式
- 若算子实现先 cast 到 FP32 再累加，Golden 也应采用相同方式

### 2.6 数据类型不一致

**问题现象**：精度损失，数据类型转换导致误差累积。

**原因分析**：通信算子与计算算子的数据类型不一致，或低精度类型使用不当。

**检查点**：

- 通信算子数据类型需与计算算子一致
- 低精度类型（BF16/FP16）需使用合理的误差阈值
- 注意累加操作的数据类型影响

## 第三阶段：精度定位方法

### 3.1 数据对比分析

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

### 3.2 问题定位技巧

#### 二分法定位

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

## 第四阶段：验证与回归

### 4.1 精度验证

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

### 4.2 回归测试

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