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
精度问题定位流程
    │
    ├─ 第一阶段：基础预检
    │   └─ 基础预检
    │
    ├─ 第二阶段：常见场景排查
    │   ├─ 切块语义不匹配
    │   ├─ 多轮通信需 data_clear
    │   ├─ 依赖关系配置错误
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

在进行精度定位前，需先完成以下基础检查，确保问题确实是精度问题而非功能异常或环境问题。

### 精度误差阈值检查
- [ ] 阈值检查

**检查目的**：确保误差阈值设置符合数据类型的精度范围和累加操作的特性。

通信算子中使用 `shmem_put` 的 `AtomicType.ADD` 进行累加时，累加精度由 `shmem_tensor` 的 dtype 决定，阈值需根据累加方式和累加次数调整：

| 累加精度 | 建议阈值 | 说明 |
|---------|---------|------|
| FP32 累加 | ≤ 5e-3（千分之五） | FP32 精度较高，累加误差较小，阈值应满足千分之五 |
| FP16 累加 | 按累加次数计算 | FP16 mantissa 10 bit，相对精度约 1/1024，需按累加次数计算阈值 |
| BF16 累加 | 按累加次数计算 | BF16 mantissa 7 bit，相对精度约 1/128，需按累加次数计算阈值 |

**低精度累加阈值计算方法**：

```python
def calculate_accumulation_threshold(
    dtype: str, 
    world_size: int, 
    safety_factor: float = 2.0
):
    """
    计算低精度累加的理论误差阈值
    
    精度特性：
    - FP16: mantissa 10 bit，相对精度约为 1/1024 ≈ 0.001
    - BF16: mantissa 7 bit，相对精度约为 1/128 ≈ 0.0078
    
    每次累加的舍入误差约为精度类型的 0.5 倍（ulp）
    累加 N 次后，误差粗略估计为 N * 0.5 * dtype_precision
    
    Args:
        dtype: 数据类型，"FP16" 或 "BF16"
        world_size: 累加次数（通信的 rank 数量）
        safety_factor: 安全系数，建议 2.0 以覆盖更多误差来源
    
    Returns:
        建议的相对误差阈值
    """
    # 精度类型对应的相对精度
    precision_map = {
        "FP16": 1 / 1024,   # mantissa 10 bit
        "BF16": 1 / 128,    # mantissa 7 bit
    }
    
    dtype_precision = precision_map.get(dtype)
    if dtype_precision is None:
        raise ValueError(f"Unsupported dtype: {dtype}, only FP16/BF16 supported")
    
    # 单次累加误差估计
    single_accumulation_error = 0.5 * dtype_precision
    
    # N 次累加后的累积误差（线性估计）
    accumulated_error = world_size * single_accumulation_error
    
    # 应用安全系数
    threshold = accumulated_error * safety_factor
    
    return threshold


# 示例计算
print("FP16 累加阈值示例：")
for ws in [2, 4, 8, 16]:
    threshold = calculate_accumulation_threshold("FP16", ws)
    print(f"  world_size={ws}: {threshold:.6f}")

print("\nBF16 累加阈值示例：")
for ws in [2, 4, 8, 16]:
    threshold = calculate_accumulation_threshold("BF16", ws)
    print(f"  world_size={ws}: {threshold:.6f}")
```

**输出示例**：

```
FP16 累加阈值示例：
  world_size=2: 0.001953
  world_size=4: 0.003906
  world_size=8: 0.007812
  world_size=16: 0.015625

BF16 累加阈值示例：
  world_size=2: 0.015625
  world_size=4: 0.031250
  world_size=8: 0.062500
  world_size=16: 0.125000
```
**说明**：

- 以上为线性估计，实际误差可能因数据分布、数值范围等因素有所不同
- **FP16 的动态范围有限**：最大值为 65504，累加时需注意溢出问题
- **大数吃小数问题**：若数据数值差异较大（如大值和小值混合累加），误差可能显著高于理论值
- 对于精度要求较高的场景，建议使用 FP32 进行累加以获得更高精度（参见 2.4.1）

#### 检查要点
- Atomic Add 累加操作需根据累加精度和累加次数调整阈值：
  - FP32 累加：阈值应满足千分之五（≤ 5e-3）
  - FP16 累加：阈值按累加次数计算，world_size=8 时约千分之八（~0.008）
  - BF16 累加：阈值按累加次数计算，world_size=8 时约千分之六十二（~0.062）
- 建议根据实际数据类型和累加方式调整阈值，避免阈值过严导致误报

### Device 日志检查

- [ ] plog 日志无 error 报错，排除编译问题
- [ ] Device 日志无 error 报错，排除功能问题

**检查目的**：排除编译问题和功能异常，确保是纯精度问题。

**操作步骤**：

```bash
# 1. 检查 plog 日志（用于检查编译阶段的错误）
grep -i "error" $ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-*.log

# 2. 检查 Device 日志（日志路径由 ASCEND_PROCESS_LOG_PATH 环境变量指定）
grep -i "error" $ASCEND_PROCESS_LOG_PATH/debug/device*/device*.log
```

**检查要点**：

- plog 日志中无 error 报错表明编译阶段正常
- Device 日志中无 error 报错表明算子功能执行正常
- 若 plog 中存在 Pass 报错，需先定位并修复编译问题
- 若存在 AICore error、AICPU error 等错误，需先定位并修复功能问题
- 编译问题与功能问题应优先解决，再进行精度定位

### 通信算子标准测试用例检查

- [ ] 通信算子标准测试用例能正常执行且精度无异常

**检查目的**：排除环境配置问题，确认通信基础功能正常。

**操作步骤**：

**C++ 测试用例**（需配置 mpirun）：

```bash
# 进入测试执行目录
cd build/output/bin

# 执行 AllReduce 测试用例（world_size 为参与通信的卡数）
mpirun -n 4 ./tile_fwk_stest_distributed run \
    --gtest_filter=TestDistributedOps/DistributedTest.TestOps/0 \
    --frontend=cpp

# 执行所有通信算子测试用例
mpirun -n 4 ./tile_fwk_stest_distributed run \
    --gtest_filter=TestDistributedOps/DistributedTest.TestOps \
    --frontend=cpp
```

**Python 测试用例**：

```bash
# 进入测试用例目录（从 pypto 工程根目录）
cd models/experimental/distributed

# 执行通信算子测试用例
python3 test_matmul_allreduce_add_rmsnorm.py
```

**检查要点**：

- C++ 测试用例需使用 `mpirun` 启动多进程通信
- Python 测试用例框架内部已集成多进程启动机制，无需配置 mpirun
- 若标准用例执行失败或精度异常，说明环境配置可能有问题
- 若标准用例正常但当前算子异常，问题可能在于算子实现本身

上述三项检查通过后，可确认问题为纯精度问题，进入后续定位阶段。

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
- **信号量语义不匹配，可能导致精度异常**

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

### 2.2 多轮通信场景需先进行 data_clear

**问题现象**：多轮通信场景下，首轮数据影响后续轮次，精度异常。

**原因分析**：多轮通信需在每轮前执行 `shmem_clear_data` 清理共享内存，且 clear 之后必须进行屏障同步，确保所有 rank 清理完成后再开始通信。

**正确配置**：

```python
# 每轮通信前：先清理，再屏障同步
data_clear_out = pypto.distributed.shmem_clear_data(shmem_tensor, shmem_shape, [0, 0])
signal_clear_out = pypto.distributed.shmem_clear_signal(shmem_tensor)

# clear 之后必须执行 barrier，确保所有 rank 清理完成
barrier_out = pypto.distributed.shmem_barrier_all(
    shmem_barrier_signal, [data_clear_out, signal_clear_out])

# 后续通信操作需依赖 barrier_out
put_out = pypto.distributed.shmem_put(..., pred=[barrier_out])
```

**关键说明**：

- **data_clear 后必须 barrier**：若 clear 后直接开始通信，部分 rank 可能还在清理中，其他 rank 已开始写入，导致数据竞争或首轮数据残留
- **每轮通信前执行 clear**：多轮场景需在每轮前执行 clear，避免首轮数据影响后续轮次

### 2.3 依赖关系配置错误

**问题现象**：精度异常，且每次执行问题表现可能不一样（随机位置异常）。

**原因分析**：依赖关系配置错误或算子语义理解偏差，导致执行顺序或等待条件不符合预期。

#### 常见错误类型

**错误 1：pred 参数传递错误**

`pred` 参数传递错误，导致通信算子执行顺序不符合语义要求，产生数据竞争：

```python
# 错误示例：shmem_get 未依赖 wait_until_out，可能在数据写入完成前读取
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0],
    pred=[barrier_out])  # 错误：未依赖 wait_until_out

# 正确示例：shmem_get 必须依赖 wait_until_out
wait_until_out = pypto.distributed.shmem_wait_until(...)
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0],
    pred=[wait_until_out])  # 正确：依赖 wait_until_out
```

**错误 2：wait_until 等待条件不正确**

`shmem_wait_until` 等待条件设置错误，未等待所有 rank 完成写入，导致读取结果不完整：

```python
# 问题示例（AllReduce）：等待值设置为 1，仅等待一个 rank 完成
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, 1, shmem_shape, [0, 0],  # 错误：等待值应为 world_size
    cmp=pypto.OpType.EQ, 
    clear_signal=True,
    pred=[barrier_out])

# 结果：仅等待一个 rank 完成，其他 rank 数据未写入，读取结果不完整
```

**等待条件根据算子语义确定**，确保与实际累加次数匹配：

```python
# 正确示例（AllReduce）：等待所有 rank 发送完成，等待值为 world_size
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0],
    cmp=pypto.OpType.EQ,
    clear_signal=True,
    pred=[barrier_out])
```

#### 标准通信流程依赖关系

```
shmem_clear_data → shmem_clear_signal → shmem_barrier_all → 
    shmem_put → shmem_signal → shmem_wait_until → shmem_get
```

**关键原则**：

1. **初始化阶段**：clear 和 barrier 必须在通信前执行，确保初始状态正确
2. **写入阶段**：shmem_signal 必须依赖 shmem_put，确保数据写入完成后再通知
3. **读取阶段**：shmem_get 必须依赖 shmem_wait_until，确保所有 rank 写入完成后再读取
4. **等待条件**：wait_until 的等待条件应与算子语义匹配

#### 正确配置示例

```python
# 1. 清理共享内存和信号量
data_clear_out = pypto.distributed.shmem_clear_data(
    shmem_tensor, shmem_shape, [0, 0], pred=[input_tensor])
signal_clear_out = pypto.distributed.shmem_clear_signal(
    shmem_tensor, pred=[input_tensor])

# 2. 屏障同步，确保所有 rank 完成清理
barrier_out = pypto.distributed.shmem_barrier_all(
    shmem_barrier_signal, [data_clear_out, signal_clear_out])

# 3. 数据写入（依赖 barrier）
put_out = pypto.distributed.shmem_put(
    input_tensor, [0, 0], shmem_tensor, dyn_idx,
    put_op=pypto.AtomicType.ADD, pred=[barrier_out])

# 4. 信号量通知（依赖 put）
pypto.distributed.shmem_signal(
    shmem_tensor, dyn_idx, 1, shmem_shape, [0, 0],
    target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out])

# 5. 等待所有 rank 写入完成（依赖 barrier）
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0],
    cmp=pypto.OpType.EQ, clear_signal=True, pred=[barrier_out])

# 6. 数据读取（依赖 wait_until）
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0],
    pred=[wait_until_out], valid_shape=shmem_shape)
```

### 2.4 Golden 计算逻辑不一致

**问题现象**：精度误差存在但误差不大，累加结果与 golden 存在精度差异。

**原因分析**：Golden 计算逻辑与算子实现中的累加精度不一致，导致计算结果存在合理的精度差异。

#### 累加精度差异示例（AllReduce）

**Golden 实现**（CPU 上计算）：

```python
# Golden: 在 CPU 上对 BF16 数据进行累加
# 注意：torch 在 CPU 上执行 BF16 算术运算时，会自动转换为 FP32 做加法，最后将结果转为bf16
def allreduce_golden(inputs):
    result = torch.zeros(shape, dtype=torch.bfloat16)
    for tensor in inputs:  # inputs 为 BF16 类型
        result += tensor  # 实际按 FP32 累加
    return result
```

**算子实现**（NPU 上计算）：

```python
# 算子: shmem_tensor dtype 为 BF16，累加在 BF16 精度下进行
shmem_tensor = pypto.distributed.create_shmem_tensor(
    group_name, world_size, pypto.DT_BF16, shmem_shape)  # BF16 累加

# shmem_put 使用 AtomicType.ADD，累加精度由 shmem_tensor dtype 决定
put_out = pypto.distributed.shmem_put(
    input_tensor, [0, 0], shmem_tensor, target_pe,
    put_op=pypto.AtomicType.ADD, pred=[...])
```

**问题分析**：

| 对比项 | Golden | 算子实现 |
|-------|--------|---------|
| 累加精度 | FP32（torch CPU 自动转换） | BF16 |
| 精度损失 | 小 | 大（BF16 mantissa 仅 7 bit） |
| 结果差异 | - | 累加次数越多，误差越大 |

**解决方法**：确保 Golden 与算子实现使用相同的累加精度：

```python
# 算子使用 FP32 累加（与 Golden 一致）
shmem_tensor = pypto.distributed.create_shmem_tensor(
    group_name, world_size, pypto.DT_FP32, shmem_shape)  # FP32 累加
```

**关键检查点**：

- Golden 的累加精度（CPU 上 torch 对 BF16 的处理方式）
- 算子中 `shmem_tensor` 的 dtype（决定累加精度）
- 确保两者一致，避免计算模式差异导致误差

## 第三阶段：通信算子精度定位思路

### 3.1 精度问题的可复现性分析法

通过多次执行观察精度异常的表现形式，初步判断问题类型：

#### 情况 1：稳定位置精度异常

**特征**：多次执行后（固定随机数种子），精度异常的位置和误差值稳定一致。

**问题推断**：切分或数据处理逻辑问题，而非同步问题。

**定位方向**：参考第二阶段常见场景排查：

- **2.1 切块语义不匹配**：检查 shmem_signal 与 shmem_put、shmem_get 与 shmem_wait_until 的 TileShape 是否匹配
- **2.4 Golden 计算逻辑不一致**：检查累加精度是否一致
- **数据类型处理**：检查输入输出数据类型转换是否正确

#### 情况 2：随机位置精度异常

**特征**：多次执行后，精度异常的位置和误差值不一致，甚至部分执行正常、部分执行异常。

**问题推断**：同步问题或数据竞争，可能涉及：

- **依赖关系配置错误**（参见 2.3）：pred 参数传递错误，导致执行顺序不符合语义
- **缺少屏障同步**（参见 2.2）：clear 后未执行 barrier，导致数据竞争
- **信号量语义不匹配**（参见 2.1）：写入和等待的数据范围不一致

**定位方向**：重点检查第二阶段中的依赖关系和同步机制：

- 验证 `pred` 参数是否正确传递
- 验证 `shmem_barrier_all` 是否在首轮通信前执行
- 验证切分逻辑是否正确，可参考 [通信算子切块设置指南](distributed_operatoion_tiling_guide.md)

### 3.2 减小问题规模定位法

缩小问题规模旨在简化场景，提高定位效率。缩小后可查看 Pass 图是否符合预期，分析各算子的连接关系、数据流、切分方式。需确保缩小后仍能复现原始问题。

通常通过以下方法缩小问题规模：

**方法 1：减小 world_size**

将 world_size 从大值减小到最小可复现值（如 world_size = 2）。若 world_size = 2 正常但更大值异常，问题可能与累加次数相关（参见 2.4）。

**方法 2：减少切块数量或不切块**

增大 TileShape 减少切块数量，或直接使用不切分的配置。不切分时问题消失，说明切分逻辑有问题（参见 2.1）；不切分时问题仍存在，说明问题与切分无关。

**方法 3：减小 Shape 规格**

调小模型的 Shape 规格（如 batch_size、hidden_size），降低计算和通信规模，便于直接对计算结果进行分析。

### 3.3 定位出现问题的算子

定位首个出现精度问题的算子，详细方法参考 [精度调试指南](../debug/precision.md)。

**核心方法**：

- **二分法定位**：逐步移除尾部计算，找到首个异常算子。对于通信算子，可在 shmem_get 之后设置检查点，验证 shmem_put 写入 win 区的数据是否正确
- **中间结果对比**：对比各算子输出与 golden 中间结果，确定差异首次出现的位置
- **精度工具**：当前通信算子暂不支持，待支持后可通过 Pass 阶段模拟计算与基准数据对比，定位首个异常计算节点

### 定位思路总结

```
定位思路：
    │
    ├─ 思路 1：确认可复现性
    │   ├─ 稳定位置 → 检查切分逻辑（2.1）
    │   └─ 随机位置 → 检查同步机制（2.2、2.3）
    │
    ├─ 思路 2：减小问题规模
    │   ├─ 减小 world_size（如从 8 减到 2）
    │   ├─ 减少切块数量或不切块
    │   └─ 减小 Shape 规格
    │
    └─ 思路 3：定位出问题的算子
        ├─ 方法 1：二分法定位
        │   ├─ 设置检查点，逐步移除尾部计算
        │   ├─ 执行并检查首个异常检查点
        │   └─ 定位到具体算子后，进一步分析
        │
        └─ 方法 2：精度工具（当前通信算子暂不支持）
            ├─ 待支持后，可参考 [精度调试指南](../debug/precision.md)
            └─ 通过 Pass 阶段模拟计算与基准数据对比，定位首个异常计算节点
```

## 第四阶段：验证与回归

修复后需验证精度并回归测试，确保问题彻底解决。

**验证要点**：

- 与 golden 数据对比，确认精度达标
- 验证多卡一致性
- 验证可复现性（多次执行结果一致）

**回归测试范围**：

- 原问题场景验证
- 不同 shape 规格测试
- 不同 world_size 测试
- 多轮执行稳定性测试