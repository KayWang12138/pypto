# pypto.uniform

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |

## 功能说明

uniform 操作使用 Philox 算法生成均匀分布的随机数。Philox 是一种基于计数器的随机数生成器，通过密钥（key）和计数器（counter）产生确定性的均匀随机序列。

该操作生成的随机数服从均匀分布，取值范围为 [0, 1)。通过指定相同的 key 和 counter，可以复现相同的随机数序列，适用于需要确定性随机数的场景，如模型训练中的 dropout、参数初始化等。

## 函数原型

```python
uniform(
    key: int,
    counter0: int,
    counter1: int,
    shape: List[int],
    rounds: int = 10,
    dtype: DataType = None
) -> Tensor
```

## 参数说明

| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| key         | 输入      | 支持的类型为：int。<br> 一个 uint64 值，作为均匀随机数序列的密钥。 |
| counter0    | 输入      | 支持的类型为：int。<br> 128 位计数器的第一个 uint64 值。 |
| counter1    | 输入      | 支持的类型为：int。<br> 128 位计数器的第二个 uint64 值。 |
| shape       | 输入      | 支持的类型为：List[int]。<br> 输出张量的形状，目前仅支持 1 维。 |
| rounds      | 输入      | 支持的类型为：int。<br> Philox 算法的轮数，支持 7 或 10，默认为 10。 |
| dtype       | 输入      | 支持的类型为：DataType。<br> 输出张量的数据类型，支持 DT_FP32、DT_FP16、DT_BF16，默认为 DT_FP32。 |

## 返回值说明

result ：Tensor，形状由 shape 参数指定，数据类型由 dtype 参数指定。张量中的元素为均匀分布的随机数，取值范围为 [0, 1)。

## 约束说明

1. shape 必须为 1 维列表，如 [1024]，不支持多维形状。
2. rounds 参数仅支持 7 或 10。
3. dtype 仅支持 DT_FP32、DT_FP16、DT_BF16。
4. 通过相同的 key、counter0、counter1 参数可以复现相同的随机数序列。

## 调用示例

### TileShape设置示例

说明：调用该 operation 接口前，应通过 set_vec_tile_shapes 设置 TileShape。

TileShape 维度应和输出一致。

示例：输出 shape 为 [1024]，TileShape 设置为 [256]，则 256 用于切分输出张量。

```python
pypto.set_vec_tile_shapes(256)
```

### 接口调用示例

```python
import pypto

key = 12345678901234
counter0 = 0
counter1 = 0
shape = [1024]
rounds = 10

output = pypto.uniform(key, counter0, counter1, shape, rounds, pypto.DT_FP32)
```

### 完整调用示例

```python
import pypto
import math

output_shape = (256,)
view_shape = (128,)
tile_shape = (64,)

pypto.runtime._device_init()

output = pypto.tensor(output_shape, pypto.DT_FP32, "PTO_TENSOR_output")

loop_num = math.ceil(output_shape[0] / view_shape[0])

key = 12345678901234
counter0 = 0
counter1 = 0
rounds = 10

with pypto.function("MAIN", output):
    for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
        offset = idx * view_shape[0]
        
        pypto.set_vec_tile_shapes(tile_shape[0])
        res = pypto.uniform(key, counter0, counter1, view_shape, rounds, pypto.DataType.DT_FP32)
        pypto.assemble(res, [offset], output)

pypto.runtime._device_fini()
```

### 不同数据类型示例

```python
import pypto

key = 12345678901234
counter0 = 0
counter1 = 0
shape = [1024]

out_fp32 = pypto.uniform(key, counter0, counter1, shape, rounds=10, dtype=pypto.DT_FP32)

out_fp16 = pypto.uniform(key, counter0, counter1, shape, rounds=10, dtype=pypto.DT_FP16)

out_bf16 = pypto.uniform(key, counter0, counter1, shape, rounds=10, dtype=pypto.DT_BF16)
```
