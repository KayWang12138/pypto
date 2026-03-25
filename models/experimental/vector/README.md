# scatter_nd_sub

## 功能说明

`scatter_nd_sub` 算子实现了对目标张量指定位置的减法更新操作。该算子将 updates 张量中的值从 target 张量的指定索引位置减去，支持对同一位置的多次更新（累加模式）。

该算子对应 TensorFlow 的 `tf.scatter_nd_sub` 操作，常用于稀疏更新场景，如嵌入表更新、梯度更新等。

## 数学公式

$$
\text{target}[\text{indices}[i], :] = \text{target}[\text{indices}[i], :] - \text{updates}[i, :]
$$

其中：
- $\text{target}$：目标张量，将被原地修改
- $\text{indices}$：索引张量，指定要更新的位置
- $\text{updates}$：更新值张量，包含要从目标位置减去的值

对于重复索引的情况，多次减法操作会累加执行。

## 函数原型

```Python
def scatter_nd_sub_kernel(
    target: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    updates: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32)
) -> None:
```

## 参数说明

> **说明：**
> - M 表示目标张量的第一维度大小
> - N 表示目标张量的第二维度大小（特征维度）
> - K 表示更新操作的数量（indices 和 updates 的第一维度）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|--------|-----------|------|----------|----------|-------------|
| target | 输入/输出 | 目标张量，将被原地修改 | float32 | ND | [M, N] |
| indices | 输入 | 索引张量，指定要更新的位置 | int32 | ND | [K, 1] |
| updates | 输入 | 更新值张量，包含要从目标位置减去的值 | float32 | ND | [K, N] |

## 实现原理

该算子通过以下步骤实现：

1. 使用 `pypto.loop_unroll` 对 indices 进行分块处理，支持 [2048, 1024, 512, 256, 1] 等多种分块大小
2. 对每个分块：
   - 提取当前批次的 indices 和 updates
   - 将 updates 取负（乘以 -1）
   - 使用 `pypto.index_put_` 将负值累加到目标位置

## 调用示例

```Python
import torch
import pypto

# 准备数据
target = torch.rand([10000, 16], dtype=torch.float32)
indices = torch.randint(0, 10000, [1024, 1], dtype=torch.int32)
updates = torch.rand([1024, 16], dtype=torch.float32)

# 调用算子
scatter_nd_sub_kernel(target, indices, updates)
```

## 精度验证

本算子使用 TensorFlow 的 `tf.compat.v1.scatter_nd_sub` 作为基准进行精度验证，最大误差容忍度为 1e-1。

## 支持的输入场景

算子已验证以下输入场景：

| target shape | indices shape | updates shape |
|--------------|---------------|---------------|
| [10, 16] | [1024, 1] | [1024, 16] |
| [1000000, 16] | [12900, 1] | [12900, 16] |
| [104007, 8] | [2048, 1] | [2048, 8] |
| [22692, 8] | [2048, 1] | [2048, 8] |
| [295467, 32] | [512, 1] | [512, 32] |
| [3000000, 32] | [201892, 1] | [201892, 32] |
| [301044, 32] | [512, 1] | [512, 32] |
| [32449, 16] | [1024, 1] | [1024, 16] |
| [4635, 8] | [2048, 1] | [2048, 8] |
| [6000000, 32] | [342312, 1] | [342312, 32] |
| [634054, 32] | [512, 1] | [512, 32] |
| [875000, 8] | [22704, 1] | [22704, 8] |
| [9153, 16] | [1024, 1] | [1024, 16] |
| [934708, 64] | [256, 1] | [256, 64] |

## 详细实现
- 详见 [scatter_nd_sub.py](./scatter_nd_sub.py)

---

# avg_pool2d

## 功能说明

`avg_pool2d` 算子实现了 2D 平均池化操作。该算子对输入张量应用滑动窗口进行平均池化，支持 SAME 和 VALID 两种填充模式。

该算子对应 TensorFlow 的 `tf.nn.avg_pool2d` 操作，常用于卷积神经网络中的下采样操作，减少特征图的空间维度。

## 数学公式

$$
\text{output}[n, c, oh, ow] = \frac{1}{k_h \times k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} \text{input}[n, c, oh \times s_h + i, ow \times s_w + j]
$$

其中：
- $\text{input}$：输入张量 (batch_size, channels, in_h, in_w)
- $\text{output}$：输出张量 (batch_size, channels, out_h, out_w)
- $k_h, k_w$：池化窗口的高度和宽度
- $s_h. s_w$：步长的高度和宽度
- $oh, ow$：输出特征图的位置索引

对于 SAME padding：
$$
out_h = \lceil \frac{in_h}{s_h} \rceil, \quad out_w = \lceil \frac{in_w}{s_w} \rceil
$$

对于 VALID padding:
$$
out_h = \lceil \frac{in_h - k_h + 1}{s_h} \rceil. \quad out_w = \lceil \frac{in_w - k_w + 1}{s_w} \rceil
$$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|--------|-----------|------|----------|----------|-------------|
| input_tensor | 输入 | 输入特征图 | float32 | ND | [batch_size, channels, in_h, in_w] |
| output_result | 输出 | 输出特征图 | float32 | ND | [batch_size, channels, out_h, out_w] |
| kernel_size | 属性 | 池化窗口大小 | tuple | - | (k_h, k_w) |
| stride | 属性 | 步长大小 | tuple | - | (s_h, s_w)，默认为 kernel_size |
| padding_mode | 属性 | 填充模式 | string | - | 'SAME' 或 'VALID' |

## 实现原理

该算子通过以下步骤实现：

1. 使用 `PoolParams` dataclass 封装所有池化参数（batch_size, channels, kernel_size, stride, padding 等）
2. 使用 `pypto.loop_unroll` 对 batch*channel 维度进行分块处理，支持 [8, 4, 2, 1] 等多种分块大小
3. 对每个输出位置 (oh, ow)：
   - 计算输入特征的滑动窗口范围，考虑 padding 偏移
   - 对窗口内的数据进行高度方向的求和（reduce）
   - 对宽度方向进行求和并计算平均值
   - 使用 `pypto.assemble` 将结果写入输出位置

## 核心优化

1. **参数封装优化**：使用 `@dataclass(frozen=True)` 封装池化参数，提升代码可读性和类型安全性
2. **向量化计算**：Golden 实现使用 `np.mean(window, axis=(2,3))` 进行批量计算，减少循环层数
3. **动态 shape 支持**：支持 batch_size 和 channels 的动态 shape
4. **循环展开**：支持多种 unroll 大小以适应不同输入规模

## 调用示例

```Python
import torch
import pypto

# 准备数据
batch_size, channels, in_h, in_w = 2, 3, 6, 6
x = torch.rand([batch_size, channels, in_h, in_w], dtype=torch.float32)
y = torch.empty([batch_size, channels, 3, 3], dtype=torch.float32)

# 创建并调用算子
kernel = avg_pool_2d(
    shape=x.shape,
    kernel_size=(2, 2),
    stride=(2, 2),
    padding_mode='SAME',
    run_mode='npu',
    dynamic=True
)
kernel(x, y)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
```

## 支持的测试用例

| batch_size | channels | in_h | in_w | kernel_size | stride | padding_mode | out_h | out_w |
|-----------|----------|------|------|-------------|--------|--------------|-------|-------|
| 2 | 3 | 6 | 6 | (2, 2) | (2, 2) | SAME | 3 | 3 |
| 4 | 8 | 12 | 12 | (3, 3) | (2, 2) | VALID | 5 | 5 |

## 详细实现

- 详见 [avg_pool2d.py](./avg_pool2d.py)
