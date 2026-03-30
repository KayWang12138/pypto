# avg_pool2d

## 功能说明

`avg_pool2d` 算子实现了 2D 平均池化操作。该算子对输入张量应用滑动窗口进行平均计算，支持 SAME 和 VALID 两种填充模式，常用于卷积神经网络中的下采样操作。

## 数学公式

$$
\text{output}[n, c, oh, ow] = \frac{1}{k_h \times k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} \text{input}[n, c, oh \times s_h + i, ow \times s_w + j]
$$

其中:
- $\text{input}$: 输入张量 (batch_size, channels, in_h, in_w)
- $\text{output}$: 输出张量 (batch_size, channels, out_h, out_w)
- $k_h, k_w$: 池化窗口的高度和宽度
- $s_h, s_w$: 步长的高度和宽度
- $oh, ow$: 输出特征图的位置索引

对于 SAME padding:
$$
out_h = \lceil \frac{in_h}{s_h} \rceil, \quad out_w = \lceil \frac{in_w}{s_w} \rceil
$$

对于 VALID padding:
$$
out_h = \lceil \frac{in_h - k_h + 1}{s_h} \rceil, \quad out_w = \lceil \frac{in_w - k_w + 1}{s_w} \rceil
$$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|--------|-----------|------|----------|----------|-------------|
| x | 输入 | 输入特征图 | float32 | ND | [batch_size, channels, in_h, in_w] |
| output | 输出 | 输出特征图 | float32 | ND | [batch_size, channels, out_h, out_w] |
| kernel_size | 属性 | 池化窗口大小 | tuple | - | (k_h, k_w) |
| stride | 属性 | 步长大小 | tuple | - | (s_h, s_w), 默认为 kernel_size |
| padding_mode | 属性 | 填充模式 | string | - | 'SAME' 或 'VALID' |

## 实现原理

该算子通过以下步骤实现:

1. 使用 `loop_unroll` 对 batch*channel 维度进行分块处理, 支持 [8, 4, 2, 1] 等多种分块大小
2. 对每个输出位置 (oh, ow):
   - 计算输入特征的滑动窗口范围, 考虑 padding 偏移
   - 对窗口内的数据进行高度方向的求和 (reduce)
   - 对宽度方向进行求和并计算平均值
   - 使用 `pypto.assemble` 将结果写入输出位置

## 核心优化

1. **动态 shape 支持**: 支持 batch_size 和 channels 的动态 shape
2. **循环展开**: 使用 `loop_unroll` 优化 batch*channel 维度的并行度
3. **分步归约**: 先对高度维度求和, 再对宽度维度求和, 避免一次性加载整个窗口
4. **边界处理**: 使用 max/min clamp 边界, 避免越界访问

## 调用示例

```python
import torch
import pypto
from avg_pool2d_impl import avg_pool2d_wrapper

# 准备数据
batch_size, channels, in_h, in_w = 2, 3, 6, 6
x = torch.rand([batch_size, channels, in_h, in_w], dtype=torch.float32)

# 调用算子
output = avg_pool2d_wrapper(
    x,
    kernel_size=(2, 2),
    stride=(2, 2),
    padding_mode='SAME'
)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

## 支持的测试用例

| 配置名称 | 类型 | kernel_size | stride | padding | 输入 Shape | 输出 Shape |
|----------|------|-------------|--------|---------|------------|------------|
| SAME_P0 | 性能 | (2, 2) | (2, 2) | SAME | [2, 3, 6, 6] | [2, 3, 3, 3] |
| VALID_P0 | 功能 | (3, 3) | (2, 2) | VALID | [4, 8, 12, 12] | [4, 8, 5, 5] |
| SAME_large | 性能 | (3, 3) | (2, 2) | SAME | [8, 64, 56, 56] | [8, 64, 28, 28] |
| VALID_stride1 | 功能 | (2, 2) | (1, 1) | VALID | [2, 16, 8, 8] | [2, 16, 7, 7] |

## 精度要求

- atol: 0.001
- rtol: 0.001

## 文件清单

| 文件 | 说明 |
|------|------|
| spec.md | 需求规范 |
| api_report.md | API 探索报告 |
| design.md | 设计文档 |
| avg_pool2d_golden.py | Golden 参考实现 |
| avg_pool2d_impl.py | 算子核心实现 |
| test_avg_pool2d.py | 测试用例 |
| README.md | 本文件 |

## 运行测试

```bash
# 设置环境变量
export TILE_FWK_DEVICE_ID=0

# 运行所有测试
cd operators/avg_pool2d
python test_avg_pool2d.py

# 运行特定测试
python test_avg_pool2d.py --test SAME_P0

# 模拟模式运行
python test_avg_pool2d.py --run_mode sim
```
