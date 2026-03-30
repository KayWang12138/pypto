# max_pool2d 算子

## 概述

`max_pool2d` 是对输入 tensor 在空间维度（H 和 W）上进行最大池化操作。对于输入 tensor 的每个滑动窗口，输出窗口内的最大值。常用于 CNN 网络中下采样特征图。

## 数学公式

$$\text{output}[n, c, h, w] = \max_{i \in [0, kH), j \in [0, kW)} \text{input}[n, c, s_h \cdot h + i \cdot d_h - p_h, s_w \cdot w + j \cdot d_w - p_w]$$

其中：
- $kH, kW$ 为 kernel_size 的高度和宽度
- $s_h, s_w$ 为 stride 的高度和宽度
- $p_h, p_w$ 为 padding 的高度和宽度偏移
- $d_h, d_w$ 为 dilation 的高度和宽度

## 支持特性

| 特性 | 支持 |
|------|------|
| kernel_size | int 或 (int, int) |
| stride | int 或 (int, int)，默认等于 kernel_size |
| padding | int 或 (int, int)，默认 0 |
| dilation | int 或 (int, int)，默认 1 |
| ceil_mode | bool，默认 False |
| 数据类型 | float16, float32 |
| 输入维度 | 3D (C, H, W) 或 4D (N, C, H, W) |
| 动态轴 | batch, H, W |

## 目录结构

```
operators/max_pool2d/
├── spec.md                 # 需求规范
├── api_report.md           # API 探索报告
├── design.md               # 设计文档
├── max_pool2d_golden.py    # Golden 参考实现
├── max_pool2d_impl.py      # PyPTO 算子实现
├── test_max_pool2d.py      # 测试代码
├── README.md               # 本文件
└── .orchestrator_state.json # 状态文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 设置 pto-isa 路径
export PTO_TILE_LIB_CODE_PATH=./pto_isa/pto-isa/
```

### 运行测试

```bash
# 运行所有测试
python operators/max_pool2d/test_max_pool2d.py

# 运行单个测试
python operators/max_pool2d/test_max_pool2d.py max_pool2d::test_resnet_pool

# 列出所有测试用例
python operators/max_pool2d/test_max_pool2d.py --list

# 使用模拟器模式
python operators/max_pool2d/test_max_pool2d.py --run_mode sim
```

## 测试用例

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape |
|----------|------|--------|------|------------|------------|
| resnet_pool | 性能 | P0 | kernel=3, stride=2, padding=1 | [32, 64, 112, 112] | [32, 64, 56, 56] |
| vgg_pool | 性能 | P0 | kernel=2, stride=2, padding=0 | [32, 128, 56, 56] | [32, 128, 28, 28] |
| dynamic_batch | 功能 | P0 | kernel=3, stride=2, padding=1 | [?, 64, 224, 224] | [?, 64, 112, 112] |
| dynamic_hw | 功能 | P0 | kernel=2, stride=2, padding=0 | [16, 32, ?, ?] | [16, 32, H/2, W/2] |
| float16_test | 功能 | P1 | kernel=3, stride=2, padding=1 | [16, 64, 56, 56] (fp16) | [16, 64, 28, 28] (fp16) |
| dilation_test | 功能 | P1 | kernel=5, stride=1, padding=2, dilation=2 | [8, 32, 32, 32] | [8, 32, 28, 28] |
| ceil_mode_test | 功能 | P1 | kernel=2, stride=2, padding=0, ceil_mode=True | [8, 32, 7, 7] | [8, 32, 4, 4] |

## 精度标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.005 | 0.005 |

## 已知限制

1. **padding 限制**: `pypto.pad` 不支持左/上填充，kernel 内通过边界检查处理
2. **dilation 支持**: 通过分步切片实现空洞池化
3. **amax 归约**: PyPTO 的 `amax` 仅支持单轴归约，需要对 H 和 W 维度分别归约

## 应用场景

- ResNet, VGG, MobileNet, EfficientNet 等 CNN 网络
- 下采样层，通常在卷积层之后
