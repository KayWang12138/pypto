# adaptive_avg_pool2d

自适应平均池化算子，将任意尺寸的输入池化到指定的输出尺寸。

## 算子概述

**功能**: 对输入张量进行自适应平均池化，池化窗口的大小和位置根据输入和输出尺寸动态计算。

**数学公式**:

$$output[b,c,h,w] = \frac{1}{(h_{end} - h_{start}) \times (w_{end} - w_{start})} \sum_{i=h_{start}}^{h_{end}-1} \sum_{j=w_{start}}^{w_{end}-1} input[b,c,i,j]$$

其中池化窗口位置动态计算：
- $h_{start} = \lfloor h \times iH / oH \rfloor$
- $h_{end} = \lceil (h+1) \times iH / oH \rceil$
- $w_{start} = \lfloor w \times iW / oW \rfloor$
- $w_{end} = \lceil (w+1) \times iW / oW \rceil$

## 目录结构

```
operators/adaptive_avg_pool2d/
├── spec.md                          # 需求规范
├── design.md                        # 设计文档
├── adaptive_avg_pool2d_golden.py    # Golden 参考实现 (PyTorch)
├── adaptive_avg_pool2d_impl.py      # PyPTO 实现
├── test_adaptive_avg_pool2d.py      # 测试代码
└── README.md                        # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU device
export TILE_FWK_DEVICE_ID=0

# 确认 CANN 环境
echo ${PATH} | grep cann
```

### 运行测试

```bash
# 运行所有测试
python test_adaptive_avg_pool2d.py

# 运行指定测试
python test_adaptive_avg_pool2d.py adaptive_avg_pool2d::test_level0

# 查看所有可用测试
python test_adaptive_avg_pool2d.py --list
```

### 测试用例

| 配置名称 | 输入 Shape | 输出 Shape | 说明 |
|----------|------------|------------|------|
| Level 0 | [2, 4, 8, 8] | [2, 4, 4, 4] | 基础功能验证 |
| 性能_P0 | [16, 256, 14, 14] | [16, 256, 7, 7] | CNN 特征图下采样 |
| 功能_P0 | [8, 512, 7, 7] | [8, 512, 1, 1] | 全局平均池化 |
| 单值尺寸_P1 | [4, 128, 16, 16] | [4, 128, 8, 8] | 单值 output_size |
| 非对齐_P1 | [2, 64, 7, 7] | [2, 64, 5, 5] | 非对齐窗口 |

## 精度标准

| Dtype | rtol | atol |
|-------|------|------|
| float32 | 1e-3 | 1e-3 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 |

## 已知限制

1. **输出尺寸必须是编译时常量**: `output_size` 参数需要在编译时确定
2. **4D 输入**: 当前实现仅支持 4D 输入 [N, C, H, W]
3. **动态轴**: N (batch) 和 C (channels) 支持动态，H 和 W 需要在编译时确定

## 参考

- PyTorch: `torch.nn.functional.adaptive_avg_pool2d`
- 设计文档: `design.md`
- 需求规范: `spec.md`
