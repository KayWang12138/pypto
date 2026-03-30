# BatchNorm 算子

BatchNorm (Batch Normalization) 算子的 PyPTO 实现。

## 概述

Batch Normalization 对输入张量在通道维度上进行归一化处理。广泛应用于 CNN 等模型中，加速训练收敛并提高模型稳定性。

### 数学公式

$$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$$

其中:
- $E[x]$ 是沿 channels 维度计算的均值
- $Var[x]$ 是沿 channels 维度计算的方差
- $\gamma, \beta$ 是可学习的缩放和偏移参数
- $\epsilon$ 是数值稳定性常数

## 目录结构

```
operators/batchnorm/
├── spec.md                    # 需求规范
├── api_report.md              # API 探索报告
├── design.md                  # 设计文档
├── batchnorm_golden.py        # Golden 参考实现
├── batchnorm_impl.py          # 算子核心实现
├── test_batchnorm.py          # 测试代码
└── README.md                  # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 运行测试

```bash
# 运行所有测试
cd operators/batchnorm
python test_batchnorm.py

# 运行单个测试
python test_batchnorm.py batchnorm::test_batchnorm_5d

# 查看所有测试用例
python test_batchnorm.py --list

# 使用模拟器模式
python test_batchnorm.py --run_mode sim
```

## 支持的数据类型

| Dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 0.001 | 0.001 | 默认，最高精度 |
| float16 | 0.01 | 0.01 | 混合精度场景 |
| bfloat16 | 0.01 | 0.01 | 混合精度场景 |

## 支持的输入维度

| 维度 | Shape | 说明 |
|------|-------|------|
| 3D | [batch, seq_len, channels] | BatchNorm1d 配置 |
| 5D | [batch, seq_len, channels, H, W] | BatchNorm2d 配置 |

## 实现说明

### 核心算法

1. 将 5D/3D 输入 reshape 为 2D [N, channels]
2. 沿 dim=0 计算 sum 和 sum_sq
3. 计算 mean = sum / N
4. 计算 var = sum_sq / N
5. 归一化: normalized = (x - mean) / sqrt(var + eps)
6. 仿射变换: output = normalized * gamma + beta
7. reshape 回原始维度

### Tiling 策略

- 使用 `pypto.set_vec_tile_shapes(64, channels)` 配置
- channels 维度完整保留以保证归约精度

### 动态轴支持

- 支持 batch 和 seq_len 动态轴
- 使用 `pypto.DYNAMIC` 标记动态维度

## 已知限制

1. **尾轴对齐**: channels * dtype_size 需要是 32 的倍数
2. **Shape 限制**: 仅支持 3D 和 5D 输入
3. **训练模式**: 当前实现仅支持训练模式（计算 batch 统计量）

## 参考信息

- PyTorch 参考实现: `torch.nn.functional.batch_norm`
- 论文: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- 类似算子: layer_norm, rms_norm, instance_norm

## 性能优化建议

1. 对于大 channels 场景，可调整 tile_rows 参数
2. 考虑使用 `pypto.rsqrt` 替代 `sqrt + div` 组合
3. 中间计算使用 FP32 保证数值稳定性
