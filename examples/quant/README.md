# FP8E4M3 Per-Token 量化示例

本示例展示了如何使用 PyPTO 实现 per-token FP8E4M3 量化，这是大模型量化中常用的技术。

## 总览介绍

Per-token 量化是指对每个 token（行）独立计算缩放因子并进行量化。这种量化方式可以更好地保持每个 token 的数值精度，常用于 Transformer 模型的激活值量化。

本示例包含：
- Per-token FP8E4M3 量化算子实现
- Golden 参考实现用于精度验证
- 完整的测试和验证流程

## 量化原理

### FP8E4M3 格式
FP8E4M3 是一种 8 位浮点格式：
- 1 位符号位
- 4 位指数位
- 3 位尾数位

### FP8E8M0 格式
FP8E8M0 是一种 8 位纯指数格式：
- 1 位符号位
- 8 位指数位
- 0 位尾数位

### Per-Token 量化流程
1. 对每个 token 计算绝对值的最大值作为 scale（规约轴为 -1）
2. 将输入数据除以对应的 scale 进行归一化
3. 将归一化后的数据转换为 FP8E4M3 格式
4. 将 scale 转换为 FP8E8M0 格式

## 数据类型和形状

### 输入
- **类型**: BF16 (torch.bfloat16, pypto.DT_BF16)
- **形状**: (m, n) 二维张量

### 输出
- **量化输出**: FP8E4M3 (torch.float8_e4m3fn, pypto.DT_FP8_E4M3)
  - **形状**: (m, n) 二维张量
- **Scale 输出**: FP8E8M0 (torch.float8_e8m0fnu, pypto.DT_FP8_E8M0)
  - **形状**: (m, 1) 二维张量

## 代码结构

- **`fp8e4m3_per_token_quant.py`**: Per-token FP8E4M3 量化示例脚本
  - `create_quant_kernel()`: 创建量化算子
  - `golden_per_token_quantize()`: Golden 参考实现
  - `test_per_token_quantize()`: 测试函数

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 执行脚本

```bash
# 运行 per-token 量化示例
python3 fp8e4m3_per_token_quant.py

# 运行特定测试
python3 fp8e4m3_per_token_quant.py quant::test_per_token_quantize

# 以仿真模式运行
python3 fp8e4m3_per_token_quant.py --run_mode sim

# 列出所有可用示例
python3 fp8e4m3_per_token_quant.py --list
```

## 算子实现细节

### 量化算子
```python
@pypto.frontend.jit(runtime_options={"run_mode": mode})
def quant_kernel(
    x: pypto.Tensor([m, n], pypto.DT_BF16),
    out_quant: pypto.Tensor([m, n], pypto.DT_FP8_E4M3),
    out_scale: pypto.Tensor([m, 1], pypto.DT_FP8_E8M0),
):
    pypto.set_vec_tile_shapes(m, n, 1, 1)
    
    x_abs = pypto.abs(x)
    scale = pypto.max(x_abs, axis=1, keepdims=True)
    scale = pypto.maximum(scale, 1e-6)
    
    out_scale[:] = scale
    out_quant[:] = x / scale
```

### Golden 实现
Golden 实现在 CPU 上执行相同的计算流程，用于验证 NPU 上的结果正确性：
1. 计算每个 token 的最大绝对值作为 scale（沿 axis=1 规约）
2. 归一化输入数据
3. 转换为 FP8E4M3 格式
4. 将 scale 转换为 FP8E8M0 格式

## 测试结果

示例会输出：
- 输入张量的形状和数据类型
- 量化输出张量的形状和数据类型
- Scale 输出张量的形状和数据类型
- 反量化后的最大误差
- 反量化后的平均误差

## 注意事项

- FP8E4M3 量化会引入一定的精度损失，这是正常的
- Per-token 量化比 per-tensor 量化能更好地保持数值精度
- Per-token 量化的规约轴为 -1（最后一个维度）
- Per-channel 量化的规约轴为 -2（倒数第二个维度）
- 本示例使用相对误差 1e-2 和绝对误差 1e-2 作为验证标准
- 在实际应用中，可能需要根据具体场景调整精度容忍度

## 扩展建议

- 可以扩展支持 per-channel 量化（规约轴为 -2）
- 可以添加不同的量化策略（如对称量化、非对称量化）
- 可以实现反量化算子
- 可以添加性能测试和基准对比
