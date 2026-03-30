# exp 算子

逐元素计算指数函数: y = exp(x) = e^x

## 公式

```
y = exp(x) = e^x
```

## 功能说明

对输入张量中的每个元素计算 e 的该元素次方。

## 目录结构

```
operators/exp/
├── spec.md              # 需求规范
├── api_report.md        # API 探索报告
├── design.md            # 设计文档
├── exp_golden.py        # Golden 参考实现
├── exp_impl.py          # 算子核心实现
├── test_exp.py          # 测试用例
└── README.md            # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 确保已安装 pypto
python3 -c "import pypto; print(pypto.__version__)"
```

### 运行测试

```bash
# 运行所有测试
python3 operators/exp/test_exp.py

# 运行单个测试
python3 operators/exp/test_exp.py exp::test_exp_level0

# 列出所有测试用例
python3 operators/exp/test_exp.py --list

# 使用模拟器模式
python3 operators/exp/test_exp.py --run_mode sim
```

## 验证入口

| 测试用例 | 说明 |
|----------|------|
| test_exp_level0 | 小数据量基础功能验证 (16 元素) |
| test_exp_level1 | 典型场景验证 (1K 元素) |
| test_exp_level2 | 2D 大规模验证 (1M 元素) |
| test_exp_level3_fp16 | FP16 dtype 验证 |
| test_exp_level4_performance | 性能场景验证 (4D 大规模) |

## 精度标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-3 |
| float16 | 1e-3 | 1e-2 |

## 已知限制

1. **输入维度**: 仅支持 2D 或 4D 输入
2. **dtype**: 支持 FP32 和 FP16
3. **连续性**: 输入必须是 contiguous tensor
4. **空 tensor**: 不支持空 tensor
5. **Shape Size**: 不超过 INT32_MAX

## 实现策略

- 使用 `pypto.exp` 直接 API
- 4D 输入使用 reshape(-1, last_dim) 策略，避免 4D tiling 编译问题
- Tiling 配置: `pypto.set_vec_tile_shapes(64, 128)`

## 参考

- PyTorch API: `torch.exp(x)`
- PyPTO API: `pypto.exp(x)`
