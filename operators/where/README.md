# where 算子

## 概述

where 算子根据条件张量从 x 或 y 中选择元素。当 condition 对应位置为 True 时选择 x 的元素，为 False 时选择 y 的元素。

### 数学公式

$out_i = condition_i \ ? \ x_i \ : \ y_i$

即：
$$
result_{i}=
\begin{cases}
input_{i} & \text{if } condition_{i}==True \\
other_{i} & \text{if } condition_{i}==False
\end{cases}
$$

## 目录结构

```
operators/where/
├── spec.md              # 需求规范
├── api_report.md        # API 探索报告
├── design.md            # 设计文档
├── where_golden.py      # Golden 参考实现
├── where_impl.py        # 算子实现代码
├── test_where.py        # 测试代码
└── README.md            # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 进入算子目录
cd operators/where
```

### 运行测试

```bash
# 运行所有测试（默认 NPU 模式）
python test_where.py

# 运行指定测试用例
python test_where.py where::test_where_level0
python test_where.py where::test_where_level1

# 列出所有测试用例
python test_where.py --list

# 使用模拟器模式
python test_where.py --run_mode sim
```

### 运行结果

测试通过时输出：
```
[PRECISION_PASS] All tests passed!
```

测试失败时输出：
```
[PRECISION_FAIL] <错误详情>
```

## 验证入口

| 测试用例 | 说明 | 优先级 |
|----------|------|--------|
| where::test_where_level0 | 小数据量基础功能验证 | P0 |
| where::test_where_level1 | 广播场景验证 | P0 |
| where::test_where_level2 | 标量 y 场景验证 | P1 |
| where::test_where_level3 | 不同 dtype 测试 | P1 |

## 精度标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-3 | 1e-3 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 |

## 已知限制

1. **广播限制**: PyPTO where 只支持单轴广播，多轴广播需要预处理
2. **标量精度**: fp16 场景建议使用 `pypto.Element` 传入标量，直接传 float 不保证正确性
3. **维度限制**: Shape 仅支持 2-4 维
4. **空 Tensor**: 不支持空 Tensor 输入

## 实现说明

### Tiling 策略

- 使用 `pypto.set_vec_tile_shapes(1, 128, 8, 64)` 配置 TileShape
- TileShape 维度与输出 shape 一致

### 动态轴处理

- batch 和 seq_len 标记为动态轴 (`dynamic_axis=[0, 1]`)
- 编译器自动处理动态轴的数据切分

### 标量处理

- 支持 Tensor 和标量两种输入类型
- 标量使用 `pypto.Element` 类型确保 dtype 正确性

## 参考

- PyTorch API: `torch.where(condition, input, other)`
- PyPTO API: `pypto.where(condition, input, other)`
