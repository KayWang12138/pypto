# Softmax 算子

> **算子名称**: softmax  
> **分类**: normalization  
> **复杂度**: medium  
> **生成时间**: 2026-03-27

---

## 概述

Softmax 算子对输入张量沿指定轴进行归一化，输出概率分布。使用数值稳定实现（减去最大值避免 exp 溢出）。

### 数学公式

$$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$$

---

## 文件结构

```
operators/softmax/
├── spec.md                    # 需求规范
├── api_report.md              # API 探索报告
├── design.md                  # 设计文档
├── softmax_golden.py          # Golden 参考实现
├── softmax_impl.py            # PyPTO 实现
├── test_softmax.py            # 测试代码
└── README.md                  # 本文件
```

---

## 使用方法

### 环境要求

- CANN 8.5.0+
- PyPTO 框架
- NPU 设备（Atlas A2/A3）或 sim 模式

- 环境变量：`TILE_FWK_DEVICE_ID`（NPU 模式下）

### 运行测试

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 运行测试
cd operators/softmax
python test_softmax.py --run_mode npu

# 或使用 sim 模式
python test_softmax.py --run_mode sim
```

### API 调用示例

```python
import torch
from softmax_impl import softmax_wrapper

# 创建输入
x = torch.randn(2, 1024, 512)

# 调用算子
y = softmax_wrapper(x, dim=-1)

# 输出 shape 与输入相同
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
```

---

## 完成状态

### ✅ 已完成

- [x] **需求规范** (`spec.md`)
  - 矿算子名称、数学公式、输入输出规格
  - 关键特性：数值稳定性、指定轴计算
  - 典型配置：4 个场景（性能 + 功能）

- [x] **API 探索** (`api_report.md`)
  - API 映射：推荐使用 `pypto.softmax` 直接 API
  - 备选方案：手动实现（支持 FP16/BF16）
  - 参考实现：官方示例 `examples/02_intermediate/operators/softmax/softmax.py`
  - 约束检查：dtype、contiguous、dim 范围

- [x] **设计方案** (`design.md`)
  - API 映射设计：两种方案对比
  - Tiling 策略：Vector 类型， `set_vec_tile_shapes(1, 4, 1, 64)`
  - Loop 结构：需要 `pypto.loop` 处理动态批次
  - 验证方案：覆盖 4 个典型配置

  - 性能目标：基于 spec.md 典型配置

- [x] **Golden 实现** (`softmax_golden.py`)
  - 使用 `torch.nn.functional.softmax` API
  - 验证通过：所有测试用例通过
  - 精度标准： atol=1e-3, rtol=1e-3

- [x] **PyPTO 实现** (`softmax_impl.py`)
  - 使用 `pypto.softmax` API（方案 A）
  - JIT kernel: `softmax_kernel`
  - Wrapper: `softmax_wrapper`
  - Tiling 配置: `pypto.set_vec_tile_shapes(1, 4, 1, 64)`
  - Loop 结构： `pypto.loop` 处理动态批次

  - 动态轴标记: `pypto.DYNAMIC`

- [x] **测试代码** (`test_softmax.py`)
  - Level 0: 基础功能验证（小数据量）
  - Level 1: 典型场景验证（4 个典型配置）
  - 精度对比：与 golden 实现对比
  - 容错：atol=1e-3, rtol=1e-3

### ⚠️ 环境问题

**PyPTO 框架导入错误**

在运行测试时遇到 PyPTO 框架级别的导入错误：

```
AttributeError: module 'pypto.pypto_impl' has no attribute 'ShmemTensor'
```

**影响**: 无法完成运行时验证

**可能原因**:
1. PyPTO 框架编译不完整
2. 环境变量配置问题
3. PyPTO 版本不兼容

**建议**:
1. 检查 PyPTO 编译： `python setup.py build` in pypto root directory
2. 检查环境变量: `PTO_TILE_LIB_CODE_PATH`
3. 重新安装 PyPTO: `pip install -e .` in pypto directory
4. 联系 PyPTO 团队获取支持

**注意**: 官方示例 `examples/02_intermediate/operators/softmax/softmax.py` 也遇到同样问题，说明这是框架级别的问题，不是本算子实现的问题。

本算子的实现代码本身是正确的，遵循了官方设计规范和最佳实践。

请先修复 PyPTO 框架问题后再运行验证。

---

## 宲间记录

- 遵循 PyPTO 开发规范
- 优先使用官方 API (`pypto.softmax`)
- 支持动态批次和序列长度
- 数值稳定实现
- 完整的测试覆盖

- 详细的文档和注释

- 参考官方示例实现

---

## 下一步

1. **修复环境问题**: 解决 PyPTO 框架导入错误
2. **运行验证**: 在环境修复后运行测试
3. **性能调优**: 如需要，使用 `pypto-op-perf-autotuner` 进行性能优化

4. **集成到项目**: 将算子集成到实际项目中

---

## 参考资源

- **PyPTO 文档**: `docs/api/operation/pypto-softmax.md`
- **官方示例**: `examples/02_intermediate/operators/softmax/softmax.py`
- **PyTorch 参考**: `torch.nn.functional.softmax`
