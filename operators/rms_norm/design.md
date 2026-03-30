# {operator_name} 设计文档

> **算子名称**: rms_norm
> **算子分类**: normalization
> **生成时间**: 2026-03-28T00:00:00Z
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述
均方根归一化（Root Mean Square Normalization)。对输入张量在最后一个维度上计算均方根(Rms)，然后用 RMS 进行归一化，最后乘以可学习的缩放参数。与 Layer Norm 相比, RMS Norm 不进行均值中心化,计算更简单,在 Transformer 架构（如 LLaMA, GPT-NeoX)中广泛使用。
### 1.2 数学公式
y = x * γ / sqrt(mean(x²) + ε)
### 1.3 数据流图
```
    输入 x                    weight γ               eps
┌──────────────┐         ┌──────────────┐         ┌───┐
│  [b, s, d]   │         │    [d]       │         │1e-6│
└──────┬───────┘         └──────┬───────┘         └─┬─┘
       │                        │                   │
       │    ┌───────────────────┼───────────────────┘
       ▼    ▼                   ▼
    ┌─────────────┐
    │   x^2       │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ mean(x^2)   │  ← 在最后一个维度上求均值
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ + eps       │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   sqrt()    │  ← RMS
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ x / RMS     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   * γ       │
    └──────┬──────┘
           │
           ▼
    ┌──────────────┐
    │  输出 y       │
    │  [b, s, d]   │
    └──────────────┘

公式: y = x * γ / sqrt(mean(x^2) + ε)
动态轴: b (batch), s (seq_len)
```

---
## 2. API 映射设计
### 2.1 数学公式分解
将公式拆解为基本操作步骤:
| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | x² | 对输入求平方 |
| 2 | mean(x²) | 在最后一个维度上求均值 |
| 3 | mean_sq + eps | 添加防止除零的小常数 |
| 4 | sqrt(mean_sq + eps) | 计算均方根 |
| 5 | x / rms | 归一化（广播 rms） |
| 6 | normalized * γ | 应用缩放参数（广播 γ) |
### 2.2 PyPTO API 映射表
| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | x² | `pypto.mul(x, x)` | lhs=x, rhs=x | `docs/api/operation/pypto-mul.md` |
| 2 | mean(x²) | `pypto.sum(x, dim=-1, keepdim=True)` | x, dim=-1, keepdim=True | `docs/api/operation/pypto-sum.md` |
| 3 | mean_sq + eps | `pypto.add(mean_sq, eps)` | mean_sq, eps | `docs/api/operation/pypto-add.md` |
| 4 | sqrt(...) | `pypto.sqrt(...)` | x | `docs/api/operation/pypto-sqrt.md` |
| 5 | x / rms | `pypto.div(x, rms)` | x, rms | `docs/api/operation/pypto-div.md` |
| 6 | normalized * γ | `pypto.mul(normalized, gamma)` | normalized, gamma | `docs/api/operation/pypto-mul.md` |
### 2.3 计算步骤序列
```python
# 伪代码展示计算流程
squared = x * x
mean_sq = pypto.sum(squared, dim=-1, keepdim=True) / hidden_size
rms = pypto.sqrt(mean_sq + eps)
normalized = x / rms
output = normalized * gamma
```
### 2.4 设计依据
- 来源: api_report.md + docs/api/operation/pypto-rms_norm.md
- 说明:
  - **方案 A (推荐)**: PyPTO 已提供 `pypto.rms_norm` 内置 API，直接调用即可完成全部计算,代码简洁、性能优化。
  - **方案 B**: 使用基础 API 组合实现, 提供更灵活的控制能力,适合需要自定义计算流程的场景。
  - **选择**: 由于 PyPTO 已提供 `pypto.rms_norm` 内置 API,本设计采用方案 A。
---
## 3. 数据规格设计
### 3.1 OperatorInput dataclass
```python
from dataclasses import dataclass
import pypto
from pypto import DataType


@dataclass
class RmsNormInput:
    x: pypto.Tensor  # 输入张量, shape: [b, s, d], dtype: FP32/BF16
    weight: pypto.Tensor  # 缩放参数, shape: [d], dtype: FP32/BF16
    eps: float = 1e-6  # 防止除零的小常数
```
### 3.2 OperatorOutput dataclass
```python
from dataclasses import dataclass
import pypto
from pypto import DataType


@dataclass
class RmsNormOutput:
    y: pypto.Tensor  # 输出张量, shape: [b, s, d], dtype: FP32/BF16
```
### 3.3 中间 Tensor 定义
| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| squared | [b, s, d] | FP32/BF16 | x 的平方 |
| mean_sq | [b, s, 1] | FP32/BF16 | x² 的均值 |
| rms | [b, s, 1] | FP32/BF16 | 均方根 |
| normalized | [b, s, d] | FP32/BF16 | 归一化后的 x |
### 3.4 数据格式选择
| Tensor | 格式 | 说明 |
|--------|------|------|
| x | ND | 输入数据,默认 ND 格式 |
| weight | ND | 缩放参数,默认 ND 格式 |
| y | ND | 输出数据,默认 ND 格式 |
### 3.5 动态轴定义
| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| b | batch 维度 | [1, INT32_MAX] |
| s | seq_len 维度 | [1, INT32_MAX] |
动态轴通过 `pypto.from_torch(..., dynamic_axis=[0, 1])` 标记。
### 3.6 JIT 装饰器配置
```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def rms_norm(inputs: RmsNormInput) -> RmsNormOutput:
    ...
```
---
## 4. Tiling 策略
### 4.1 算子类型判断
- **类型**: Vector
- **判断依据**: 计算仅含逐元素操作(mul, div, sqrt)和归约操作(sum),无矩阵乘法(matmul),因此属于 Vector 类型算子。
### 4.2 TileShape 初值设置
```python
pypto.set_vec_tile_shapes(64, 128)
```
### 4.3 设置依据
- **参考**: `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` 中 rms_norm 使用 `pypto.set_vec_tile_shapes(64, 128)`
- **分析**: 64 对齐尾轴 32B 对齐要求, 128 是合理的 hidden_size
- **适用**: 适用于 hidden_size <= 8192 的场景。对于更大的 hidden_size, 需要调整 tiling 参数.
### 4.4 注意事项
- 尾轴 128 需满足 32B 对齐要求
- 对于不同的 hidden_size, 需要调整 tiling 参数
### 4.5 判断依据与适用条件
- 判断依据: 参考 `layer_norm.py` 示例中的 tiling 配置,并考虑 32B 对齐要求
- 适用条件: 适用于 hidden_size <= 8192 的场景,支持动态 batch 和 seq_len
- 不适用场景: hidden_size > 8192 时需要人工调整 tiling 参数
---
## 5. Loop 结构设计
### 场景 A: 不需要 Loop
> 适用于所有轴编译期已知、单次 Tile 可处理的算子(如逐元素运算)。
- **结论**: 不需要 pypto.loop
- **原因**: 所有操作均可单次 Tile 完成,编译期可确定所有维度大小
- **适用条件**: 静态 shape 或动态 shape 但数据量不超过单个 Tile 宷量
- **限制**: 不适用于数据量超过单个 Tile 容量的场景
- **处理方式**: 编译器自动处理数据切分,无需手动循环
### 场景 B: 需要 Loop
> 适用于存在动态轴且数据量可能超过单个 Tile 容量的场景。
#### 5.1 Loop 判断结论
- **结论**: 需要 Loop
- **原因**: 存在动态轴 b 和 s, 当 b * s * hidden_size 超过单个 Tile 容量时,需要循环处理
- **Loop 类型**: pypto.loop
- **适用条件**: 动态 batch 和 seq_len,数据量可能超过单个 Tile 容量
- **限制**: 需要正确处理尾块
#### 5.2 黍态轴 vs 动态轴处理
| 轴 | 类型 | 处理方式 |
|----|------|----------|
| b | 动态 | pypto.loop |
| s | 动态 | pypto.loop |
| d | 静态 | Python for |
#### 5.3 Loop 合并策略
- **batch 和 seq_len 双重循环**: 外层循环 batch, 内层循环 seq_len, 或合并为单层循环
- **选择**: 根据 hidden_size 和 tiling 配置决定是否合并循环
#### 5.4 数据依赖处理
- **依赖**: normalized 依赖 rms, rms 依赖 mean_sq, mean_sq 依赖 squared, squared 依赖 x
- **处理**: 在循环内按依赖顺序计算,确保数据一致性
#### 5.5 尾块处理策略
- **问题**: 当 seq_len 不是 tiling 廴数的倍数时,最后一个 block 需要特殊处理
- **方案**: 使用条件判断或处理尾块,或使用 padding 策略
#### 5.6 loop_unroll 配置
```python
# 动态轴范围跨度大时使用,可在编译期生成多版本代码
# 本设计采用 pypto.loop,不使用 loop_unroll
```
---
## 6. 验证方案
### 6.1 Golden 函数设计
```python
def rms_norm_golden(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS Norm 参考实现"""
    x_squared = x * x
    mean_sq = x_squared.mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    normalized = x / rms
    return normalized * weight
```
### 6.2 测试用例设计
#### 基于 spec.md 所有典型配置
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| LLaMA-7B | 性能 | P0 | eps=1e-6, d=4096 | x:[b,s,4096], weight:[4096] | y:[b,s,4096] | LLaMA-7B 模型配置 |
| LLaMA-13B | 性能 | P0 | eps=1e-6, d=5120 | x:[b,s,5120], weight:[5120] | y:[b,s,5120] | LLaMA-13B 模型配置 |
| 功能验证 | 功能 | P0 | eps=1e-6, d=64 | x:[2,128,64], weight:[64] | y:[2,128,64] | 小规模功能验证 |
| 动态轴测试 | 功能 | P0 | eps=1e-6, d=256 | x:[b,s,256], weight:[256] | y:[b,s,256] | 动态 batch 和 seq_len |
#### 边界情况测试
| 场景 | 参数 | 说明 |
|------|------|------|
| 最小 batch | b=1 | 单样本测试 |
| 最小 seq_len | s=1 | 单 token 测试 |
| 大 hidden_size | d=8192 | 大隐藏层测试 |
### 6.3 精度验证标准
| Dtype | atol | rtol |
|-------|------|------|
| FP32 | 0.001 | 0.001 |
| BF16 | 0.01 | 0.01 |
---
## 7. 性能指标与开箱配置
### 7.1 性能目标
基于 spec.md 典型配置(性能类)的预期性能:
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| LLaMA-7B | 性能 | P0 | eps=1e-6, d=4096 | x:[b,s,4096], weight:[4096] | y:[b,s,4096] | 待测试后确定 |
| LLaMA-13B | 性能 | P0 | eps=1e-6, d=5120 | x:[b,s,5120], weight:[5120] | y:[b,s,5120] | 待测试后确定 |
### 7.2 开箱性能配置
```python
# Tiling 配置
pypto.set_vec_tile_shapes(64, 128)
```
### 7.3 pass_options 配置
无特殊 pass_options 配置需求。
### 7.4 runtime_options 配置
```python
runtime_options={"run_mode": pypto.RunMode.NPU}
```
---
## 8. 风险点与注意事项
### 8.1 已知约束
- 输入 Tensor 必须是连续的(is_contiguous() == True)
- x 和 weight 的 dtype 必须一致
- eps 值必须为正数
### 8.2 常见错误规避
| 风险/错误 | 触发场景 | 影响 | 原因 | 规避方法 |
|-------------|----------|-------------|----------|-------------|
| 非连续输入 | 输入 Tensor 不连续 | 编译失败 | from_torch 要求连续 | 确保输入 Tensor 连续 |
| dtype 不匹配 | x 和 weight dtype 不同 | 运行时错误 | 计算要求数据类型一致 | 确保 x 和 weight dtype 一致 |
| eps 值过小 | eps <= 0 | 数值不稳定 | 除零保护失效 | 确保 eps > 0 |
### 8.3 特殊场景处理
- **动态轴**: 通过 `dynamic_axis=[0, 1]` 标记 batch 和 seq_len 为动态轴,支持运行时变化
- **大 hidden_size**: 当 hidden_size > 8192 时,需要调整 tiling 参数
### 8.4 实现建议
| 建议项 | 说明 |
|--------|------|
| 优先使用内置 API | `pypto.rms_norm` 已经过优化,推荐直接使用 |
| 动态轴支持 | 使用 `dynamic_axis` 参数支持动态 batch 和 seq_len |
| 精度验证 | 使用 torch.allclose 验证结果,注意 bfloat16 的精度要求较宽松 |
---
## 9. 交付件清单
### 9.1 目录结构
```
custom/rms_norm/
├── spec.md                      # 需求规范(已有)
├── design.md                    # 设计文档(本文件)
├── rms_norm_golden.py          # Golden 参考实现
├── rms_norm_impl.py            # 算子实现代码
├── test_rms_norm.py            # 测试代码
└── output/                      # 运行输出(自动生成)
```
### 9.2 文件清单
| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| design.md | 设计 | 算子设计文档 | pypto-op-design(本 skill) |
| rms_norm_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| rms_norm_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_rms_norm.py | 代码 | 测试用例 | 后续实现 |
### 9.3 命名规范
| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `rms_norm` |
| 目录名 | 与算子名称一致 | `custom/rms_norm/` |
| Golden 文件 | `{op}_golden.py` | `rms_norm_golden.py` |
| 实现文件 | `{op}_impl.py` | `rms_norm_impl.py` |
| 测试文件 | `test_{op}.py` | `test_rms_norm.py` |
### 9.4 生成顺序
```
spec.md → design.md → rms_norm_golden.py → rms_norm_impl.py → test_rms_norm.py
```
