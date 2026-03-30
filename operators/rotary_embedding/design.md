# rotary_embedding 算子设计文档

# PyPTO 算子开发速查

> 详细信息通过搜索 docs/ 获取，本文件仅提供核心原则、约束

# 1. 概述
- **算子名称**: rotary_embedding
- **算子分类**: embedding
- **生成时间**: 2026-03-28
- **基于**: spec.md, api_report.md

# 2. 功能描述
实现 Rotary Position Embedding (RoPE)， 通过旋转矩阵编码位置信息。核心计算公式为:
y = x * cos + rotate_half(x) * sin
 其中 rotate_half 操作将输入张量分成前后两半并交换位置并取负。
然后拼接得到旋转后的张量。
最终通过乘法和加法实现旋转。
计算。
# 3. 数学公式
- 旋转角度: $\theta_i = \text{position} \times \text{freq}_i$, where $\text{freq}_i = 1 / (10000^{2i/d})$
$
- 旋转变换: $\text{RoPE}(x, \text{position}) = x \cdot \cos(\theta) + \text{rotate\_half}(x) \cdot \sin(\theta)$
$
- rotate_half: $[x_1, x_2, ..., x_d] \to [-x_{d/2+1}, ..., -x_d, x_1, ..., x_{d/2}]$

$
# 4. 算法描述
```
Algorithm: Rotary Position Embedding (Forward)
────────────────────────────────────────
输入: x [batch, seq_len, num_heads, head_dim]
      cos [seq_len, head_dim]
      sin [seq_len, head_dim]
输出: y [batch, seq_len, num_heads, head_dim]

1. 分割 x 为前后两半:
   x1 = x[..., :head_dim//2]
   x2 = x[..., head_dim//2:]

2. 对后半部分取负: neg_x2 = -x2
3. 拼接负_x2 和 x1 得旋转后的张量:
   rotated_x = pypto.concat([neg_x2, x1], dim=-1)
4. 讇旋转后的结果:
   y = x * cos + rotated_x * sin
5. 返回 y
```
# 5. 数据流图
```
    输入 x [b, s, n, d]          cos [s, d]          sin [s, d]
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│ [batch, seq,    │          │ [seq, head_dim] │          │ [seq, head_dim] │
│  heads, dim]    │          │                 │          │                 │
│    float32      │          │    float32      │          │    float32      │
└────────┬────────┘          └────────┬────────┘          └────────┬────────┘
         │                            │                            │
         ▼                            ▼                            ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │                        Rotary Embedding                            │
    │  1. 分割 x 为 x1 (前半) 和 x2 (后半)                               │
    │  2. rotated_x = concat([-x2, x1])                                 │
    │  3. y = x * cos + rotated_x * sin                                 │
    └────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ 输出 y           │
                              │ [b, s, n, d]    │
                              │    float32      │
                              └─────────────────┘

动态轴: batch, seq_len
```
# 6. 数据规格设计
## 6.1 输入数据规格

### 6.1.1 输入数据类
```python
@dataclass
@dataclass
class OperatorInputData:
    x: Tensor  # [batch, seq_len, num_heads, head_dim]
    cos: Tensor  # [seq_len, head_dim] 或 [1, seq_len, 1, head_dim]
    sin: Tensor  # [seq_len, head_dim] 或 [1, seq_len, 1, head_dim]
```

### 6.1.2 输出数据类
```python
@dataclass
@dataclass
class OperatorOutputData:
    y: Tensor  # [batch, seq_len, num_heads, head_dim]
```
## 6.2 数据类型支持

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.005 | 0.005 |
| bfloat16 | 0.01 | 0.01 |
## 6.3 数据格式选择
| Tensor | 格式 | 说明 |
|--------|------|------|
| x | ND | 输入张量， ND 格式最常用 |
| cos | ND | 余弦张量， ND 格式便于广播 |
| sin | ND | 正弦张量, ND 格式便于广播 |
## 6.4 动态轴定义
<!-- 根据需求，动态轴包括 batch, seq_len, num_heads -->
| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| batch | 批次大小 | [1, 65536]
| seq_len | 序列长度 | [1, 32768]
| num_heads | 注意力头数 | [1, 128]
# 7. JIT 装饰器配置
```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.run_mode.NPU}
)
def rotary_embedding_wrapper(inputs: OperatorInputData) -> OperatorOutputData:
    ...
```
# 8. Tiling 策略
## 8.1 算子类型判断
- **类型**: Vector
- **判断依据**: 核心计算为逐元素乘法和加法运算,不涉及矩阵乘法
  公式: y = x * cos + rotate_half(x) * sin
  - 无 matmul 操作 -> Vector 类型
  - 仅需设置 `set_vec_tile_shapes()`
## 8.2 TileShape 初值设置
```python
pypto.set_vec_tile_shapes(1, 128, 8, 64)
```
### 8.3 设置依据
- **原则 1**: 尽量填满 UB
- **原则 2**: 尾轴对齐 64 的倍数,利于向量化效率
- **原则 3**: 考虑动态轴的影响,编译期 TileShape 固定，运行时动态调整
### 8.4 注意事项
- 尾轴 head_dim 黺议为 64 的倍数,以提高向量化效率
- batch 和 seq_len 作为动态轴,编译期 TileShape 錆入固定值
- num_heads 蚂作为静态轴,编译期 TileShape 需填入该维度
- 动态轴场景下,编译器会自动处理数据分块
## 8.5 判断依据适用条件
- **判断依据**: 算子为逐元素计算密集型,适合向量化
- **适用条件**: 所有输入输出 shape 已知的场景
- **不适用场景**: head_dim 不是 64 的倍数时可能需要调整
# 9. Loop 结构设计
## 9.1 Loop 判断结论
- **结论**: 不需要 Loop
- **原因**: 所有轴编译期已知,单次 Tile 可以覆盖全部数据
- **适用条件**: 无动态轴场景
- **限制**: head_dim 必须为 64 的倍数
- **处理方式**: 编译器自动处理数据切分,无需手动循环
## 9.2 騡式选择
根据 spec.md 中的典型配置,所有计算可以在单次 Tile 中完成,不需要 pypto.loop。
## 9.3 騡式说明
- **模式**: 无 Loop 模式
- **原因**:
  1. 所有轴在编译期已知大小或单次 Tile 可以处理完整输入
  2. 计算为纯逐元素操作,无数据依赖
  3. 动态轴通过编译器自动处理
# 10. 麞态轴 vs 动态轴处理
所有轴都是静态轴(编译期已知),不需要特殊处理。
# 11. 数据依赖处理
- **x1, x2 依赖**: x, cos, sin
- **rotate_x 依赖**: neg_x2, x1
- **输出 y 依赖**: x * cos, rotated_x * sin
# 12. 尾块处理策略
无尾块,所有数据在单次 Tile 中处理完成。
# 13. loop_unroll 配置
不适用(无动态轴场景)
# 14. 验证方案
## 14.1 Golden 函数设计
```python
def rotary_embedding_golden(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """rotary_embedding golden reference implementation.
    Args:
        x: Input tensor.
        cos: Cosine tensor.
        sin: Sine tensor.
    Returns:
        Rotated tensor.
    """
    # Split x into two halves
    head_dim = x.shape[-1]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    # Rotate half: [-x2, x1]
    neg_x2 = -x2
    rotated_x = torch.cat([neg_x2, x1], dim=-1)
    # Apply rotary embedding
    y = x * cos + rotated_x * sin
    return y
```
## 14.2 测试用例设计
### 基于 spec.md 典型配置
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| LLaMA-7B | 性能 | P0 | head_dim=128, num_heads=32 | x:[1,4096,32,128], cos:[4096,128], sin:[4096,128] | y:[1,4096,32,128] | LLaMA-7B 典型配置 |
| 小规模验证 | 功能 | P0 | head_dim=64, num_heads=8 | x:[2,128,8,64], cos:[128,64], sin:[128,64] | y:[2,128,8,64] | 功能验证小规模 |
| 动态shape | 功能 | P1 | head_dim=64, num_heads=12 | x:[b,s,12,64], cos:[s,64], sin:[s,64] | y:[b,s,12,64] | 动态batch和seq |
### 边界情况测试
| 场景 | 参数 | 说明 |
|------|------|------|
| 零值 | x=0 | 飞边界值,输出为输入 |
| 极值 | x=1e10 或 麢近最大值, 输出应正常 |
| NaN | x=NaN | 输出为 NaN |
### 精度验证标准
| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.005 | 0.005 |
| bfloat16 | 0.01 | 0.01 |
# 15. 性能指标与开箱配置
## 15.1 性能目标
基于 spec.md 典型配置(性能类)的预期性能
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 齟 kernel 耗时(ms) |
|----------|------|--------|------|------------|------------|---------------|
| LLaMA-7B | 性能 | P0 | head_dim=128, num_heads=32 | x:[1,4096,32,128], cos:[4096,128], sin:[4096,128] | y:[1,4096,32,128] | 待测量 |
## 15.2 开箱性能配置
```python
# Tiling: 1, 128, 8, 64 (针对 [batch, seq_len, num_heads, head_dim])
pypto.set_vec_tile_shapes(1, 128, 8, 64)
```
## 15.3 pass_options 配置
无特殊配置
## 15.4 runtime_options 配置
```python
runtime_options = {"run_mode": pypto.run_mode.NPU}
```
# 16. 风险点与注意事项
## 16.1 已知约束
- mul/add API 仅支持 2-4 维
- concat dim=-1 要求最后一个维度必须可以拼接
- view 操作需要明确 valid_shape
## 16.2 常见错误规避
| 风险 / 错误 | 触发场景 | 影响 | 原因 | 规避方法 |
|------------|----------|-------------|----------|
| Shape 不匹配 | cos/sin 与 x shape 不匹配 | 计算错误 | 确保 cos/sin 的 seq_len 维度与 x 一致 |
| 广播失败 | cos/sin 无法广播到 x | 计算错误 | 确保 cos/sin 的 head_dim 与 x 一致 |
## 16.3 特殊场景处理
- 对于动态轴场景,需要使用 SymbolicScalar 夌符号性处理
- 对于非 64 塍数的 head_dim, 需要调整 TileShape
## 16.4 实现建议
| 建议项 | 说明 |
|--------|------|
| 优先使用 float32 | 保证精度,减少类型转换 |
| cos/sin 预计算 | 可在外部预计算 cos/sin 并传入,减少 kernel 内计算 |
# 17. 交付件清单
## 17.1 目录结构
```
custom/rotary_embedding/
├── spec.md                          # 黸求规范(已有)
├── design.md                        # 设计文档(本文件)
├── rotary_embedding_golden.py        # Golden 参考实现
├── rotary_embedding_impl.py          # 算子实现代码
├── test_rotary_embedding.py          # 测试代码
└── output/                          # 运行输出(自动生成)
```
## 17.2 文件清单
| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | API | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design(本 skill) |
| rotary_embedding_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| rotary_embedding_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_rotary_embedding.py | 代码 | 测试用例 | 后续实现 |
## 17.3 命名规范
| 项目 | 规范 | 示例 |
|------|------|------|
| 简子名称 | 小写字母 + 下划线 | `rotary_embedding` |
| 目录名 | 与算子名称一致 | `custom/rotary_embedding/` |
| Golden 文件 | `{op}_golden.py` | `rotary_embedding_golden.py` |
| 实现文件 | `{op}_impl.py` | `rotary_embedding_impl.py` |
| 测试文件 | `test_{op}.py` | `test_rotary_embedding.py` |
## 17.4 生成顺序
```
spec.md -> api_report.md -> design.md -> rotary_embedding_golden.py -> rotary_embedding_impl.py -> test_rotary_embedding.py
```
