# batch_matmul 算子设计文档

> **算子名称**: batch_matmul
> **算子分类**: matmul
> **生成时间**: 2026-03-29
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

批量矩阵乘法，对 batch 维度的每个矩阵执行矩阵乘法。 支持可选的转置配置，用于灵活处理不同的矩阵存储格式。

### 1.2 数学公式
$$C[b,m,n] = \sum_{k} A[b,m,k] \times B[b,k,n]$$

### 1.3 算法描述
简单算子，无需分块迭代或直接调用 pypto.matmul 即可。

### 1.4 数据流图

```
      输入 x1              输入 x2
   ┌────────────┐     ┌────────────┐
   │ [b, m, k]  │     │ [b, k, n]  │
   └─────┬──────┘     └─────┬──────┘
         │                  │
         │    (可选transpose)│
         │                  │
         ▼                  ▼
    ┌───────────────────────────┐
    │      pypto.matmul         │
    │   C[b,m,n] = A[b,m,k]     │
    │            @ B[b,k,n]     │
    └─────────────┬─────────────┘
                  │
                  ▼
            ┌────────────┐
            │ 输出 y      │
            │ [b, m, n]  │
            └────────────┘

公式: y[b,i,j] = sum_k(x1[b,i,k] * x2[b,k,j])  for all b, i, j
动态轴: batch, m, k, n
```

---

## 2. API 映射设计
### 2.1 数学公式分解
将公式拆解为基本操作步骤：
| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | C = A @ B | 批量矩阵乘法，支持转置参数 |
### 2.2 PyPTO API 映射表
| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | C = A @ B | `pypto.matmul()` | input, mat2, out_dtype, a_trans, b_trans | docs/api/operation/pypto-matmul.md |
### 2.3 讱算步骤序列
```python
# 核心计算步骤
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1])

# 根据 transpose 参数选择计算方式
if transpose_x1 and transpose_x2:
    out = pypto.matmul(x1, x2, out_dtype, a_trans=True, b_trans=True)
elif transpose_x1:
    out = pypto.matmul(x1, x2, out_dtype, a_trans=True)
elif transpose_x2:
    out = pypto.matmul(x1, x2, out_dtype, b_trans=True)
else:
    out = pypto.matmul(x1, x2, out_dtype)

output[:] = out
```
### 2.4 设计依据
- **来源**: api_report.md
- **说明**: pypto.matmul 原生支持 2-4 维矩阵乘法，包含批量处理和转置参数，无需组合替代

---

## 3. 数据规格设计
### 3.1 OperatorInput dataclass
```python
@dataclass
class BatchMatmulInput:
    x1: Tensor  # 左矩阵, shape: [batch, m, k], dtype: float32
    x2: Tensor  # 右矩阵, shape: [batch, k, n], dtype: float32
    transpose_x1: bool = False  # 是否对 x1 转置
    transpose_x2: bool = False  # 是否对 x2 转置
```
### 3.2 OperatorOutput dataclass
```python
@dataclass
class BatchMatmulOutput:
    y: Tensor  # 输出矩阵, shape: [batch, m, n], dtype: float32
```
### 3.3 中间 Tensor 定义
| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| y | [batch, m, n] | float32 | 输出矩阵,由 matmul 直接产生 |
### 3.4 数据格式选择
| Tensor | 格式 | 说明 |
|--------|------|------|
| x1, x2, y | ND | 默认 ND 格式,与 PyTorch 兼容性好 |
### 3.5 动态轴定义
| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| batch | 批次大小 | [1, INT32_MAX] |
| m | 输出矩阵行数 | [1, INT32_MAX] |
| k | 内积维度 | [1, INT32_MAX] |
| n | 输出矩阵列数 | [1, INT32_MAX] |
### 3.6 JIT 装饰器配置
```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def batch_matmul(inputs: BatchMatmulInput) -> BatchMatmulOutput:
    ...
```
---

## 4. Tiling 策略
### 4.1 算子类型判断
- **类型**: Cube
- **判断依据**: 核心操作是批量矩阵乘法,pypto.matmul 属于 Cube 类型运算
### 4.2 TileShape 初值设置
```python
# Cube Tiling: [mL0, mL1], [kL0, kL1], [nL0, nL1]
# 参考示例: examples/01_beginner/tiling/tiling_config.py
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
```
### 4.3 设置依据
- **mL0, mL1**: M 轴切分,控制左矩阵行方向的分块
- **kL0, kL1**: K 轴切分,控制内积维度的分块
- **nL0, nL1**: N 轴切分,控制右矩阵列方向的分块
- **默认值选择**: [128, 128] 是常用起点,平衡 L0/L1 容量和并行度
- **来源**: docs/api/config/pypto-set_cube_tile_shapes.md
### 4.4 注意事项
- matmul 前必须调用 set_cube_tile_shapes
- 3D/4D matmul 还需调用 set_vec_tile_shapes 设置 batch 维度的切分
- TileShape 需满足 L0/L1 buffer 容量约束
### 4.5 判断依据与适用条件
- **判断依据**: pypto.matmul 是 Cube 操作,必须设置 Cube TileShape
- **适用条件**: 所有使用 pypto.matmul 的场景
- **不适用场景**: 纯 Vector 操作不需要 Cube Tiling
---

## 5. Loop 结构设计
### 场景 A: 不需要 Loop
> 适用于所有轴编译期已知、单次 Tile 可处理的算子
- **结论**: 不需要 pypto.loop
- **原因**: pypto.matmul 原生支持 2-4 维矩阵乘法,编译器自动处理 batch 维度的循环,无需手动编写 loop
- **适用条件**: 输入为 2-4 维张量,shape 编译期已知或运行时动态确定
- **限制**: 极端大的 batch 维度可能需要调整 TileShape 以优化性能
- **处理方式**: 编译器自动处理 batch 维度的数据切分和并行
---

## 6. 验证方案
### 6.1 Golden 函数设计
```python
def batch_matmul_golden(x1: torch.Tensor, x2: torch.Tensor,
                        transpose_x1: bool = False,
                        transpose_x2: bool = False) -> torch.Tensor:
    """batch_matmul 参考实现"""
    if transpose_x1:
        x1 = x1.transpose(-1, -2)  # [b, k, m]
    if transpose_x2:
        x2 = x2.transpose(-1, -2)  # [b, n, k]
    return torch.bmm(x1, x2)
```
### 6.2 测试用例设计
#### 基于 spec.md 所有典型配置
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | transpose_x1=False, transpose_x2=False | x1:[1,4096,4096], x2:[1,4096,4096] | y:[1,4096,4096] | 大规模矩阵乘法性能场景 |
| 功能_P0 | 功能 | P0 | transpose_x1=False, transpose_x2=False | x1:[4,1024,1024], x2:[4,1024,1024] | y:[4,1024,1024] | 标准功能验证 |
| 功能_P1 | 功能 | P1 | transpose_x1=False, transpose_x2=True | x1:[2,512,256], x2:[2,128,256] | y:[2,512,128] | x2转置场景 |
| 功能_P2 | 功能 | P2 | transpose_x1=True, transpose_x2=False | x1:[2,256,512], x2:[2,256,128] | y:[2,512,128] | x1转置场景 |
#### 边界情况测试
| 场景 | 参数 | 说明 |
|------|------|------|
| 最小 batch | batch=1 | 单批次边界情况 |
| 非方阵 | m != k != n | 非方阵乘法 |
| 小尺寸 | m,k,n=16 | 最小尺寸验证 |
### 6.3 精度验证标准
| Dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 |
---

## 7. 性能指标与开箱配置
### 7.1 性能目标
基于 spec.md 典型配置(性能类)的预期性能:
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 性能_P0 | 性能 | P0 | transpose=False | x1:[1,4096,4096], x2:[1,4096,4096] | y:[1,4096,4096] | 首跑成功后 2x 提升 |
### 7.2 开箱性能配置
```python
# 推荐开箱配置:平衡性能和通用性
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
# 3D matmul 需要 vec_tile_shapes
pypto.set_vec_tile_shapes(batch, m, n)  # 根据实际 shape 调整
```
### 7.3 pass_options 配置
```python
# 无特殊 pass_options 需求
pass_options = {}
```
### 7.4 runtime_options 配置
```python
runtime_options = {
    "run_mode": pypto.RunMode.NPU
}
```
---

## 8. 风险点与注意事项
### 8.1 已知约束
- pypto.matmul 要求输入 tensor 连续 (contiguous)
- 3D/4D matmul 需要额外设置 set_vec_tile_shapes
- NZ 格式输入需要调用 pypto.set_matrix_size
### 8.2 常见错误规避
| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| 未设置 cube_tile_shapes | 调用 matmul 前 | 编译失败 | 在 matmul 前调用 set_cube_tile_shapes |
| shape 不匹配 | K 轴不一致 | 运行时错误 | 确保 x1.shape[-1] == x2.shape[-2] |
| dtype 不一致 | 左右矩阵 dtype 不同 | 编译失败 | 确保 dtype 一致或使用 cast |
### 8.3 特殊场景处理
- **transpose 场景**: 使用 a_trans/b_trans 参数,无需手动 transpose
- **广播场景**: pypto.matmul 支持 batch 维度广播,如 [1,m,k] @ [b,k,n]
- **动态 shape**: 使用 pypto.from_torch 的 dynamic_axis 参数标记动态轴
### 8.4 实现建议
| 建议项 | 说明 |
|--------|------|
| 优先使用 a_trans/b_trans | 避免手动 transpose,减少内存访问 |
| 批量大小对齐 | batch 维度建议对齐到 2 的幂次,提升并行效率 |
| 监控 L0/L1 容量 | 大 shape 场景需调整 TileShape 避免溢出 |
---

## 9. 交付件清单
### 9.1 目录结构
```
custom/batch_matmul/
├── spec.md                    # 需求规范（已有）
├── api_report.md              # API 探索报告（已有）
├── design.md                  # 设计文档（本文件）
├── batch_matmul_golden.py     # Golden 参考实现
├── batch_matmul_impl.py       # 算子实现代码
├── test_batch_matmul.py       # 测试代码
└── output/                    # 运行输出（自动生成）
```
### 9.2 文件清单
| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 探索 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| batch_matmul_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| batch_matmul_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_batch_matmul.py | 代码 | 测试用例 | 后续实现 |
### 9.3 命名规范
| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `batch_matmul` |
| 目录名 | 与算子名称一致 | `custom/batch_matmul/` |
| Golden 文件 | `{op}_golden.py` | `batch_matmul_golden.py` |
| 实现文件 | `{op}_impl.py` | `batch_matmul_impl.py` |
| 测试文件 | `test_{op}.py` | `test_batch_matmul.py` |
### 9.4 生成顺序
```
spec.md → api_report.md → design.md → batch_matmul_golden.py → batch_matmul_impl.py → test_batch_matmul.py
```
