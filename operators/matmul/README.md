# matmul 算子

矩阵乘法算子，支持 batch 维度广播的批量矩阵乘法。

## 功能

- 2D 矩阵乘法: [M, K] @ [K, N] -> [M, N]
- 3D/4D batch matmul: [B, M, K] @ [B, K, N] -> [B, M, N]
- Batch 维度广播: [B1, 1, M, K] @ [1, B2, K, N] -> [B1, B2, M, N]
- 动态轴支持: batch, M, N, K 可动态变化

- 多种 dtype: float32, float16

## 实现

- **API**: `pypto.matmul`
- **类型**: Cube
- **Tiling**: 自适应配置（根据矩阵大小选择）
- **Loop**: 3D/4D 场景需要 pypto.loop 夻理 batch 维度

## 文件
- `matmul_golden.py`: PyTorch 参考实现
- `matmul_impl.py`: PyPTO 实现
- `test_matmul.py`: 测试文件

## 测试
运行测试:
```bash
cd /workspace/code/pypto/operators/matmul
python test_matmul.py
```
## 性能
- **目标**: 首跑精度成功性能的 2 倍
- **推荐配置**:
  - 小矩阵 (M, N, K < 64): [32, 32], [64, 64], [64, 64]
  - 中等矩阵 (64-512): [128, 128], [128, 128], [128, 128]
  - 大矩阵 (>512): [256, 256], [256, 256], [256, 256]

## 约束
- 输入必须 contiguous
- K 维度必须匹配
- 3D/4D 场景需要额外的 vector tiling
## 参考
- PyPTO API 文档: `docs/api/operation/pypto-matmul.md`
- 示例代码: `examples/01_beginner/compute/matmul_ops.py`
