# NSA Compress 算子

## 算子概述

NSA Compress 是一种用于减轻 long-context 注意力计算的压缩算法，实现在 KV 序列维度进行压缩。

## 数学公式

$$\tilde{K}_t^{\text{cmp}} = f_K^{\text{cmp}}(k_{:t}) = \left\{ \varphi(k_{id+1:id+l}) \bigg| 0 \leq i \leq \left\lfloor \frac{t-l}{d} \right\rfloor \right\}$$

其中：
- $l$ = compressBlockSize（压缩滑窗大小）
- $d$ = compressStride（两次压缩滑窗间隔大小）
- $\varphi$ 是压缩函数：对滑窗内数据与权重相乘后求平均

**具体计算**：对于每个压缩窗口，`output = sum(input_window * weight) / compressBlockSize`

## 规格

| 类型 | 参数名 | shape | dtype | 说明 |
|------|--------|-------|-------|------|
| 输入 | input | [T, N, D] | bfloat16 | 待压缩张量，TND布局 |
| 输入 | weight | [compressBlockSize, N] | bfloat16 | 压缩权重 |
| 属性 | compressBlockSize | scalar | int | 压缩滑窗大小 |
| 属性 | compressStride | scalar | int | 两次压缩滑窗间隔大小 |
| 输出 | output | [T', N, D] | bfloat16 | 压缩后的结果 |

**输出 token 数量计算**：
```
output_tokens = max(0, (T - compressBlockSize) // compressStride + 1)
```

## 实现要点

1. **2D Tensor 处理**：PyPTO 对 3D tensor 的符号索引支持有限，因此实现时将输入 reshape 为 [T, N*D] 的 2D tensor
2. **滑动窗口**：每次取 compressBlockSize 个 token，与权重相乘后求平均
3. **权重广播**：使用 `expand_clone` 将权重 [N] 广播为 [N, D]

## 编译运行

```bash
# 设置环境变量
export TILE_FWK_DEVICE_ID=14
export PTO_TILE_LIB_CODE_PATH=/usr/local/Ascend/cann/aarch64-linux

# 编译 PyPTO
cd /data/x00952168/pypto_agent/pypto
python3 build_ci.py -f python3 --disable_auto_execute

# 运行测试
cd custom/nsa_compress
python3 nsa_compress.py --run_mode npu
```

## 测试结果

```
配置:
  TOTAL_TOKENS: 64
  HEAD_NUM: 4
  HEAD_DIM: 32
  COMPRESS_BLOCK_SIZE: 32
  COMPRESS_STRIDE: 16
  OUTPUT_TOKENS: 3

输入 shape: torch.Size([64, 128]) (原始: torch.Size([64, 4, 32]))
权重 shape: torch.Size([32, 4])
输出 shape: torch.Size([3, 128])

精度对比 (与 Golden 最大差异): 0.000000

✓ NSA Compress 测试通过! (atol=0.001, rtol=0.01)
```

## 已知限制

1. 当前实现为静态 shape 版本，支持固定的 HEAD_NUM 和 HEAD_DIM
2. 输入使用 2D tensor 形式 [T, N*D]，需要在外部进行 reshape
3. 暂未实现变长序列（actSeqLenOptional）支持

## 文件说明

- `nsa_compress.py`: PyPTO kernel 实现
- `nsa_compress_golden.py`: PyTorch Golden 参考实现
- `needs_analysis.md`: 需求分析文档
- `README.md`: 本文档