# NSA Compress 算子

## 算子概述

NSA Compress 是一种用于减轻 long-context 注意力计算的压缩算法，实现在 KV 序列维度进行压缩。支持动态序列长度。

## 数学公式

$$\tilde{K}_t^{\text{cmp}} = f_K^{\text{cmp}}(k_{:t}) = \left\{ \varphi(k_{id+1:id+l}) \bigg| 0 \leq i \leq \left\lfloor \frac{t-l}{d} \right\rfloor \right\}$$

其中：
- $l$ = compressBlockSize（压缩滑窗大小）
- $d$ = compressStride（两次压缩滑窗间隔大小）
- $\varphi$ 是压缩函数：对滑窗内数据与权重相乘后求平均
- $t$ 为动态序列长度（支持任意长度）

**具体计算**：对于每个压缩窗口，`output = sum(input_window * weight) / compressBlockSize`

## 规格

| 类型 | 参数名 | shape | dtype | 说明 |
|------|--------|-------|-------|------|
| 输入 | input | [T, N, D] | bfloat16 | 待压缩张量，TND布局，**T为动态维度** |
| 输入 | weight | [compressBlockSize, N] | bfloat16 | 压缩权重 |
| 属性 | compressBlockSize | scalar | int | 压缩滑窗大小（固定为32） |
| 属性 | compressStride | scalar | int | 两次压缩滑窗间隔大小（固定为16） |
| 输出 | output | [T', N, D] | bfloat16 | 压缩后的结果，**T'为动态维度** |

**输出 token 数量计算**：
```
output_tokens = max(0, (T - compressBlockSize) // compressStride + 1)
```

## 实现要点

1. **动态序列长度支持**：使用 `pypto.DYNAMIC` 标记序列维度 T，支持运行时动态序列长度
2. **动态输出计算**：output_tokens 根据输入序列长度动态计算
3. **2D Tensor 处理**：PyPTO 对 3D tensor 的符号索引支持有限，因此实现时将输入 reshape 为 [T, N*D] 的 2D tensor
4. **滑动窗口**：每次取 compressBlockSize 个 token，与权重相乘后求平均
5. **权重广播**：使用 `expand_clone` 将权重 [N] 广播为 [N, D]
6. **Assemble 写入**：使用 `pypto.assemble` 将结果写入到动态输出 tensor

## 编译运行

```bash
# 设置环境变量
export TILE_FWK_DEVICE_ID=4
export PTO_TILE_LIB_CODE_PATH=/usr/local/Ascend/cann/aarch64-linux

# 编译 PyPTO
cd /data/x00952168/pypto_agent/pypto
python3 build_ci.py -f python3 --disable_auto_execute

# 运行测试（测试多个序列长度）
cd models/experimental/ops-transformer/nsa_compress
python3 nsa_compress.py --run_mode npu

# 或者指定特定序列长度
python3 nsa_compress.py --run_mode npu --total_tokens 128
```

## 测试结果

### 动态序列长度测试（多长度验证）

```
============================================================
Testing with seq_len=64
============================================================
配置:
  TOTAL_TOKENS (动态): 64
  HEAD_NUM: 4
  HEAD_DIM: 32
  COMPRESS_BLOCK_SIZE: 32
  COMPRESS_STRIDE: 16
  OUTPUT_TOKENS (动态): 3

输入 shape: torch.Size([64, 128]) (原始: torch.Size([64, 4, 32]))
权重 shape: torch.Size([32, 4])
输出 shape: torch.Size([3, 128])

精度对比 (与 Golden 最大差异): 0.000000

✓ NSA Compress 测试通过! (atol=0.001, rtol=0.01)

============================================================
Testing with seq_len=128
============================================================
配置:
  TOTAL_TOKENS (动态): 128
  OUTPUT_TOKENS (动态): 7

精度对比 (与 Golden 最大差异): 0.000000
✓ NSA Compress 测试通过!

============================================================
Testing with seq_len=256
============================================================
配置:
  TOTAL_TOKENS (动态): 256
  OUTPUT_TOKENS (动态): 15

精度对比 (与 Golden 最大差异): 0.000000
✓ NSA Compress 测试通过!

============================================================
All dynamic sequence length tests passed!
============================================================
```

## 动态轴支持

### 已支持动态维度
- **序列长度（T）**：支持任意序列长度，运行时动态计算
- **输出 token 数（T'）**：根据输入序列长度动态计算，公式为 `(T - 32) // 16 + 1`

### 使用方式
```python
# 调用示例
input_tensor = torch.randn(seq_len, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device='npu:4')
input_2d = input_tensor.reshape(seq_len, TOTAL_DIM)
weight_tensor = torch.randn(32, HEAD_NUM, dtype=torch.bfloat16, device='npu:4')
output_tokens = (seq_len - 32) // 16 + 1
output_tensor = torch.empty(output_tokens, TOTAL_DIM, dtype=torch.bfloat16, device='npu:4')

nsa_compress_kernel(input_2d, weight_tensor, output_tensor)
```

## 已知限制

1. 当前实现支持动态序列长度，但 HEAD_NUM 和 HEAD_DIM 为固定值（4和32）
2. compressBlockSize 和 compressStride 固定为 32 和 16
3. 输入使用 2D tensor 形式 [T, N*D]，需要在外部进行 reshape

## 文件说明

- `nsa_compress.py`: PyPTO kernel 实现（支持动态序列长度）
- `README.md`: 本文档