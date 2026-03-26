# MLA Prolog PyPTO 算子

Multi-Head Latent Attention (MLA) 前处理算子的 PyPTO 实现。

## 功能说明

MLA Prolog 是 MLA (Multi-Head Latent Attention) 的前处理算子，用于推理场景。主要计算 Query 和 Key 的预处理。

### 计算公式

#### RmsNorm 公式
```
RmsNorm(x) = gamma * (x_i / RMS(x))
RMS(x) = sqrt((1/N) * sum(x_i^2) + epsilon)
```

#### Query 计算路径
```
c^Q = RmsNorm(token_x @ W^DQ)           # 下采样 + RmsNorm
q^N = c^Q @ W^UQ @ W^UK                # 两次上采样 -> query 输出
q^R = ROPE(c^Q @ W^QR)                 # ROPE -> query_rope 输出
```

#### Key 计算路径
```
c^KV = RmsNorm(token_x @ W^DKV)        # 下采样 + RmsNorm -> kv latent
k^R = ROPE(token_x @ W^KR)             # ROPE -> key rope
```

## 参数说明

### 输入张量

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| token_x | (T, He) | BF16 | 输入 token，T 为 token 数量，He 为隐藏维度 |
| weight_dq | (He, Hcq) | BF16 | Query 下采样权重 |
| weight_uq_qr | (Hcq, N*(D+Dr)) | BF16 | Query 上采样和 ROPE 权重 |
| weight_uk | (N, D, Hckv) | BF16 | Query 最终上采样权重 |
| weight_dkv_kr | (He, Hckv+Dr) | BF16 | Key 下采样和 ROPE 权重 |
| rmsnorm_gamma_cq | (Hcq,) | BF16 | c^Q 的 RmsNorm gamma 参数 |
| rmsnorm_gamma_ckv | (Hckv,) | BF16 | c^KV 的 RmsNorm gamma 参数 |
| rope_sin | (T, Dr) | BF16 | ROPE 正弦参数 |
| rope_cos | (T, Dr) | BF16 | ROPE 余弦参数 |

### 输出张量

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| query | (T, N, Hckv) | BF16 | Query 输出 (q^N) |
| query_rope | (T, N, Dr) | BF16 | Query ROPE 输出 (q^R) |
| c_kv_out | (T, Hckv) | BF16 | KV latent 输出 |
| k_r_out | (T, Dr) | BF16 | Key ROPE 输出 |

### 维度说明

- **T**: Token 数量
- **He**: Hidden dimension (输入隐藏层维度)
- **Hcq**: Latent Query dimension
- **Hckv**: Latent KV dimension
- **N**: Number of heads (头数量)
- **D**: Dimension per head (每个头的维度)
- **Dr**: ROPE dimension per head (ROPE 维度)

## 编译运行

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=14
export PTO_TILE_LIB_CODE_PATH=/usr/local/Ascend/cann/aarch64-linux
```

### 编译 PyPTO

```bash
cd /data/x00952168/pypto_agent/pypto
python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto-*.whl --force-reinstall --no-deps
```

### 运行测试

```bash
python3 custom/mla_prolog/mla_prolog.py
```

## 测试结果

```
配置: T=8, He=256, Hcq=64, Hckv=32, N=4, D=16, Dr=8

输出形状:
  query: torch.Size([8, 4, 32])
  query_rope: torch.Size([8, 4, 8])
  c_kv_out: torch.Size([8, 32])
  k_r_out: torch.Size([8, 8])

精度对比 (与 Golden 最大差异):
  query: 0.500000
  query_rope: 0.125000
  c_kv_out: 0.015625
  k_r_out: 0.250000

✓ MLA Prolog 测试通过! (BF16精度容差)
```

## 实现要点

### 1. RmsNorm 实现
使用手动实现的 RmsNorm，避免精度损失：
- 计算 x^2 的均值
- 添加 epsilon 避免除零
- 归一化并乘以 gamma

### 2. ROPE 实现
对输入张量的相邻维度对进行旋转位置编码：
```python
对于维度对 (2i, 2i+1):
out[2i] = in[2i] * cos[i] - in[2i+1] * sin[i]
out[2i+1] = in[2i+1] * cos[i] + in[2i] * sin[i]
```

### 3. 权重拆分
- `weight_uq_qr` 拆分为 W^UQ 和 W^QR
- `weight_dkv_kr` 拆分为 W^DKV 和 W^KR

### 4. 多头处理
对每个 head 单独进行矩阵乘法，然后通过 concat 组合结果。

## 已知限制

1. 当前版本仅支持非量化场景
2. 仅支持 ND 格式输入
3. 不包含 Cache 更新功能（需要单独实现）
4. BF16 精度损失约 1/128，差异在预期范围内

## 文件说明

```
custom/mla_prolog/
├── mla_prolog.py              # PyPTO kernel 实现和 Golden 参考实现
└── README.md                  # 本文档
```