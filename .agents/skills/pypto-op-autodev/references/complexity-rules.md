# 算子复杂度判定规则

## 复杂度定义

| 复杂度 | 判定条件 |
|--------|----------|
| `easy` | 单一 elementwise 操作；无规约；无 tiling 切分；无复杂内存访问；单 kernel 可完成 |
| `medium` | 涉及规约（sum/mean/max 等）；需要 tiling 切分；归一化类（含均值/方差）；需要多个中间变量 |
| `hard` | 涉及 attention 或 matmul；复杂内存访问模式（如 transpose + 计算）；需要多 pass 或多 kernel；FlashAttention 类在线算法 |

## 典型判定示例

| 算子 | 复杂度 | 理由 |
|------|--------|------|
| add, mul, relu, sigmoid | easy | 纯 elementwise |
| tanh_linear (y=tanh(x)*x) | easy | 两个 elementwise 组合 |
| sum, mean, max | medium | 含规约 |
| softmax | medium | 含规约（exp + sum + div）+ tiling |
| layernorm | medium | 含均值/方差计算 + tiling |
| rms_norm | medium | 含平方均值 + tiling |
| matmul | hard | 矩阵乘法 |
| scaled_dot_product_attention | hard | QKV matmul + softmax + matmul |
| flash_attention | hard | 在线算法 + 复杂内存访问 |

## 判定优先级

1. 算子名称含 `attention` / `matmul` / `gemm` / `conv` → `hard`
2. 算子含规约或归一化逻辑 → `medium`
3. 其余 elementwise 组合 → `easy`
4. 不确定时，取保守估计（宁高勿低）
