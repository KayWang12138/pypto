## 用户需求
本文档记录了用户开发算子的需求，请根据描述进行需求检测及分析。

---

## 需求-1
请开发一个pypto算子：
- 算子名称：sinh
- 公式：$ y = sinh(x) = (e^x - e^{-x}) / 2 $
- 规格：

| 类型  | shape  | dtype  |
| ------------ | ------------ | ------------ |
| 输入 x| \[b, s, n, d\]  | float32  |
| 输出 y| \[b, s, n, d\]  | float32  |

- 精度标准：atol=0.000025, rtol=0.005


## 需求-2
请开发一个pypto算子：
- 算子名称：matmul_add
- 公式：$ y(a, b, c) = a @ b^T + c $
- 规格：

| 类型  | shape  | dtype  |
| ------------ | ------------ | ------------ |
| 输入 a| \[m, k\] | bfloat16  |
| 输入 b| \[n, k\] | bfloat16  |
| 输入 c| \[m, n\] | bfloat16  |
| 输出 y| \[m, n\] | bfloat16  |

- 精度标准：atol=0.0001, rtol=0.0078125
- 其中要求m是动态轴


## 需求-3
请开发一个 pypto 算子：
- 算子名称：flash_attention_score
- 公式：$attention\_out = Softmax(\frac{Q @ K^T}{\sqrt{d}} \cdot mask) @ V$

  其中：
  - $Q$ 为 query 张量
  - $K$ 为 key 张量  
  - $V$ 为 value 张量
  - $d$ 为 head dimension
  - $mask$ 为注意力掩码（0表示参与计算，1表示不参与计算）

- 规格：

| 类型 | shape | dtype |
| ------------ | ------------ | ------------ |
| 输入 query | [B, N, Sq, D] | bfloat16 |
| 输入 key | [B, N, Skv, D] | bfloat16 |
| 输入 value | [B, N, Skv, D] | bfloat16 |
| 输入 atten_mask | [Sq, Skv] | uint8 |
| 输出 attention_out | [B, N, Sq, D] | bfloat16 |

- 精度标准：atol=0.0001, rtol=0.0078125
  
- 注意事项：
  - 运行时使用空闲卡
  - 掩码处理：atten_mask值为1的位置不参与计算，值为0的位置参与计算
  - 缩放因子 scale = 1/sqrt(D)
  - attention计算需要实现online softmax版本
  - 不需要实现 dropout 功能 与 动态轴
  - 认真阅读pypto/docs/api下有关接口使用的指导
  
- 完成开发后，逐项确认以下事项：
  - 公式、shape、dtype、精度标准全部符合规格
  - mask、缩放因子、online softmax全部实现
  - 所有报错已根本解决，未删减功能