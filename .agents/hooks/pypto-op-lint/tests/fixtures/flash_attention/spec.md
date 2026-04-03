# flash_attention 算子需求规格

## 算子名
flash_attention

## 数学公式
attention_out = softmax((Q @ K^T) / sqrt(d) + mask) @ V

## 输入规格
- query: shape=[B, N, Sq, D]，dtype=bfloat16
- key: shape=[B, N, Skv, D]，dtype=bfloat16
- value: shape=[B, N, Skv, D]，dtype=bfloat16
- atten_mask: shape=[Sq, Skv]，dtype=float32

## 输出规格
- output: shape=[B, N, Sq, D]，dtype=bfloat16

## 精度要求
rtol=0.0078125, atol=0.0001
