# flash_attention API 探索报告

## API 映射
- pypto.matmul 对应 QK^T 与加权输出计算
- pypto.loop 对应分块 online softmax 迭代

## 约束说明
- mask 参与计算前需转换到 float32
- 中间累加结果应保留较高精度

## Tiling 说明
- 沿序列维度分块处理 key/value
- 使用 tile 形状控制局部计算与访存
