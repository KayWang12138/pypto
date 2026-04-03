# flash_attention 设计文档

## API 映射
- 使用 pypto.matmul 计算分块 attention score
- 使用 pypto.loop 串联 online softmax 状态更新

## 数据切分策略
- 沿 Skv 维度进行分块处理
- 通过向量化 tile 形状约束局部计算规模

## 验证方案
- 运行 test_flash_attention.py
- 使用 assert_allclose 对比 golden 输出
