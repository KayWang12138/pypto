# good_op 设计文档

## API 映射
- pypto.sin 对应 torch.sin

## 数据切分策略
向量化处理，使用 set_vec_tile_shapes(8, 8)

## 验证方案
- 运行 test_good_op.py
- 使用 assert_allclose 校验结果
