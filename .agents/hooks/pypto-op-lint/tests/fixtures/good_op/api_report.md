# good_op API 探索报告

## API 映射
pypto.sin -> torch.sin

## 约束说明
- 输入与输出 dtype 保持 float32
- 使用 PyPTO kernel 包装函数导出结果

## Tiling 说明
- 采用向量化 tile 切分
- 使用 set_vec_tile_shapes(8, 8)
