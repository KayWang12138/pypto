# Flash Attention 算子开发计划

## 状态：开发中（遇到技术障碍）

## 已完成

1. ✅ 创建 golden 测试文件 `flash_attention_golden.py`
   - 包含 `ifa_flash_torch` 函数实现
   - 包含 `gen_block_table` 辅助函数
   - 包含基础测试用例

2. ✅ 分析 golden 代码逻辑
   - 理解 flash attention 的 online softmax 实现
   - 理解数据流和索引方式

## 遇到的技术障碍

### 1. 动态形状处理
- **问题描述**: `actual_s2_tile` 是运行时确定的值，pypto 对动态形状的切片支持有限
- **尝试方案**:
  - 使用 `pypto.view` 配合 `valid_shape` 参数
  - 使用静态形状简化问题
- **当前状态**: 部分解决，使用静态形状测试

### 2. TileShape 对齐要求
- **问题描述**: `amax` 和 `sum` 等归约操作要求最后一维的 TileShape 32字节对齐
- **约束条件**:
  - 尾轴要 32bytes 对齐
  - TileShape 维度应和输入 tensor 一致
  - TileShape 次尾轴要小于等于255
- **尝试方案**: 在每个操作前重新设置 TileShape
- **当前状态**: 已部分解决

### 3. 数据类型一致性
- **问题描述**: matmul 操作要求非 FP8 输入的数据类型一致
- **解决方案**: 在 matmul 前进行类型转换

### 4. Tensor 形状匹配
- **问题描述**: assemble 操作要求源和目标 shape 一致
- **解决方案**: 使用 unsqueeze 调整维度

## 当前实现状态

已实现简化版本的 flash attention kernel（仅支持 s2_loop=1 的情况），但编译过程中遇到 "Run pass failed" 错误，需要进一步调试。

## 下一步计划

1. 简化实现，先测试基本的矩阵乘法和 softmax 组合
2. 确认每个操作的 TileShape 设置
3. 逐步添加完整的 flash attention 逻辑
4. 支持 s2_loop > 1 的情况

## 关键代码位置

- Golden 实现: `custom/flash_attention/flash_attention_golden.py`
- PyPTO 实现: `custom/flash_attention/flash_attention.py`

## 参考资料

- PyPTO API 文档: `docs/api/`
- Flash Attention 论文: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness