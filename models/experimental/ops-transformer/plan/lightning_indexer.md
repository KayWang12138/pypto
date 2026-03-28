# LightningIndexer 算子开发总结

## 开发状态：✅ 成功

## 算子信息
- 名称：LightningIndexer
- 功能：基于操作得到每个 token 对应的 Top-k 个位置
- 公式：Indices = Top-k(W ⊙ ReLU(Q_index @ K_index^T))

## 开发过程

### 1. 需求分析
- 阅读算子文档 `/mnt/workspace/gitCode/cann/ops-transformer/attention/lightning_indexer/README.md`
- 理解计算公式和数据流
- 确定 API 映射关系

### 2. API 查阅
- `pypto.matmul`: 矩阵乘法，支持 BF16 输入
- `pypto.relu`: ReLU 激活函数
- `pypto.mul`: 元素乘法（支持广播）
- `pypto.cast`: 类型转换
- `pypto.topk`: 获取 Top-k 索引（仅支持 FP32）
- `pypto.transpose`: 维度交换

### 3. 遇到的问题及解决方案

#### 问题 1: TileShape 维度不匹配
- **错误信息**: `TileShape dim num should same to input.`
- **原因**: weights 是 3D tensor，但使用了 4D TileShape
- **解决方案**: 在测试代码中预先将 weights transpose 并 unsqueeze 到 4D

#### 问题 2: 非连续 tensor
- **错误信息**: `not all tensors are contiguous`
- **原因**: PyTorch transpose 操作产生非连续 tensor
- **解决方案**: 调用 `.contiguous()` 使 tensor 连续

#### 问题 3: OoOSchedule pass 失败
- **错误信息**: `Run pass [OoOSchedule] failed.`
- **原因**: TileShape 设置不正确
- **解决方案**: 调整 TileShape 为 `pypto.set_vec_tile_shapes(1, 1, SEQ_LEN_Q, HEAD_DIM)` 和 `pypto.set_vec_tile_shapes(1, 1, SEQ_LEN_Q, SEQ_LEN_KV)`

## 测试结果

### 测试配置
- B=1, N=8, Sq=64, Skv=64, topk=8
- 数据类型：BF16
- 设备：NPU

### 测试输出
```
Input query shape: torch.Size([1, 64, 8, 128])
Input key shape: torch.Size([1, 64, 8, 128])
Input weights shape: torch.Size([1, 8, 64, 1])
Output indices shape: torch.Size([1, 64, 8, 8])
Exact match: True
Match ratio: 100.00% (4096/4096)
✓ LightningIndexer test completed
```

### 精度验证
- ✅ 与 PyTorch 参考实现完全一致
- ✅ 所有 4096 个索引值精确匹配

## 文件路径
- 算子实现：`/mnt/workspace/gitCode/cann/pypto/custom/lightning_indexer/lightning_indexer.py`
- 文档：`/mnt/workspace/gitCode/cann/pypto/custom/lightning_indexer/README.md`

## 已知限制
1. 当前实现仅支持静态 shape
2. weights tensor 需要预先处理为 [B, N, Sq, 1] 形状
3. topk 操作需要 FP32 数据类型，内部会进行类型转换

## 关键代码片段

```python
@pypto.frontend.jit
def lightning_indexer_kernel(
    query: pypto.Tensor((BATCH_SIZE, SEQ_LEN_Q, NUM_HEADS, HEAD_DIM), pypto.DT_BF16),
    key: pypto.Tensor((BATCH_SIZE, SEQ_LEN_KV, NUM_HEADS, HEAD_DIM), pypto.DT_BF16),
    weights: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1), pypto.DT_BF16),
    indices: pypto.Tensor((BATCH_SIZE, SEQ_LEN_Q, NUM_HEADS, TOPK), pypto.DT_INT32),
):
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 1, SEQ_LEN_Q, HEAD_DIM)
    
    query_t = pypto.transpose(query, 1, 2)
    key_t = pypto.transpose(key, 1, 2)
    
    scores = pypto.matmul(query_t, key_t, out_dtype=pypto.DT_BF16, b_trans=True)
    
    pypto.set_vec_tile_shapes(1, 1, SEQ_LEN_Q, SEQ_LEN_KV)
    scores_relu = pypto.relu(scores)
    
    weighted_scores = pypto.mul(scores_relu, weights)
    
    weighted_scores_fp32 = pypto.cast(weighted_scores, pypto.DT_FP32)
    
    _, topk_indices = pypto.topk(weighted_scores_fp32, TOPK, dim=-1, largest=True)
    
    indices_t = pypto.cast(topk_indices, pypto.DT_INT32)
    indices.move(pypto.transpose(indices_t, 1, 2))
```